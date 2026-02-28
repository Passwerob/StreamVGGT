import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from streamvggt.models.aggregator import Aggregator
from streamvggt.models.fusion import get_rgb_tokens
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from transformers.file_utils import ModelOutput
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None


class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        fusion="crossattn",
        fusion_heads=8,
        fusion_mlp_ratio=4.0,
        event_in_chans=8,
        debug_fusion=False,
        mae_head="linear",
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            fusion=fusion,
            fusion_heads=fusion_heads,
            fusion_mlp_ratio=fusion_mlp_ratio,
            event_in_chans=event_in_chans,
            debug_fusion=debug_fusion,
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.rgb_recon_head = nn.Linear(2 * embed_dim, 3 * patch_size * patch_size)

        self.patch_size = patch_size
        self.fusion = fusion
        self.mae_head_type = mae_head
        if mae_head == "linear":
            self.mae_pred_head = nn.Linear(embed_dim, 3 * patch_size * patch_size)
        elif mae_head == "decoder":
            hidden = embed_dim * 2
            self.mae_pred_head = nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3 * patch_size * patch_size),
            )
        else:
            raise ValueError(f"Unknown mae_head: {mae_head}")

    def build_patch_mask(self, bsz: int, hp: int, wp: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
        n = hp * wp
        n_mask = int(round(mask_ratio * n))
        n_mask = min(max(n_mask, 1), n - 1)
        noise = torch.rand(bsz, n, device=device)
        ids = noise.argsort(dim=1)
        mask = torch.zeros(bsz, n, device=device, dtype=torch.bool)
        mask.scatter_(1, ids[:, :n_mask], True)
        return mask

    def patch_mask_to_pixel(self, mask_patch: torch.Tensor, hp: int, wp: int, patch: int) -> torch.Tensor:
        bsz = mask_patch.shape[0]
        mask = mask_patch.view(bsz, hp, wp).float()
        mask = mask.repeat_interleave(patch, dim=1).repeat_interleave(patch, dim=2)
        return mask.unsqueeze(1)

    def apply_mask(self, rgb: torch.Tensor, mask_pixel: torch.Tensor, fill: str = "mean") -> torch.Tensor:
        if fill == "zero":
            fill_value = torch.zeros_like(rgb)
        elif fill == "mean":
            fill_value = rgb.mean(dim=(2, 3), keepdim=True).expand_as(rgb)
        else:
            raise ValueError(f"Unknown mask_fill mode: {fill}")
        return rgb * (1.0 - mask_pixel) + fill_value * mask_pixel

    def patchify(self, rgb: torch.Tensor) -> torch.Tensor:
        bsz, c, h, w = rgb.shape
        p = self.patch_size
        assert h % p == 0 and w % p == 0, f"patchify requires H,W divisible by patch_size, got H={h}, W={w}, p={p}"
        hp, wp = h // p, w // p
        x = rgb.reshape(bsz, c, hp, p, wp, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(bsz, hp * wp, p * p * c)
        return x

    def unpatchify(self, patches: torch.Tensor, h: int, w: int) -> torch.Tensor:
        bsz, n, dim = patches.shape
        p = self.patch_size
        c = 3
        hp, wp = h // p, w // p
        assert hp * wp == n, f"unpatchify mismatch: n={n}, hp*wp={hp*wp}, H={h}, W={w}, p={p}"
        x = patches.reshape(bsz, hp, wp, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(bsz, c, h, w)
        return x

    def _extract_backbone_tokens_with_cls(self, rgb_normed: torch.Tensor):
        backbone = self.aggregator.patch_embed
        out = backbone.forward_features(rgb_normed) if hasattr(backbone, "forward_features") else backbone(rgb_normed)
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            patch_tokens = out["x_norm_patchtokens"]
            cls_token = out.get("x_norm_clstoken", None)
            if cls_token is None:
                cls_token = torch.zeros(patch_tokens.shape[0], 1, patch_tokens.shape[-1], device=patch_tokens.device, dtype=patch_tokens.dtype)
            else:
                cls_token = cls_token.unsqueeze(1) if cls_token.ndim == 2 else cls_token
            return torch.cat([cls_token, patch_tokens], dim=1), patch_tokens

        patch_tokens = get_rgb_tokens(backbone, rgb_normed)
        cls_token = torch.zeros(patch_tokens.shape[0], 1, patch_tokens.shape[-1], device=patch_tokens.device, dtype=patch_tokens.dtype)
        return torch.cat([cls_token, patch_tokens], dim=1), patch_tokens

    def mae_forward(
        self,
        rgb: torch.Tensor,
        event_voxel: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.6,
        mask_fill: str = "mean",
        mae_loss: str = "mse",
        fusion: Optional[str] = None,
        return_images: bool = False,
    ):
        bsz, c, h, w = rgb.shape
        p = self.patch_size
        assert h % p == 0 and w % p == 0, f"H/W must be divisible by patch size, got H={h}, W={w}, patch={p}"
        hp, wp = h // p, w // p

        mask_patch = self.build_patch_mask(bsz, hp, wp, mask_ratio, rgb.device)
        mask_pixel = self.patch_mask_to_pixel(mask_patch, hp, wp, p)
        rgb_masked = self.apply_mask(rgb, mask_pixel, fill=mask_fill)

        mean = self.aggregator._resnet_mean.to(rgb.device)
        std = self.aggregator._resnet_std.to(rgb.device)
        rgb_masked_normed = (rgb_masked - mean[:, 0]) / std[:, 0]

        x_rgb_full, x_patch = self._extract_backbone_tokens_with_cls(rgb_masked_normed)

        fusion_mode = self.fusion if fusion is None else fusion
        if fusion_mode == "crossattn":
            if event_voxel is None:
                raise ValueError("train_mode=mae with fusion=crossattn requires event_voxel")
            if event_voxel.ndim != 4:
                raise ValueError(f"event_voxel must be [B,Cevt,H,W], got {event_voxel.shape}")
            be, _, he, we = event_voxel.shape
            assert be == bsz and he == h and we == w, (
                f"Event/RGB size mismatch: rgb=({bsz},{h},{w}), event=({be},{he},{we}), patch={p}, tokenN={x_patch.shape[1]}"
            )
            x_evt = self.aggregator.event_patch_embed(event_voxel)
            assert x_evt.shape[1] == x_patch.shape[1], (
                f"Token mismatch in MAE fusion: rgb={x_patch.shape}, evt={x_evt.shape}, H={h}, W={w}, patch={p}"
            )
            x_patch = self.aggregator.cross_attn_block(x_patch, x_evt)
        elif fusion_mode != "none":
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")

        pred_patches = self.mae_pred_head(x_patch)
        target_patches = self.patchify(rgb)

        mask_expand = mask_patch.unsqueeze(-1).expand_as(pred_patches)
        pred_masked = pred_patches[mask_expand]
        tgt_masked = target_patches[mask_expand]

        if mae_loss == "mse":
            loss = F.mse_loss(pred_masked, tgt_masked)
        elif mae_loss == "l1":
            loss = F.l1_loss(pred_masked, tgt_masked)
        else:
            raise ValueError(f"Unknown mae_loss: {mae_loss}")

        out = {
            "loss": loss,
            "mask": mask_patch,
            "gate": self.aggregator.cross_attn_block.gate.detach().float(),
        }
        if return_images:
            out["pred_rgb"] = self.unpatchify(pred_patches, h, w)
            out["rgb_masked"] = rgb_masked
            out["rgb"] = rgb
        return out

    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
        event_voxels: Optional[torch.Tensor] = None,
    ):
        images = torch.stack([view["img"] for view in views], dim=0).permute(1, 0, 2, 3, 4)

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        if event_voxels is None and isinstance(views[0], dict) and "event_voxel" in views[0]:
            event_voxels = torch.stack([view["event_voxel"] for view in views], dim=0).permute(1, 0, 2, 3, 4)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images, event_voxels=event_voxels, fusion=self.fusion)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            predictions["rgb_recon"] = self._decode_rgb(aggregated_tokens_list[-1], patch_start_idx, images)

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]
                predictions["vis"] = vis
                predictions["conf"] = conf

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    "pts3d_in_other_view": predictions["world_points"][:, s],
                    "conf": predictions["world_points_conf"][:, s],
                    "depth": predictions["depth"][:, s],
                    "depth_conf": predictions["depth_conf"][:, s],
                    "camera_pose": predictions["pose_enc"][:, s, :],
                    "rgb": predictions["rgb_recon"][:, s],
                    **({"valid_mask": views[s]["valid_mask"]} if "valid_mask" in views[s] else {}),
                    **(
                        {
                            "track": predictions["track"][:, s],
                            "vis": predictions["vis"][:, s],
                            "track_conf": predictions["conf"][:, s],
                        }
                        if "track" in predictions
                        else {}
                    ),
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)

    def _decode_rgb(self, tokens, patch_start_idx, images):
        b, s, _, _ = tokens.shape
        _, _, _, h, w = images.shape
        hp, wp = h // self.patch_size, w // self.patch_size
        patch_tokens = tokens[:, :, patch_start_idx:, :]
        if patch_tokens.shape[2] != hp * wp:
            raise ValueError(
                f"RGB decode token mismatch: tokens={patch_tokens.shape[2]}, grid={hp}x{wp}, H={h}, W={w}, patch={self.patch_size}"
            )
        patch_rgb = self.rgb_recon_head(patch_tokens)
        patch_rgb = patch_rgb.view(b, s, hp, wp, 3, self.patch_size, self.patch_size)
        rgb = patch_rgb.permute(0, 1, 2, 5, 3, 6, 4).reshape(b, s, h, w, 3).contiguous()
        return rgb

    def inference(self, frames, query_points: torch.Tensor = None, past_key_values=None):
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth

        all_ress = []
        processed_frames = []

        for i, frame in enumerate(frames):
            images = frame["img"].unsqueeze(0)
            event_voxels = frame.get("event_voxel", None)
            if event_voxels is not None:
                if event_voxels.ndim == 3:
                    event_voxels = event_voxels.unsqueeze(0).unsqueeze(0)
                elif event_voxels.ndim == 4:
                    event_voxels = event_voxels.unsqueeze(0)
            aggregator_output = self.aggregator(
                images,
                event_voxels=event_voxels,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=i,
                fusion=self.fusion,
            )

            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output

            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(
                        aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True
                    )
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(aggregated_tokens, images=images, patch_start_idx=patch_start_idx)
                    depth = depth[:, 0]
                    depth_conf = depth_conf[:, 0]

                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(aggregated_tokens, images=images, patch_start_idx=patch_start_idx)
                    pts3d = pts3d[:, 0]
                    pts3d_conf = pts3d_conf[:, 0]

                rgb_recon = self._decode_rgb(aggregated_tokens[-1], patch_start_idx, images)[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                    )
                    track = track_list[-1][:, 0]
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            all_ress.append(
                {
                    "pts3d_in_other_view": pts3d,
                    "conf": pts3d_conf,
                    "depth": depth,
                    "depth_conf": depth_conf,
                    "camera_pose": camera_pose,
                    "rgb": rgb_recon,
                    **({"valid_mask": frame["valid_mask"]} if "valid_mask" in frame else {}),
                    **(
                        {"track": track, "vis": vis, "track_conf": track_conf}
                        if query_points is not None
                        else {}
                    ),
                }
            )
            processed_frames.append(frame)

        output = StreamVGGTOutput(ress=all_ress, views=processed_frames)
        return output
