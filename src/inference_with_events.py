import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from streamvggt.models.streamvggt import StreamVGGT


"""
Minimal RGB+Event inference entrypoint.

Key point:
- We do NOT re-implement token logic in this script.
- We only pass `event_voxel` into `model.inference(...)` with `fusion='crossattn'`.
- Inside Aggregator, RGB patch tokens are replaced by fused tokens:
      patch_tokens = cross_attn_block(rgb_tokens, event_tokens)
  and all downstream heads consume those fused tokens.
"""


def parse_args():
    p = argparse.ArgumentParser("StreamVGGT inference with events (fused tokens)")
    p.add_argument("--weights", type=str, required=True, help="checkpoint path (.pth)")
    p.add_argument("--image_dir", type=str, required=True, help="RGB image directory")
    p.add_argument("--event_dir", type=str, required=True, help="event voxel .pt directory")
    p.add_argument("--output", type=str, default="inference_with_events_out.pt", help="output .pt path")
    p.add_argument("--export_dir", type=str, default="", help="optional directory to export visualized outputs")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resolution", type=int, nargs=2, default=[518, 518], metavar=("W", "H"))
    p.add_argument("--event_in_chans", type=int, default=8)
    p.add_argument("--max_frames", type=int, default=0, help="0 means all")
    p.add_argument(
        "--allow_random_fusion_init",
        action="store_true",
        help="allow running when fusion params are missing in checkpoint (otherwise raise error)",
    )
    p.add_argument("--save_rgb_png", action="store_true", help="export reconstructed RGB PNGs")
    p.add_argument("--save_depth_png", action="store_true", help="export normalized depth PNGs")
    p.add_argument("--save_ply", action="store_true", help="export point clouds as PLY (requires open3d)")
    return p.parse_args()


def _collect_pairs(image_dir: Path, event_dir: Path) -> List[Tuple[Path, Path]]:
    img_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    evt_map = {p.stem: p for p in sorted(event_dir.glob("*.pt"))}
    pairs = [(img, evt_map[img.stem]) for img in img_files if img.stem in evt_map]
    if not pairs:
        raise RuntimeError(f"No matched RGB/Event pairs in {image_dir} and {event_dir}")
    return pairs


def _load_rgb(path: Path, size_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    w, h = size_hw
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.squeeze(0).to(device)


def _load_event(path: Path, size_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    w, h = size_hw
    evt = torch.load(path, map_location="cpu")
    if not torch.is_tensor(evt):
        evt = torch.as_tensor(evt)
    evt = evt.float()
    if evt.ndim == 2:
        evt = evt.unsqueeze(0)
    elif evt.ndim == 4 and evt.shape[0] == 1:
        evt = evt.squeeze(0)
    if evt.ndim != 3:
        raise ValueError(f"Unsupported event shape {tuple(evt.shape)} @ {path}")
    evt = F.interpolate(evt.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
    return evt.squeeze(0).to(device)


def _extract_state_dict(ckpt: Dict) -> Dict:
    if not isinstance(ckpt, dict):
        return ckpt
    if "model" in ckpt:
        return ckpt["model"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


def _validate_loaded_keys(missing_keys: List[str], allow_random_fusion_init: bool):
    fusion_prefixes = ("aggregator.event_patch_embed.", "aggregator.cross_attn_block.")
    fusion_missing = [k for k in missing_keys if k.startswith(fusion_prefixes)]
    mae_missing = [k for k in missing_keys if k.startswith("mae_pred_head.")]
    recon_missing = [k for k in missing_keys if k.startswith("rgb_recon_head.")]
    other_missing = [k for k in missing_keys if k not in set(fusion_missing + mae_missing + recon_missing)]

    if missing_keys:
        print("[Load] missing key details:")
        if fusion_missing:
            print(f"  - fusion_missing ({len(fusion_missing)}):")
            for k in fusion_missing:
                print(f"      {k}")
        if mae_missing:
            print(f"  - mae_missing ({len(mae_missing)}): {mae_missing}")
        if recon_missing:
            print(f"  - rgb_recon_missing ({len(recon_missing)}): {recon_missing}")
        if other_missing:
            print(f"  - other_missing ({len(other_missing)}):")
            for k in other_missing:
                print(f"      {k}")

    if fusion_missing and not allow_random_fusion_init:
        raise RuntimeError(
            "Checkpoint is missing fusion weights, so fused inference would use random-initialized fusion modules. "
            "Please use a checkpoint trained with fusion enabled, or pass --allow_random_fusion_init to override."
        )


def _save_png(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _export_visualizations(results: List[dict], export_dir: Path, save_rgb_png: bool, save_depth_png: bool, save_ply: bool):
    export_dir.mkdir(parents=True, exist_ok=True)
    ply_enabled = save_ply
    o3d = None
    if save_ply:
        try:
            import open3d as o3d_module

            o3d = o3d_module
        except Exception as exc:  # optional dependency
            print(f"[Warn] open3d unavailable, skipping PLY export: {exc}")
            ply_enabled = False

    for i, res in enumerate(results):
        if save_rgb_png and "rgb" in res:
            rgb = res["rgb"]
            if torch.is_tensor(rgb):
                rgb = rgb.squeeze(0).numpy()
            rgb = np.clip(rgb, 0.0, 1.0)
            _save_png((rgb * 255.0).astype(np.uint8), export_dir / "rgb" / f"{i:05d}.png")

        if save_depth_png and "depth" in res:
            depth = res["depth"]
            if torch.is_tensor(depth):
                depth = depth.squeeze(0).numpy()
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            dmin, dmax = float(depth.min()), float(depth.max())
            if dmax > dmin:
                depth = (depth - dmin) / (dmax - dmin)
            else:
                depth = np.zeros_like(depth)
            _save_png((depth * 255.0).astype(np.uint8), export_dir / "depth" / f"{i:05d}.png")

        if ply_enabled and "pts3d_in_other_view" in res and "rgb" in res:
            pts = res["pts3d_in_other_view"]
            rgb = res["rgb"]
            if torch.is_tensor(pts):
                pts = pts.squeeze(0).reshape(-1, 3).numpy()
            if torch.is_tensor(rgb):
                rgb = rgb.squeeze(0).reshape(-1, 3).numpy()
            valid = np.isfinite(pts).all(axis=1)
            if valid.any():
                (export_dir / "ply").mkdir(parents=True, exist_ok=True)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts[valid])
                pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb[valid], 0.0, 1.0))
                o3d.io.write_point_cloud(str(export_dir / "ply" / f"{i:05d}.ply"), pcd)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Force fusion default path exactly as requested: fused_tokens replace rgb_tokens downstream.
    model = StreamVGGT(fusion="crossattn", event_in_chans=args.event_in_chans)
    ckpt = torch.load(args.weights, map_location="cpu")
    load_msg = model.load_state_dict(_extract_state_dict(ckpt), strict=False)
    model = model.to(device).eval()

    print("[Load] missing_keys:", len(load_msg.missing_keys))
    print("[Load] unexpected_keys:", len(load_msg.unexpected_keys))
    _validate_loaded_keys(load_msg.missing_keys, args.allow_random_fusion_init)

    pairs = _collect_pairs(Path(args.image_dir), Path(args.event_dir))
    if args.max_frames > 0:
        pairs = pairs[: args.max_frames]

    w, h = args.resolution
    if h % model.patch_size != 0 or w % model.patch_size != 0:
        raise ValueError(f"resolution {(w, h)} must be divisible by patch_size={model.patch_size}")

    frames = [
        {"img": _load_rgb(img, (w, h), device), "event_voxel": _load_event(evt, (w, h), device)}
        for img, evt in pairs
    ]

    with torch.no_grad():
        out = model.inference(frames)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "results": _to_cpu(out.ress),
            "num_frames": len(out.ress),
            "resolution": [w, h],
            "weights": args.weights,
            "fusion": "crossattn",
            "pairs": [{"rgb": str(i), "event": str(e)} for i, e in pairs],
        },
        output_path,
    )
    print(f"Saved {len(out.ress)} frames to {output_path}")

    if args.export_dir:
        _export_visualizations(
            _to_cpu(out.ress),
            Path(args.export_dir),
            save_rgb_png=args.save_rgb_png,
            save_depth_png=args.save_depth_png,
            save_ply=args.save_ply,
        )
        print(f"Exported visualization assets to {args.export_dir}")


if __name__ == "__main__":
    main()
