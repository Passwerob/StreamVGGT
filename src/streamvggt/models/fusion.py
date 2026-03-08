import torch
import torch.nn as nn


class EventPatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, event_voxel: torch.Tensor) -> torch.Tensor:
        _, _, h, w = event_voxel.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, (
            f"Event voxel H/W must be divisible by patch_size={self.patch_size}, got H={h}, W={w}"
        )
        event_tokens = self.proj(event_voxel)
        return event_tokens.flatten(2).transpose(1, 2)


class EventProj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, event_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(event_tokens))


class CrossAttnFuse(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(dim)
        self.event_norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, rgb_tokens: torch.Tensor, event_kv: torch.Tensor, h: int = None, w: int = None, patch: int = None):
        has_cls = rgb_tokens.shape[1] == event_kv.shape[1] + 1
        if has_cls:
            cls = rgb_tokens[:, :1, :]
            rgb_patch = rgb_tokens[:, 1:, :]
        else:
            cls = None
            rgb_patch = rgb_tokens

        n_rgb = rgb_patch.shape[1]
        n_evt = event_kv.shape[1]
        assert n_evt == n_rgb, (
            f"Token number mismatch for fusion: N_evt={n_evt}, N_rgb_patch={n_rgb}, H={h}, W={w}, patch={patch}"
        )

        attn_out, _ = self.mha(
            query=self.rgb_norm(rgb_patch),
            key=self.event_norm(event_kv),
            value=self.event_norm(event_kv),
            need_weights=False,
        )
        fused_patch = rgb_patch + attn_out
        if cls is not None:
            return torch.cat([cls, fused_patch], dim=1)
        return fused_patch
