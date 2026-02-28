import torch
import torch.nn as nn


class EventPatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, event_voxel: torch.Tensor) -> torch.Tensor:
        b, _, h, w = event_voxel.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise AssertionError(
                f"Event voxel spatial size must be divisible by patch_size={self.patch_size}, got H={h}, W={w}."
            )

        tokens = self.proj(event_voxel)
        tokens = tokens.flatten(2).transpose(1, 2)
        return tokens


class EventProj(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class CrossAttnFuse(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(dim)
        self.event_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, rgb_tokens: torch.Tensor, event_tokens: torch.Tensor) -> torch.Tensor:
        q = self.rgb_norm(rgb_tokens)
        kv = self.event_norm(event_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        return rgb_tokens + self.gate * attn_out
