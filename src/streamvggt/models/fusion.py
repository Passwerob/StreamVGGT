import torch
import torch.nn as nn


class EventPatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"EventPatchEmbed expects 4D [B,C,H,W], got {x.shape}")
        b, c, _, _ = x.shape
        if c != self.in_chans:
            raise ValueError(f"EventPatchEmbed channel mismatch: expected {self.in_chans}, got {c}")
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))

        hidden_dim = int(dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x_rgb: torch.Tensor, x_evt: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(x_rgb)
        kv = self.norm_kv(x_evt)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = x_rgb + self.gate * attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


def get_rgb_tokens(backbone: nn.Module, rgb: torch.Tensor) -> torch.Tensor:
    if hasattr(backbone, "forward_features"):
        out = backbone.forward_features(rgb)
    else:
        out = backbone(rgb)

    if isinstance(out, dict):
        if "x_norm_patchtokens" in out:
            return out["x_norm_patchtokens"]
        for _, value in out.items():
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                return value

    if isinstance(out, (tuple, list)):
        for value in out:
            if isinstance(value, dict) and "x_norm_patchtokens" in value:
                return value["x_norm_patchtokens"]
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                return value

    if isinstance(out, torch.Tensor) and out.ndim == 3:
        return out

    if hasattr(backbone, "get_intermediate_layers"):
        layers = backbone.get_intermediate_layers(rgb, n=1)
        if isinstance(layers, (tuple, list)) and len(layers) > 0:
            val = layers[-1]
            if isinstance(val, torch.Tensor):
                return val
            if isinstance(val, (tuple, list)) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                return val[0]

    raise RuntimeError(f"Unable to extract rgb patch tokens from backbone output type={type(out)}")
