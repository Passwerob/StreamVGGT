# StreamVGGT (RGB + Event Cross-Attention, Minimal Intrusion)

This repository keeps the original StreamVGGT RGB path intact and adds an optional RGB+Event token-level fusion path.

## What is added

- Optional event branch with lightweight patch embedding:
  - `EventPatchEmbed`: `Conv2d(in_chans=C_evt, out_chans=embed_dim, kernel_size=patch_size, stride=patch_size)`
  - Output token shape: `[B*S, N, C]`
- Optional event projection:
  - `event_proj = LayerNorm(embed_dim) + Linear(embed_dim, embed_dim)`
  - `event_tokens -> event_kv`
- Optional RGB-dominant cross-attention fusion:
  - `Q = rgb_patch`
  - `K = event_kv`
  - `V = event_kv`
  - Residual update: `fused_patch = rgb_patch + attn_out`

## Minimal-intrusion integration point

Fusion is inserted after RGB patch tokens are formed and before they are concatenated with StreamVGGT special tokens (camera/register) and sent into the original alternating-attention trunk.

This preserves the original downstream heads/loss/decoder flow.

## New optional parameters

### `StreamVGGT(...)`
- `fusion: str = "none"` (`"none" | "crossattn"`)
- `use_event: bool = False`
- `event_in_chans: int = 5`
- `event_patch_size: Optional[int] = None` (defaults to RGB `patch_size`)

### `forward(...)`
- `event_voxel: Optional[Tensor] = None` with expected shape `[B, S, C_evt, H, W]`
- `fusion: Optional[str] = None` (runtime override)
- `use_event: Optional[bool] = None` (runtime override)

### `inference(...)`
- `fusion: Optional[str] = None`
- `use_event: Optional[bool] = None`
- each frame dict can optionally include `"event_voxel"`

## Fallback behavior

- If `fusion != "crossattn"`, or `use_event=False`, or `event_voxel is None`, model follows original RGB path.
- If event token counts mismatch RGB patch token counts, an explicit error is raised (no silent pad/truncate/interpolate).

## Shapes

- RGB:
  - `images`: `[B, S, 3, H, W]`
  - reshape -> `[B*S, 3, H, W]`
  - `rgb_tokens`: `[B*S, N, C]`
- Event:
  - `event_voxel`: `[B, S, C_evt, H, W]`
  - reshape -> `[B*S, C_evt, H, W]`
  - `event_tokens`: `[B*S, N, C]`
  - `event_kv`: `[B*S, N, C]`
- Fusion:
  - input `rgb_tokens`: `[B*S, N, C]` (or `[B*S, N+1, C]` if cls token exists)
  - input `event_kv`: `[B*S, N, C]`
  - output `fused_tokens`: `[B*S, N, C]` (or `[B*S, N+1, C]`)

## Checkpoint compatibility

`StreamVGGT.load_state_dict` keeps strict loading attempt first and automatically falls back to `strict=False` for compatibility when new event/fusion modules are absent in old checkpoints.
