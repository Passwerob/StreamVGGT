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
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resolution", type=int, nargs=2, default=[518, 518], metavar=("W", "H"))
    p.add_argument("--event_in_chans", type=int, default=8)
    p.add_argument("--max_frames", type=int, default=0, help="0 means all")
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


if __name__ == "__main__":
    main()
