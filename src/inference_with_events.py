import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from streamvggt.models.streamvggt import StreamVGGT


def parse_args():
    p = argparse.ArgumentParser("Inference with RGB+Event for StreamVGGT")
    p.add_argument("--weights", type=str, required=True, help="Path to checkpoint-final.pth or raw state_dict .pth")
    p.add_argument("--image_dir", type=str, required=True, help="Directory with RGB .png/.jpg frames")
    p.add_argument("--event_dir", type=str, required=True, help="Directory with event voxel .pt frames")
    p.add_argument("--output", type=str, default="inference_with_events_out", help="Output directory")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fusion", type=str, default="crossattn", choices=["none", "crossattn"])
    p.add_argument("--event_in_chans", type=int, default=8)
    p.add_argument("--resolution", type=int, nargs=2, default=[518, 518], metavar=("W", "H"))
    p.add_argument("--max_frames", type=int, default=0, help="0 means all")
    return p.parse_args()


def _collect_pairs(image_dir: Path, event_dir: Path) -> List[Tuple[Path, Path]]:
    img_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    evt_map = {p.stem: p for p in sorted(event_dir.glob("*.pt"))}
    pairs = []
    for img in img_files:
        evt = evt_map.get(img.stem)
        if evt is not None:
            pairs.append((img, evt))
    if not pairs:
        raise RuntimeError(f"No matched RGB/Event pairs found in {image_dir} and {event_dir}")
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
        raise ValueError(f"Unsupported event tensor shape: {tuple(evt.shape)} @ {path}")
    evt = F.interpolate(evt.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
    return evt.squeeze(0).to(device)


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = StreamVGGT(fusion=args.fusion, event_in_chans=args.event_in_chans)
    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_msg = model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    image_dir = Path(args.image_dir)
    event_dir = Path(args.event_dir)
    pairs = _collect_pairs(image_dir, event_dir)
    if args.max_frames > 0:
        pairs = pairs[: args.max_frames]

    w, h = args.resolution
    if h % model.patch_size != 0 or w % model.patch_size != 0:
        raise ValueError(f"resolution {(w, h)} must be divisible by patch_size={model.patch_size}")

    frames = []
    for img_path, evt_path in pairs:
        frames.append({
            "img": _load_rgb(img_path, (w, h), device),
            "event_voxel": _load_event(evt_path, (w, h), device),
        })

    with torch.no_grad():
        out = model.inference(frames)

    meta = {
        "weights": args.weights,
        "fusion": args.fusion,
        "missing_keys": load_msg.missing_keys,
        "unexpected_keys": load_msg.unexpected_keys,
        "num_frames": len(out.ress),
        "resolution": [w, h],
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    for i, res in enumerate(out.ress):
        save = {}
        for k in ["depth", "depth_conf", "camera_pose", "rgb", "conf", "pts3d_in_other_view"]:
            if k in res:
                v = res[k]
                if torch.is_tensor(v):
                    save[k] = v.detach().cpu().numpy()
        np.savez_compressed(output_dir / f"frame_{i:05d}.npz", **save)

    print(f"Saved {len(out.ress)} frames to {output_dir}")


if __name__ == "__main__":
    main()
