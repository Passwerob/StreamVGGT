from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tvf
import torch.nn.functional as F

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class RGVIntervalFixed49Dataset(Dataset):
    """Fixed-length (49 frames) RGB+Event dataset for StreamVGGT training."""

    def __init__(
        self,
        data_root: str,
        split: str = "all",
        fixed_frames: int = 49,
        event_in_chans: int | None = None,
        resolution: tuple[int, int] | None = None,
    ):
        if fixed_frames != 49:
            raise ValueError(f"RGVIntervalFixed49Dataset only supports fixed_frames=49, got {fixed_frames}")
        if split not in {"all", "daylight", "night"}:
            raise ValueError(f"Unsupported split={split}, expected one of all/daylight/night")

        self.data_root = Path(data_root)
        self.split = split
        self.fixed_frames = fixed_frames
        self.event_in_chans = event_in_chans
        self.resolution = resolution  # (W, H), same convention as existing config

        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root does not exist: {self.data_root}")

        self.sequences = self._scan_sequences()
        if len(self.sequences) == 0:
            raise RuntimeError(f"No valid sequence found under {self.data_root} for split={self.split}")

    def _scan_sequences(self) -> List[Dict]:
        seqs: List[Dict] = []
        for seq_dir in sorted(self.data_root.glob("screen-*")):
            if not seq_dir.is_dir():
                continue
            name = seq_dir.name
            if self.split == "daylight" and not name.endswith("-daylight"):
                continue
            if self.split == "night" and not name.endswith("-night"):
                continue

            image_dir = seq_dir / "images"
            event_dir = seq_dir / "events"
            if not image_dir.is_dir() or not event_dir.is_dir():
                raise RuntimeError(f"Missing images/events directory in sequence: {seq_dir}")

            image_files = sorted(image_dir.glob("*.png"))
            event_files = sorted(event_dir.glob("*.pt"))
            if len(image_files) != self.fixed_frames or len(event_files) != self.fixed_frames:
                raise RuntimeError(
                    f"Sequence {name} must contain exactly {self.fixed_frames} images and events, "
                    f"got {len(image_files)} images and {len(event_files)} events"
                )

            for i in range(self.fixed_frames):
                frame_id = f"{i:06d}"
                image_path = image_dir / f"{frame_id}.png"
                event_path = event_dir / f"{frame_id}.pt"
                if not image_path.exists() or not event_path.exists():
                    raise RuntimeError(
                        f"Sequence {name} frame alignment error at {frame_id}: "
                        f"image_exists={image_path.exists()}, event_exists={event_path.exists()}"
                    )

            seqs.append(
                {
                    "seq_name": name,
                    "image_paths": [image_dir / f"{i:06d}.png" for i in range(self.fixed_frames)],
                    "event_paths": [event_dir / f"{i:06d}.pt" for i in range(self.fixed_frames)],
                    "frame_ids": [f"{i:06d}" for i in range(self.fixed_frames)],
                }
            )
        return seqs

    def __len__(self):
        return len(self.sequences)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.resolution is not None:
            img = img.resize(self.resolution, resample=Image.BICUBIC)
        return ImgNorm(img)  # reuse project default ToTensor + Normalize(0.5,0.5,0.5)

    def _load_event(self, path: Path, out_hw: tuple[int, int]) -> torch.Tensor:
        evt = torch.load(path, map_location="cpu")
        if not isinstance(evt, torch.Tensor):
            evt = torch.as_tensor(evt)
        evt = evt.float()

        if evt.ndim == 2:
            evt = evt.unsqueeze(0)
        if evt.ndim != 3:
            raise RuntimeError(f"Event tensor must have shape [C,H,W] or [H,W], got {tuple(evt.shape)} from {path}")

        if self.event_in_chans is not None and evt.shape[0] != self.event_in_chans:
            raise RuntimeError(
                f"Event channel mismatch at {path}: expected C={self.event_in_chans}, got C={evt.shape[0]}"
            )

        h, w = out_hw
        if evt.shape[-2:] != (h, w):
            evt = F.interpolate(
                evt.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return evt

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]

        rgb_frames: List[torch.Tensor] = []
        event_frames: List[torch.Tensor] = []

        for img_path, evt_path in zip(seq["image_paths"], seq["event_paths"]):
            rgb = self._load_image(img_path)
            event = self._load_event(evt_path, out_hw=(rgb.shape[-2], rgb.shape[-1]))
            rgb_frames.append(rgb)
            event_frames.append(event)

        rgb = torch.stack(rgb_frames, dim=0)  # [T, 3, H, W]
        event = torch.stack(event_frames, dim=0)  # [T, Cevt, H, W]

        return {
            "rgb": rgb,
            "event": event,
            "meta": {
                "seq_name": seq["seq_name"],
                "frame_ids": seq["frame_ids"],
            },
        }
