import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F


class RGVEventMAE(Dataset):
    """Minimal RGB+Event dataset for MAE training.

    Expected tree:
      ROOT/screen-xxxxx-daylight|night/{images,events}

    Each sample is an (image, event_voxel) pair matched by sorted filename stem.
    """

    def __init__(
        self,
        ROOT: str,
        split: str = "train",
        resolution: Tuple[int, int] = (518, 518),
        val_ratio: float = 0.05,
        seed: int = 42,
        num_views: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(ROOT)
        self.split = split
        self.width, self.height = int(resolution[0]), int(resolution[1])
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.num_views = int(num_views)

        if not self.root.exists():
            raise FileNotFoundError(f"RGVEventMAE ROOT not found: {self.root}")
        if self.num_views != 1:
            raise ValueError(f"RGVEventMAE currently supports num_views=1, got {self.num_views}")

        self.samples = self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No paired samples found under {self.root}")

    def _build_samples(self) -> List[Tuple[Path, Path]]:
        scene_dirs = [p for p in sorted(self.root.iterdir()) if p.is_dir()]
        all_samples: List[Tuple[Path, Path]] = []

        for scene in scene_dirs:
            image_dir = scene / "images"
            event_dir = scene / "events"
            if not image_dir.is_dir() or not event_dir.is_dir():
                continue

            img_map = {p.stem: p for p in sorted(image_dir.glob("*.png"))}
            evt_map = {p.stem: p for p in sorted(event_dir.glob("*.pt"))}
            common = sorted(set(img_map.keys()) & set(evt_map.keys()))
            all_samples.extend([(img_map[k], evt_map[k]) for k in common])

        if self.split in {"train", "val"}:
            rng = np.random.default_rng(self.seed)
            idx = np.arange(len(all_samples))
            rng.shuffle(idx)
            n_val = max(1, int(len(idx) * self.val_ratio))
            val_idx = set(idx[:n_val].tolist())
            if self.split == "train":
                all_samples = [s for i, s in enumerate(all_samples) if i not in val_idx]
            else:
                all_samples = [s for i, s in enumerate(all_samples) if i in val_idx]

        return all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_img(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        x = F.interpolate(x.unsqueeze(0), size=(self.height, self.width), mode="bilinear", align_corners=False).squeeze(0)
        x = x * 2.0 - 1.0  # train.py maps back to [0,1]
        return x

    def _load_event(self, path: Path) -> torch.Tensor:
        evt = torch.load(path, map_location="cpu")
        if not torch.is_tensor(evt):
            evt = torch.as_tensor(evt)
        evt = evt.float()

        # make channel-first [C,H,W]
        if evt.ndim == 2:
            evt = evt.unsqueeze(0)
        elif evt.ndim == 4 and evt.shape[0] == 1:
            evt = evt.squeeze(0)
        elif evt.ndim == 4 and evt.shape[-1] in {1, 2, 4, 8, 16, 32, 49}:
            evt = evt.permute(3, 0, 1, 2).reshape(-1, evt.shape[1], evt.shape[2])

        if evt.ndim != 3:
            raise ValueError(f"Unsupported event shape {tuple(evt.shape)} in {path}")

        evt = F.interpolate(evt.unsqueeze(0), size=(self.height, self.width), mode="bilinear", align_corners=False).squeeze(0)
        return evt

    def __getitem__(self, index: int):
        img_path, evt_path = self.samples[index]
        sample = {
            "img": self._load_img(img_path),
            "event_voxel": self._load_event(evt_path),
            "instance": str(img_path),
        }
        return sample
