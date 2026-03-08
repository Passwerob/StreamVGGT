import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.utils.rotation import mat_to_quat


def parse_args():
    p = argparse.ArgumentParser("StreamVGGT event-fusion inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True, help="sequence root containing images/ and events/")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--fusion", type=str, default="crossattn", choices=["none", "crossattn"])
    p.add_argument("--event_in_chans", type=int, default=8)
    p.add_argument("--fusion_heads", type=int, default=8)
    p.add_argument("--img_size", type=int, default=518)
    p.add_argument("--patch_size", type=int, default=14)
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--strict_load", action="store_true")
    p.add_argument("--autocast", type=str, default="auto", choices=["off", "auto", "bf16", "fp16"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--conf_threshold", type=float, default=0.0)
    p.add_argument("--max_frames", type=int, default=-1)
    p.add_argument(
        "--resize_hw",
        type=int,
        nargs=2,
        default=[154, 266],
        metavar=("H", "W"),
        help="Fixed resize resolution (H W) before inference, should match training resolution policy.",
    )
    return p.parse_args()


def choose_dtype(mode: str, device: torch.device):
    if mode == "off" or device.type != "cuda":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    # auto
    bf16_ok = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if bf16_ok else torch.float16


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint is not a state_dict-like object")
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def load_sequence(data_root: Path, event_in_chans: int, resize_hw: Tuple[int, int], patch_size: int, max_frames: int = -1):
    image_dir = data_root / "images"
    event_dir = data_root / "events"
    if not image_dir.is_dir() or not event_dir.is_dir():
        raise FileNotFoundError(f"Expect {data_root}/images and {data_root}/events")

    image_paths = sorted(image_dir.glob("*.png"))
    items = []
    for ip in image_paths:
        ep = event_dir / f"{ip.stem}.pt"
        if not ep.exists():
            continue
        items.append((ip, ep))

    if len(items) == 0:
        raise RuntimeError("No aligned image/event pairs found")

    if max_frames > 0:
        items = items[:max_frames]

    views = []
    frame_names = []
    out_h, out_w = int(resize_hw[0]), int(resize_hw[1])
    if out_h % patch_size != 0 or out_w % patch_size != 0:
        raise ValueError(
            f"resize_hw {(out_h, out_w)} must be divisible by patch_size={patch_size}; "
            f"got H%patch={out_h % patch_size}, W%patch={out_w % patch_size}"
        )

    for ip, ep in items:
        img = Image.open(ip).convert("RGB")
        img = img.resize((out_w, out_h), resample=Image.BICUBIC)
        img_t = torch.from_numpy(np.array(img)).float() / 255.0
        img_t = img_t.permute(2, 0, 1).contiguous()  # [3,H,W], 0..1

        evt = torch.load(ep, map_location="cpu")
        if not isinstance(evt, torch.Tensor):
            evt = torch.as_tensor(evt)
        evt = evt.float()
        if evt.ndim == 2:
            evt = evt.unsqueeze(0)
        if evt.ndim != 3:
            raise RuntimeError(f"Bad event shape at {ep}: {tuple(evt.shape)}")
        if evt.shape[0] != event_in_chans:
            raise RuntimeError(f"event_in_chans mismatch at {ep}: expected {event_in_chans}, got {evt.shape[0]}")
        if evt.shape[-2:] != img_t.shape[-2:]:
            evt = F.interpolate(
                evt.unsqueeze(0),
                size=img_t.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        views.append({"img": img_t.unsqueeze(0), "event_voxel": evt.unsqueeze(0)})
        frame_names.append(ip.name)

    return views, frame_names


def save_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, conf: np.ndarray):
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float confidence\n")
        f.write("end_header\n")
        for i in range(xyz.shape[0]):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            c = conf[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {c:.6f}\n")


def write_colmap(cameras_dir: Path, frame_names: List[str], extrinsics: np.ndarray, intrinsics: np.ndarray, h: int, w: int):
    colmap_dir = cameras_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    with (colmap_dir / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i in range(len(frame_names)):
            fx = intrinsics[i, 0, 0]
            fy = intrinsics[i, 1, 1]
            cx = intrinsics[i, 0, 2]
            cy = intrinsics[i, 1, 2]
            f.write(f"{i+1} PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    with (colmap_dir / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, name in enumerate(frame_names):
            R = torch.from_numpy(extrinsics[i, :3, :3]).float()
            q_xyzw = mat_to_quat(R).cpu().numpy()
            qx, qy, qz, qw = q_xyzw.tolist()
            tx, ty, tz = extrinsics[i, :3, 3].tolist()
            f.write(f"{i+1} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {tx:.8f} {ty:.8f} {tz:.8f} {i+1} {name}\n\n")

    with (colmap_dir / "points3D.txt").open("w", encoding="utf-8") as f:
        f.write("# 3D point list\n")


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out = Path(args.output)
    (out / "point_cloud").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "cameras").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    model = StreamVGGT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        fusion=args.fusion,
        event_in_chans=args.event_in_chans,
        fusion_heads=args.fusion_heads,
    ).to(device)

    sd = load_checkpoint(args.checkpoint)
    msg = model.load_state_dict(sd, strict=args.strict_load)
    if not args.strict_load:
        print(f"[load] missing_keys={len(msg.missing_keys)}, unexpected_keys={len(msg.unexpected_keys)}")
        if msg.missing_keys:
            print("[load] missing (first 20):", msg.missing_keys[:20])
        if msg.unexpected_keys:
            print("[load] unexpected (first 20):", msg.unexpected_keys[:20])

    model.eval()

    views, frame_names = load_sequence(
        Path(args.data_root),
        args.event_in_chans,
        tuple(args.resize_hw),
        args.patch_size,
        args.max_frames,
    )
    autocast_dtype = choose_dtype(args.autocast, device)

    merged_xyz, merged_rgb, merged_conf = [], [], []
    all_ext, all_int = [], []

    with torch.no_grad():
        for idx, (view, frame_name) in enumerate(zip(views, frame_names)):
            for k in view:
                view[k] = view[k].to(device, non_blocking=True)

            if autocast_dtype is None:
                output = model([view], None)
            else:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    output = model([view], None)

            pred = output.ress[0]
            pts = pred["pts3d_in_other_view"][0].detach().cpu().numpy()  # [H,W,3]
            conf = pred["conf"][0].detach().cpu().numpy()  # [H,W]
            depth = pred["depth"][0].detach().cpu().numpy().squeeze(-1)  # [H,W]
            pose = pred["camera_pose"].unsqueeze(1)  # [1,1,9]

            H, W = depth.shape
            ext, intr = pose_encoding_to_extri_intri(pose, image_size_hw=(H, W))
            ext4 = np.eye(4, dtype=np.float32)
            ext4[:3, :4] = ext[0, 0].detach().cpu().numpy()
            intr3 = intr[0, 0].detach().cpu().numpy().astype(np.float32)

            all_ext.append(ext4)
            all_int.append(intr3)

            input_rgb = view["img"][0].detach().cpu().permute(1, 2, 0).numpy()  # [H,W,3] in [0,1]
            rgb_u8 = np.clip(input_rgb * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(rgb_u8).save(out / "images" / f"frame_{idx:06d}.png")

            dmin = float(np.nanmin(depth))
            dmax = float(np.nanmax(depth))
            np.save(out / "depth" / f"frame_{idx:06d}.npy", depth.astype(np.float32))
            if dmax > dmin:
                dnorm = (depth - dmin) / (dmax - dmin)
            else:
                dnorm = np.zeros_like(depth, dtype=np.float32)
            d16 = np.clip(dnorm * 65535.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(out / "depth" / f"frame_{idx:06d}.png"), d16)
            dvis = cv2.applyColorMap((dnorm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.imwrite(str(out / "depth" / f"frame_{idx:06d}_vis.png"), dvis)
            with (out / "depth" / f"frame_{idx:06d}_meta.json").open("w", encoding="utf-8") as f:
                json.dump({"min": dmin, "max": dmax}, f)

            np.savetxt(out / "cameras" / f"frame_{idx:06d}_extrinsic.txt", ext4, fmt="%.8f")
            np.savetxt(out / "cameras" / f"frame_{idx:06d}_intrinsic.txt", intr3, fmt="%.8f")

            xyz = pts.reshape(-1, 3)
            conf_flat = conf.reshape(-1)
            rgb_flat = rgb_u8.reshape(-1, 3)
            keep = conf_flat >= args.conf_threshold
            xyz_k = xyz[keep]
            conf_k = conf_flat[keep]
            rgb_k = rgb_flat[keep]

            save_ascii_ply(out / "point_cloud" / f"frame_{idx:06d}.ply", xyz_k, rgb_k, conf_k)
            merged_xyz.append(xyz_k)
            merged_conf.append(conf_k)
            merged_rgb.append(rgb_k)

    ext_np = np.stack(all_ext, axis=0)
    int_np = np.stack(all_int, axis=0)
    np.savez(out / "cameras" / "cameras.npz", extrinsics=ext_np, intrinsics=int_np, frame_names=np.array(frame_names))

    write_colmap(out / "cameras", frame_names, ext_np, int_np, H, W)

    merged_xyz_np = np.concatenate(merged_xyz, axis=0) if merged_xyz else np.zeros((0, 3), dtype=np.float32)
    merged_rgb_np = np.concatenate(merged_rgb, axis=0) if merged_rgb else np.zeros((0, 3), dtype=np.uint8)
    merged_conf_np = np.concatenate(merged_conf, axis=0) if merged_conf else np.zeros((0,), dtype=np.float32)
    save_ascii_ply(out / "point_cloud" / "merged.ply", merged_xyz_np, merged_rgb_np, merged_conf_np)

    # nerf/3dgs style transforms.json
    frames = []
    for i, name in enumerate(frame_names):
        c2w = np.linalg.inv(ext_np[i]).astype(np.float32)
        frames.append(
            {
                "file_path": f"images/frame_{i:06d}.png",
                "transform_matrix": c2w.tolist(),
                "fl_x": float(int_np[i, 0, 0]),
                "fl_y": float(int_np[i, 1, 1]),
                "cx": float(int_np[i, 0, 2]),
                "cy": float(int_np[i, 1, 2]),
                "w": int(W),
                "h": int(H),
            }
        )

    transforms = {
        "camera_model": "OPENCV",
        "frames": frames,
    }
    with (out / "transforms.json").open("w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    print(f"Done. Outputs saved to: {out}")


if __name__ == "__main__":
    main()
