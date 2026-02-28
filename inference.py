import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None

sys.path.append("src/")

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri


class StreamVGGTInference:
    """StreamVGGT inference wrapper for images or video input."""

    def __init__(
        self,
        checkpoint_path: str = "ckpt/checkpoints.pth",
        device: str = None,
        fusion: str = "none",
        event_in_chans: int = 8,
        fusion_heads: int = 8,
        freeze_backbone: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(
            checkpoint_path=checkpoint_path,
            fusion=fusion,
            event_in_chans=event_in_chans,
            fusion_heads=fusion_heads,
            freeze_backbone=freeze_backbone,
        )

    def _load_model(
        self,
        checkpoint_path: str,
        fusion: str,
        event_in_chans: int,
        fusion_heads: int,
        freeze_backbone: bool,
    ) -> StreamVGGT:
        checkpoint_path = os.path.expanduser(checkpoint_path.strip())
        print(f"Loading model from {checkpoint_path}...")

        model = StreamVGGT(
            fusion=fusion,
            event_in_chans=event_in_chans,
            fusion_heads=fusion_heads,
            freeze_backbone=freeze_backbone,
        )

        if os.path.exists(checkpoint_path):
            ckpt_path = checkpoint_path
        else:
            print("Local checkpoint not found, downloading from Hugging Face...")
            from huggingface_hub import hf_hub_download

            ckpt_path = hf_hub_download(
                repo_id="lch01/StreamVGGT",
                filename="checkpoints.pth",
                revision="main",
                force_download=True,
            )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        del ckpt

        model.to(self.device)
        model.eval()
        print(f"Model loaded on {self.device}")
        return model

    def _extract_frames_from_video(self, video_path: str, output_dir: str, fps_interval: float = 1.0) -> List[str]:
        if cv2 is None:
            raise ImportError("OpenCV is required for video inference. Please install opencv-python.")
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        frame_interval = max(int(round(fps * fps_interval)), 1)

        frame_paths = []
        count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"{frame_idx:06d}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                frame_idx += 1

        cap.release()
        print(f"Extracted {len(frame_paths)} frames from video")
        return sorted(frame_paths)

    def _get_image_paths(self, image_folder: str) -> List[str]:
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
        image_paths = sorted(list(set(image_paths)))
        print(f"Found {len(image_paths)} images")
        return image_paths

    @torch.no_grad()
    def inference(self, image_paths: List[str]) -> Tuple[dict, List[str]]:
        if len(image_paths) == 0:
            raise ValueError("No images provided")

        images = load_and_preprocess_images(image_paths).to(self.device)  # [S, 3, H, W]
        print(f"Input images shape: {tuple(images.shape)}")

        frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]

        if self.device.startswith("cuda") and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(torch.device(self.device))
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
                output = self.model.inference(frames)
        else:
            output = self.model.inference(frames)

        all_pts3d, all_conf, all_depth, all_depth_conf, all_pose = [], [], [], [], []
        for res in output.ress:
            all_pts3d.append(res["pts3d_in_other_view"].squeeze(0))
            all_conf.append(res["conf"].squeeze(0))
            all_depth.append(res["depth"].squeeze(0))
            all_depth_conf.append(res["depth_conf"].squeeze(0))
            all_pose.append(res["camera_pose"].squeeze(0))

        predictions = {
            "images": images,
            "world_points": torch.stack(all_pts3d, dim=0),
            "world_points_conf": torch.stack(all_conf, dim=0),
            "depth": torch.stack(all_depth, dim=0),
            "depth_conf": torch.stack(all_depth_conf, dim=0),
            "pose_enc": torch.stack(all_pose, dim=0),
        }

        pose_enc = predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        predictions["extrinsic"] = extrinsic.squeeze(0)
        predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None

        print("Output shapes:")
        print(f"  world_points: {tuple(predictions['world_points'].shape)}")
        print(f"  depth: {tuple(predictions['depth'].shape)}")
        print(f"  extrinsic: {tuple(predictions['extrinsic'].shape)}")
        if predictions["intrinsic"] is not None:
            print(f"  intrinsic: {tuple(predictions['intrinsic'].shape)}")
        else:
            print("  intrinsic: None")

        return predictions, image_paths

    def inference_from_folder(self, image_folder: str) -> Tuple[dict, List[str]]:
        image_paths = self._get_image_paths(image_folder)
        return self.inference(image_paths)

    def inference_from_video(self, video_path: str, temp_dir: str = "temp_frames", fps_interval: float = 1.0):
        image_paths = self._extract_frames_from_video(video_path, temp_dir, fps_interval)
        return self.inference(image_paths)


def save_point_cloud_ply(points, colors, confidences, save_path, conf_threshold=0.5):
    mask = confidences > conf_threshold
    points = points[mask]
    colors = colors[mask]

    num_points = points.shape[0]
    print(f"Saving {num_points} points to {save_path}")

    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(save_path, "w") as f:
        f.write(header)
        for i in range(num_points):
            x, y, z = points[i]
            r, g, b = colors[i].astype(np.uint8)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def save_depth_map(depth, save_path, colormap=True):
    if cv2 is None:
        raise ImportError("OpenCV is required for saving depth visualizations. Please install opencv-python.")
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min > 0:
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth)

    depth_16bit = (depth_normalized * 65535).astype(np.uint16)
    cv2.imwrite(f"{save_path}.png", depth_16bit)
    np.save(f"{save_path}.npy", depth)

    if colormap:
        depth_vis = (depth_normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        cv2.imwrite(f"{save_path}_vis.png", depth_colored)

    meta = {"min": float(depth_min), "max": float(depth_max)}
    with open(f"{save_path}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def save_camera_params(extrinsic, intrinsic, save_dir, frame_names):
    os.makedirs(save_dir, exist_ok=True)
    num_frames = extrinsic.shape[0]

    for i in range(num_frames):
        frame_name = os.path.splitext(os.path.basename(frame_names[i]))[0]
        ext_3x4 = extrinsic[i]
        ext_4x4 = np.eye(4)
        ext_4x4[:3, :] = ext_3x4
        np.savetxt(os.path.join(save_dir, f"{frame_name}_extrinsic.txt"), ext_4x4, fmt="%.8f")
        np.savetxt(os.path.join(save_dir, f"{frame_name}_intrinsic.txt"), intrinsic[i], fmt="%.8f")

    save_colmap_format(extrinsic, intrinsic, save_dir, frame_names)
    save_nerf_format(extrinsic, intrinsic, save_dir, frame_names)
    np.savez(os.path.join(save_dir, "cameras.npz"), extrinsic=extrinsic, intrinsic=intrinsic, frame_names=frame_names)


def save_colmap_format(extrinsic, intrinsic, save_dir, frame_names):
    colmap_dir = os.path.join(save_dir, "colmap")
    os.makedirs(colmap_dir, exist_ok=True)

    num_frames = extrinsic.shape[0]

    with open(os.path.join(colmap_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i in range(num_frames):
            K = intrinsic[i]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            width = int(round(cx * 2.0))
            height = int(round(cy * 2.0))
            f.write(f"{i + 1} PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    with open(os.path.join(colmap_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(num_frames):
            ext = extrinsic[i]
            R = ext[:3, :3]
            t = ext[:3, 3]
            quat = rotation_matrix_to_quaternion(R)
            qw, qx, qy, qz = quat
            tx, ty, tz = t
            frame_name = os.path.basename(frame_names[i])
            f.write(f"{i + 1} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {tx:.8f} {ty:.8f} {tz:.8f} {i + 1} {frame_name}\n")
            f.write("\n")

    with open(os.path.join(colmap_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list (empty)\n")


def save_nerf_format(extrinsic, intrinsic, save_dir, frame_names):
    num_frames = extrinsic.shape[0]
    K = intrinsic[0]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    width = int(round(cx * 2.0))
    height = int(round(cy * 2.0))
    fov_x = 2 * np.arctan(width / (2 * fx))
    fov_y = 2 * np.arctan(height / (2 * fy))

    transforms = {
        "camera_angle_x": float(fov_x),
        "camera_angle_y": float(fov_y),
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "frames": [],
    }

    for i in range(num_frames):
        ext_3x4 = extrinsic[i]
        c2w = np.eye(4)
        c2w[:3, :] = ext_3x4
        c2w = np.linalg.inv(c2w)
        c2w[:, 1:3] *= -1
        frame = {"file_path": os.path.basename(frame_names[i]), "transform_matrix": c2w.tolist()}
        transforms["frames"].append(frame)

    with open(os.path.join(save_dir, "transforms.json"), "w") as f:
        json.dump(transforms, f, indent=2)


def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def save_results(predictions, image_paths, output_dir, conf_threshold=0.5):
    if cv2 is None:
        raise ImportError("OpenCV is required for saving RGB/depth outputs. Please install opencv-python.")
    os.makedirs(output_dir, exist_ok=True)

    world_points = predictions["world_points"].cpu().numpy()
    world_points_conf = predictions["world_points_conf"].cpu().numpy()
    depth = predictions["depth"].cpu().numpy()
    images = predictions["images"].cpu().numpy()
    extrinsic = predictions["extrinsic"].cpu().numpy()
    intrinsic = predictions["intrinsic"].cpu().numpy()

    num_frames = world_points.shape[0]

    pc_dir = os.path.join(output_dir, "point_cloud")
    depth_dir = os.path.join(output_dir, "depth")
    cam_dir = os.path.join(output_dir, "cameras")
    img_dir = os.path.join(output_dir, "images")

    os.makedirs(pc_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    all_points, all_colors, all_confs = [], [], []

    print(f"Saving results for {num_frames} frames...")

    for i in range(num_frames):
        frame_name = f"frame_{i:06d}"
        pts = world_points[i].reshape(-1, 3)
        conf = world_points_conf[i].reshape(-1)
        img = images[i].transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        colors = img.reshape(-1, 3)

        save_point_cloud_ply(pts, colors, conf, os.path.join(pc_dir, f"{frame_name}.ply"), conf_threshold)
        all_points.append(pts)
        all_colors.append(colors)
        all_confs.append(conf)

        depth_frame = depth[i, :, :, 0]
        save_depth_map(depth_frame, os.path.join(depth_dir, frame_name))
        cv2.imwrite(os.path.join(img_dir, f"{frame_name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)
    merged_confs = np.concatenate(all_confs, axis=0)
    save_point_cloud_ply(merged_points, merged_colors, merged_confs, os.path.join(pc_dir, "merged.ply"), conf_threshold)

    frame_names = [os.path.basename(p) for p in image_paths]
    save_camera_params(extrinsic, intrinsic, cam_dir, frame_names)

    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="StreamVGGT Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to image folder or video file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="~/.cache/huggingface/hub/models--lch01--StreamVGGT/snapshots/f9ba55b1955bc4f34337c51142158a9aa2862c7f/checkpoints.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--fps_interval", type=float, default=1.0, help="For video: extract one frame every N seconds")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for point cloud filtering")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--fusion", type=str, default="none", choices=["none", "crossattn"], help="Fusion mode")
    parser.add_argument("--event_in_chans", type=int, default=8, help="Event voxel channels")
    parser.add_argument("--fusion_heads", type=int, default=8, help="Cross-attention heads")
    parser.add_argument("--freeze_backbone", action="store_true", help="Whether to freeze backbone at build time")
    args = parser.parse_args()

    inferencer = StreamVGGTInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        fusion=args.fusion,
        event_in_chans=args.event_in_chans,
        fusion_heads=args.fusion_heads,
        freeze_backbone=args.freeze_backbone,
    )

    if os.path.isdir(args.input):
        print(f"Processing image folder: {args.input}")
        predictions, image_paths = inferencer.inference_from_folder(args.input)
    elif os.path.isfile(args.input):
        print(f"Processing video: {args.input}")
        predictions, image_paths = inferencer.inference_from_video(args.input, fps_interval=args.fps_interval)
    else:
        raise ValueError(f"Input path not found: {args.input}")

    save_results(predictions, image_paths, args.output, args.conf_threshold)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
