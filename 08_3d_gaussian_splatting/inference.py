import os
import argparse
import numpy as np
from PIL import Image
import torch

from dataloader import GSDataLoader
from model import GaussianModel
from renderer import auto_rasterizer


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_c2w_4x4(pose_3x4: torch.Tensor) -> torch.Tensor:
    c2w = torch.eye(4, dtype=pose_3x4.dtype)
    c2w[:3, :4] = pose_3x4
    return c2w


def _orthonormalize_rotation(r: torch.Tensor) -> torch.Tensor:
    u, _, v = torch.linalg.svd(r)
    r_ortho = u @ v
    if torch.linalg.det(r_ortho) < 0:
        u[:, -1] *= -1
        r_ortho = u @ v
    return r_ortho


def interpolate_c2w(c2w_a: torch.Tensor, c2w_b: torch.Tensor, alpha: float) -> torch.Tensor:
    r_a = c2w_a[:3, :3]
    r_b = c2w_b[:3, :3]
    t_a = c2w_a[:3, 3]
    t_b = c2w_b[:3, 3]

    r = _orthonormalize_rotation((1.0 - alpha) * r_a + alpha * r_b)
    t = (1.0 - alpha) * t_a + alpha * t_b

    out = torch.eye(4, dtype=c2w_a.dtype)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def build_walk_path_from_dataset_poses(poses_3x4: torch.Tensor, n_frames: int) -> list[torch.Tensor]:
    if n_frames <= 0:
        raise ValueError("n_frames 必须 > 0")

    n_cam = poses_3x4.shape[0]
    if n_cam < 2:
        return [_to_c2w_4x4(poses_3x4[0]) for _ in range(n_frames)]

    c2ws = [_to_c2w_4x4(poses_3x4[i]) for i in range(n_cam)]
    out = []
    for k in range(n_frames):
        x = (k / max(1, n_frames)) * n_cam
        i0 = int(np.floor(x)) % n_cam
        i1 = (i0 + 1) % n_cam
        a = float(x - np.floor(x))
        out.append(interpolate_c2w(c2ws[i0], c2ws[i1], a))
    return out


def infer_model_shape(ckpt: dict) -> tuple[int, int]:
    means_key = "gauss_params.means"
    sh_rest_key = "gauss_params.sh_rest"
    if means_key not in ckpt or sh_rest_key not in ckpt:
        raise KeyError("checkpoint 缺少 gauss_params.means 或 gauss_params.sh_rest，无法推断模型形状")

    num_points = int(ckpt[means_key].shape[0])
    n_rest = int(ckpt[sh_rest_key].shape[1])
    n_coeffs = n_rest + 1
    sh_degree = int(round(np.sqrt(n_coeffs) - 1))
    return num_points, sh_degree


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError("checkpoint 格式不支持，期望是 state_dict 或包含 state_dict 的 dict")


def render_walkthrough(args):
    device = resolve_device(args.device)
    print(f"[Info] device: {device}")

    loader = GSDataLoader(args.data_path, factor=args.factor, max_init_points=-1)
    scene_radius = loader.get_normalization_params()["radius"]

    ckpt_state = load_checkpoint(args.model_path)
    num_points, sh_degree = infer_model_shape(ckpt_state)
    print(f"[Info] checkpoint points={num_points}, sh_degree={sh_degree}")

    model = GaussianModel(
        fx=loader.focal,
        fy=loader.focal,
        num_points=num_points,
        radius=scene_radius,
        sh_degree=sh_degree,
        pcd=None,
    ).to(device)

    model.load_state_dict(ckpt_state, strict=True)
    model.eval()
    model.active_sh_degree = model.sh_degree

    K = loader.K.to(device)
    c2w_list = build_walk_path_from_dataset_poses(loader.poses, args.n_frames)

    frames = []
    with torch.no_grad():
        for i, c2w in enumerate(c2w_list):
            c2w = c2w.to(device)
            w2c = torch.inverse(c2w)
            camera_pos = c2w[:3, 3]

            gaussians = model(camera_pos=camera_pos)
            image = auto_rasterizer(
                gaussians,
                w2c,
                K,
                loader.H,
                loader.W,
                tile_size=args.tile_size,
                radius_clip=args.radius_clip,
            )

            image_np = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
            frames.append(Image.fromarray(image_np))

            if (i + 1) % 20 == 0 or i == len(c2w_list) - 1:
                print(f"[Render] {i + 1}/{len(c2w_list)}")

    os.makedirs(os.path.dirname(args.output_gif), exist_ok=True)
    duration_ms = int(1000 / max(1, args.fps))
    frames[0].save(
        args.output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"[Done] GIF saved to: {args.output_gif}")


def build_parser():
    parser = argparse.ArgumentParser(description="Render 3DGS walkthrough GIF from trained checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="COLMAP scene path used during training")
    parser.add_argument("--model_path", type=str, required=True, help="Trained checkpoint path (e.g. runs/.../gs_final.pth)")
    parser.add_argument("--output_gif", type=str, default="./runs/walkthrough.gif", help="Output GIF path")

    parser.add_argument("--factor", type=int, default=2, help="Image downscale factor, should match training")
    parser.add_argument("--n_frames", type=int, default=120, help="Total frames in walkthrough")
    parser.add_argument("--fps", type=int, default=20, help="GIF frame rate")

    parser.add_argument("--tile_size", type=int, default=32, help="Render tile size")
    parser.add_argument("--radius_clip", type=float, default=0.0, help="Passed to gsplat radius_clip")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/mps/cpu")
    return parser


def main():
    args = build_parser().parse_args()
    render_walkthrough(args)


if __name__ == "__main__":
    main()
