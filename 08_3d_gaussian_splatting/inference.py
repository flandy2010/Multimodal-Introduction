import os
import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

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


def estimate_scene_center_and_radius(ckpt_state: dict) -> tuple[torch.Tensor, float]:
    means = ckpt_state["gauss_params.means"].float()  # [N, 3]

    raw_op = ckpt_state.get("gauss_params.opacities", None)
    if raw_op is not None:
        weights = torch.sigmoid(raw_op.float().squeeze()).clamp_min(1e-6)
    else:
        weights = torch.ones(means.shape[0], dtype=torch.float32)

    center = (means * weights[:, None]).sum(dim=0) / weights.sum()

    dist = torch.norm(means - center[None, :], dim=-1)
    radius_p90 = torch.quantile(dist, 0.90).item()
    radius = max(radius_p90, 0.3)
    return center, radius


def make_intrinsics(width: int, height: int, fov_deg: float, device: torch.device) -> torch.Tensor:
    fov_rad = np.deg2rad(fov_deg)
    fx = 0.5 * width / np.tan(0.5 * fov_rad)
    fy = fx
    cx, cy = width * 0.5, height * 0.5

    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    return K


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.norm(v) + eps)


def look_at_c2w(eye: torch.Tensor, target: torch.Tensor, up_hint: torch.Tensor) -> torch.Tensor:
    forward = normalize(target - eye)
    right = torch.cross(forward, up_hint, dim=0)
    if torch.norm(right) < 1e-6:
        alt_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=eye.device)
        right = torch.cross(forward, alt_up, dim=0)
    right = normalize(right)
    up = normalize(torch.cross(right, forward, dim=0))

    c2w = torch.eye(4, dtype=torch.float32, device=eye.device)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    # 与本项目 renderer 约定对齐：相机前方在 camera +Z（depth > 0 才可见）
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def build_orbit_poses(
    center: torch.Tensor,
    scene_radius: float,
    n_frames: int,
    orbit_scale: float,
    height_ratio: float,
) -> list[torch.Tensor]:
    orbit_r = scene_radius * orbit_scale
    height = scene_radius * height_ratio
    up_hint = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=center.device)

    poses = []
    angles = torch.linspace(0.0, 2.0 * np.pi, steps=n_frames + 1, device=center.device)[:-1]
    for theta in angles:
        eye = center + torch.tensor(
            [orbit_r * torch.cos(theta), height, orbit_r * torch.sin(theta)],
            dtype=torch.float32,
            device=center.device,
        )
        poses.append(look_at_c2w(eye, center, up_hint))
    return poses


def render_walkthrough(args):
    device = resolve_device(args.device)
    print(f"[Info] device: {device}")

    ckpt_state = load_checkpoint(args.model_path)
    num_points, sh_degree = infer_model_shape(ckpt_state)
    center, scene_radius = estimate_scene_center_and_radius(ckpt_state)
    center = center.to(device)

    print(f"[Info] checkpoint points={num_points}, sh_degree={sh_degree}")
    print(f"[Info] estimated center={center.tolist()}, scene_radius≈{scene_radius:.4f}")

    K = make_intrinsics(args.width, args.height, args.fov_deg, device)
    focal = float(K[0, 0].item())

    model = GaussianModel(
        fx=focal,
        fy=focal,
        num_points=num_points,
        radius=scene_radius,
        sh_degree=sh_degree,
        pcd=None,
    ).to(device)

    model.load_state_dict(ckpt_state, strict=True)
    model.eval()
    model.active_sh_degree = model.sh_degree

    c2w_list = build_orbit_poses(
        center=center,
        scene_radius=scene_radius,
        n_frames=args.n_frames,
        orbit_scale=args.orbit_scale,
        height_ratio=args.height_ratio,
    )

    frames = []
    with torch.no_grad():
        progress = tqdm(c2w_list, desc="正在渲染360°环绕视角帧", unit="frame")
        for c2w in progress:
            w2c = torch.inverse(c2w)
            camera_pos = c2w[:3, 3]

            gaussians = model(camera_pos=camera_pos)
            image = auto_rasterizer(
                gaussians,
                w2c,
                K,
                args.height,
                args.width,
                tile_size=args.tile_size,
                radius_clip=args.radius_clip,
            )

            image_np = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
            frames.append(Image.fromarray(image_np))

    out_dir = os.path.dirname(args.output_gif)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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
    parser = argparse.ArgumentParser(description="Render 3DGS 360-degree orbit GIF from trained checkpoint")
    parser.add_argument("--model_path", type=str, required=True, help="Trained checkpoint path (e.g. runs/.../gs_final.pth)")
    parser.add_argument("--output_gif", type=str, default="./runs/walkthrough_orbit.gif", help="Output GIF path")

    parser.add_argument("--width", type=int, default=800, help="Render width")
    parser.add_argument("--height", type=int, default=800, help="Render height")
    parser.add_argument("--fov_deg", type=float, default=60.0, help="Horizontal FOV in degrees")

    parser.add_argument("--n_frames", type=int, default=120, help="Total frames in 360-degree orbit")
    parser.add_argument("--fps", type=int, default=20, help="GIF frame rate")
    parser.add_argument("--orbit_scale", type=float, default=2.2, help="Orbit radius = scene_radius * orbit_scale")
    parser.add_argument("--height_ratio", type=float, default=0.15, help="Camera height offset = scene_radius * height_ratio")

    parser.add_argument("--tile_size", type=int, default=32, help="Render tile size")
    parser.add_argument("--radius_clip", type=float, default=0.0, help="Passed to gsplat radius_clip")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/mps/cpu")
    return parser


def main():
    args = build_parser().parse_args()
    render_walkthrough(args)


if __name__ == "__main__":
    main()
