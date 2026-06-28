import os
import json
import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from model import GaussianModel
from renderer import auto_rasterizer
from dataloader import GSDataLoader


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


# ============================================================================
# 轨道几何：完全基于训练相机分布拟合
# ============================================================================

def analyze_training_cameras(poses_np: np.ndarray) -> dict:
    """
    分析训练相机分布，拟合一个最贴合训练视角的 360° 轨道。

    poses_np: [N, 3, 4] COLMAP/OpenCV 约定的 c2w 矩阵
              列依次是 [right, down, forward, t]

    返回:
        center     : 轨道圆心（训练相机的均值位置）
        normal     : 轨道平面法线（≈ 训练相机平均"上"方向，世界真上）
        radius     : 平面内圆半径（训练相机到圆心在平面内的中位距离）
        height     : 圆心沿 normal 方向相对训练相机均值的偏移（拍摄高度修正）
        target     : look-at 目标点（所有相机视线的最小二乘交点 = 被拍主体）
        u_axis,
        v_axis     : 轨道平面内两条正交基向量，便于参数化圆
    """
    cam_centers = poses_np[:, :3, 3].astype(np.float64)   # 相机位置
    # COLMAP c2w 第二列是 cam +Y = down，"相机上方向" 就是 -第二列
    ups = -poses_np[:, :3, 1].astype(np.float64)
    forwards = poses_np[:, :3, 2].astype(np.float64)      # cam +Z = forward

    cam_mean = cam_centers.mean(axis=0)

    # ---- 1) 轨道平面：取相机分布最薄方向做法线，且与"平均相机上向"同侧 ----
    centered = cam_centers - cam_mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    mean_up = ups.mean(axis=0)
    mean_up /= (np.linalg.norm(mean_up) + 1e-12)
    if np.dot(normal, mean_up) < 0:
        normal = -normal
    normal /= (np.linalg.norm(normal) + 1e-12)

    # ---- 2) 平面内半径：训练相机到 cam_mean 在平面内投影距离的中位数 ----
    heights = centered @ normal                            # 沿法线的偏移
    in_plane = centered - heights[:, None] * normal
    radii = np.linalg.norm(in_plane, axis=1)
    radius = float(np.median(radii))

    # ---- 3) look-at 目标：所有相机视线的最小二乘交点 ----
    # 每条线 p_i(t) = c_i + t * f_i，求 P 最小化 ∑ ||(I - f_i f_i^T)(P - c_i)||^2
    I3 = np.eye(3)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, f in zip(cam_centers, forwards):
        f = f / (np.linalg.norm(f) + 1e-12)
        M = I3 - np.outer(f, f)
        A += M
        b += M @ c
    target = np.linalg.solve(A, b)

    # ---- 4) 在轨道平面内挑选两条正交参数化基向量 ----
    # 用全局最常见的世界轴作为种子，避免每次轨道方向随机翻转
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, normal)) > 0.9:
        seed = np.array([0.0, 0.0, 1.0])
    u_axis = seed - np.dot(seed, normal) * normal
    u_axis /= (np.linalg.norm(u_axis) + 1e-12)
    v_axis = np.cross(normal, u_axis)
    v_axis /= (np.linalg.norm(v_axis) + 1e-12)

    return {
        "center": cam_mean,
        "normal": normal,
        "radius": radius,
        "target": target,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "cam_centers": cam_centers,
    }


def make_intrinsics_from_loader(loader: GSDataLoader, width: int, height: int, device: torch.device) -> torch.Tensor:
    """
    直接复用训练时的相机内参（焦距），等比缩放到输出分辨率。
    这样 FoV / 主点都与训练完全一致，避免投影错位。
    """
    src_w, src_h = loader.W, loader.H
    src_focal = float(loader.K[0, 0].item())
    sx = width / src_w
    sy = height / src_h
    fx = src_focal * sx
    fy = src_focal * sy
    cx = width * 0.5
    cy = height * 0.5
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    return K


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.norm(v) + eps)


def look_at_c2w(eye: torch.Tensor, target: torch.Tensor, world_up: torch.Tensor) -> torch.Tensor:
    """
    构造 COLMAP/OpenCV 约定的 c2w：相机 +X 右、+Y 下、+Z 前。
    满足 right × down = forward（右手系）。
    """
    forward = normalize(target - eye)
    world_down = -world_up
    right = torch.cross(world_down, forward, dim=0)
    if torch.norm(right) < 1e-6:
        alt = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=eye.device)
        right = torch.cross(alt, forward, dim=0)
    right = normalize(right)
    down = normalize(torch.cross(forward, right, dim=0))

    c2w = torch.eye(4, dtype=torch.float32, device=eye.device)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def build_orbit_from_training(
    orbit_info: dict,
    n_frames: int,
    orbit_scale: float,
    height_offset_ratio: float,
    device: torch.device,
) -> list[torch.Tensor]:
    """
    在训练相机拟合的平面上构造均匀 360° 圆形轨道。
    所有视角都看向真实的 look-at target（训练视线的交点）。

    Args:
        orbit_info         : analyze_training_cameras 的返回
        n_frames           : 总帧数
        orbit_scale        : 在拟合半径上的放缩（1.0 = 与训练相机同距）
        height_offset_ratio: 沿轨道法线方向额外抬高的比例，相对拟合半径
                             正值=更俯视，负值=更平视/仰视，0=与训练分布同高
    """
    center = torch.tensor(orbit_info["center"], dtype=torch.float32, device=device)
    normal = torch.tensor(orbit_info["normal"], dtype=torch.float32, device=device)
    u_axis = torch.tensor(orbit_info["u_axis"], dtype=torch.float32, device=device)
    v_axis = torch.tensor(orbit_info["v_axis"], dtype=torch.float32, device=device)
    target = torch.tensor(orbit_info["target"], dtype=torch.float32, device=device)

    radius = float(orbit_info["radius"]) * orbit_scale
    height_offset = float(orbit_info["radius"]) * height_offset_ratio

    # 轨道圆心：cam_mean 沿 normal 方向上移 height_offset
    orbit_center = center + normal * height_offset

    poses: list[torch.Tensor] = []
    angles = torch.linspace(0.0, 2.0 * np.pi, steps=n_frames + 1, device=device)[:-1]
    for theta in angles:
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        eye = orbit_center + radius * (cos_t * u_axis + sin_t * v_axis)
        # 世界"上"用轨道平面法线（这就是训练相机平均上方向，最自然）
        poses.append(look_at_c2w(eye, target, normal))
    return poses


# ============================================================================
# 主流程
# ============================================================================

def render_walkthrough(args):
    device = resolve_device(args.device)
    print(f"[Info] device: {device}")

    # ---- 1) 通过 args.json 自动定位训练数据，加载真实训练相机 ----
    run_dir = os.path.dirname(os.path.abspath(args.model_path))
    train_args_path = os.path.join(run_dir, "args.json")
    if args.data_path:
        data_path = args.data_path
        factor = args.factor
    else:
        if not os.path.exists(train_args_path):
            raise FileNotFoundError(
                f"未找到 {train_args_path}，请显式传 --data_path/--factor 指定训练数据"
            )
        with open(train_args_path, "r") as f:
            ta = json.load(f)
        data_path = ta["data_path"]
        factor = int(ta.get("factor", 2))
        # args.json 里的相对路径是相对训练时 cwd（08_3d_gaussian_splatting/）
        if not os.path.isabs(data_path):
            data_path = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)
            )

    print(f"[Info] loading training cameras from: {data_path} (factor={factor})")
    loader = GSDataLoader(data_path=data_path, factor=factor, max_init_points=1000)

    orbit_info = analyze_training_cameras(loader.poses.numpy())
    print(f"[Info] orbit center       = {orbit_info['center'].tolist()}")
    print(f"[Info] orbit normal       = {orbit_info['normal'].tolist()}")
    print(f"[Info] orbit radius       = {orbit_info['radius']:.4f}")
    print(f"[Info] look-at target     = {orbit_info['target'].tolist()}")

    # ---- 2) 加载模型 ----
    ckpt_state = load_checkpoint(args.model_path)
    num_points, sh_degree = infer_model_shape(ckpt_state)
    print(f"[Info] checkpoint points  = {num_points}, sh_degree={sh_degree}")

    K = make_intrinsics_from_loader(loader, args.width, args.height, device)
    focal_x = float(K[0, 0].item())
    focal_y = float(K[1, 1].item())

    model = GaussianModel(
        fx=focal_x,
        fy=focal_y,
        num_points=num_points,
        radius=loader.scene_radius,
        sh_degree=sh_degree,
        pcd=None,
    ).to(device)

    model.load_state_dict(ckpt_state, strict=True)
    model.eval()
    model.active_sh_degree = model.sh_degree

    # ---- 3) 构造 360° 轨道 ----
    c2w_list = build_orbit_from_training(
        orbit_info=orbit_info,
        n_frames=args.n_frames,
        orbit_scale=args.orbit_scale,
        height_offset_ratio=args.height_offset_ratio,
        device=device,
    )

    # ---- 4) 逐帧渲染 ----
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
    parser = argparse.ArgumentParser(description="Render 3DGS 360-degree orbit GIF aligned with training cameras")
    parser.add_argument("--model_path", type=str, required=True, help="Trained checkpoint path (e.g. runs/.../gs_final.pth)")
    parser.add_argument("--output_gif", type=str, default="./runs/walkthrough_orbit.gif", help="Output GIF path")

    parser.add_argument("--data_path", type=str, default="", help="可选：手动指定训练数据目录；缺省自动从 args.json 读取")
    parser.add_argument("--factor", type=int, default=2, help="数据下采样因子（仅在显式传 --data_path 时生效）")

    parser.add_argument("--width", type=int, default=800, help="Render width")
    parser.add_argument("--height", type=int, default=800, help="Render height")

    parser.add_argument("--n_frames", type=int, default=120, help="Total frames in 360-degree orbit")
    parser.add_argument("--fps", type=int, default=20, help="GIF frame rate")
    parser.add_argument("--orbit_scale", type=float, default=1.0,
                        help="相对训练相机半径的放缩，1.0 = 与训练同距，<1.0 更靠近，>1.0 更远")
    parser.add_argument("--height_offset_ratio", type=float, default=0.0,
                        help="沿轨道法线额外抬高的比例（相对训练半径），0=与训练分布同高，>0=更俯视")

    parser.add_argument("--tile_size", type=int, default=32, help="Render tile size")
    parser.add_argument("--radius_clip", type=float, default=0.0, help="Passed to gsplat radius_clip")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/mps/cpu")
    return parser


def main():
    args = build_parser().parse_args()
    render_walkthrough(args)


if __name__ == "__main__":
    main()
