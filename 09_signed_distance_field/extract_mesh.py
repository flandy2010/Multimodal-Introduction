"""
extract_mesh.py — SDF 网格提取与可视化工具

用法:
    python extract_mesh.py --ckpt ./runs/demo01/sdf_final.pth [options]

依赖：skimage（marching cubes）、trimesh（保存 .obj/.ply）、matplotlib（可视化）
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # 无头渲染，在服务器上也能跑
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

sys.path.insert(0, os.path.dirname(__file__))
from model import SDFNetwork


# ---------------------------------------------------------------------------
# 核心：查询 SDF 场并运行 Marching Cubes
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_mesh_from_sdf(
    sdf_net,
    device,
    resolution: int = 128,
    bound: float = 1.0,
    level: float = 0.0,
    chunk: int = 65536,
):
    """
    在 [-bound, bound]^3 空间上建立 resolution^3 网格，查询 SDF 值，
    然后用 Marching Cubes 提取零等值面。

    返回:
        verts: np.ndarray [V, 3]
        faces: np.ndarray [F, 3]
        sdf_grid: np.ndarray [R, R, R]（可用于 SDF slice 可视化）
    """
    # 建立均匀网格点
    t = np.linspace(-bound, bound, resolution)
    xs, ys, zs = np.meshgrid(t, t, t, indexing="ij")          # [R, R, R]
    pts_np = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)    # [R^3, 3]
    pts = torch.from_numpy(pts_np).float().to(device)

    # 分批查询 SDF
    sdf_vals = []
    sdf_net.eval()
    for i in range(0, pts.shape[0], chunk):
        s = sdf_net.sdf(pts[i:i+chunk])
        sdf_vals.append(s.cpu())
    sdf_vals = torch.cat(sdf_vals, 0).squeeze(-1).numpy()      # [R^3]
    sdf_grid = sdf_vals.reshape(resolution, resolution, resolution)

    # Marching Cubes
    verts, faces, normals, values = measure.marching_cubes(sdf_grid, level=level)

    # 将体素坐标 → 世界坐标
    voxel_size = 2.0 * bound / (resolution - 1)
    verts = verts * voxel_size - bound

    return verts, faces, sdf_grid


# ---------------------------------------------------------------------------
# 保存 mesh
# ---------------------------------------------------------------------------

def save_mesh(verts, faces, out_path: str):
    ext = os.path.splitext(out_path)[-1].lower()
    if ext == ".obj":
        _save_obj(verts, faces, out_path)
    elif ext == ".ply":
        _save_ply(verts, faces, out_path)
    else:
        # 尝试 trimesh
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh.export(out_path)
        except ImportError:
            # fallback: 强制 .obj
            out_path = out_path.rsplit(".", 1)[0] + ".obj"
            _save_obj(verts, faces, out_path)
    print(f"  Mesh 已保存: {out_path}")
    return out_path


def _save_obj(verts, faces, path):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fc in faces + 1:                    # OBJ faces 从 1 开始
            f.write(f"f {fc[0]} {fc[1]} {fc[2]}\n")


def _save_ply(verts, faces, path):
    """最简 PLY ASCII 写法"""
    n_v, n_f = len(verts), len(faces)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n_v}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {n_f}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fc in faces:
            f.write(f"3 {fc[0]} {fc[1]} {fc[2]}\n")


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def visualize(verts, faces, sdf_grid, bound, out_dir):
    """
    生成三幅图并保存到 out_dir：
      1. mesh_view.png  — 三视角 3D 线框/面片渲染（azimuth 0/90/135）
      2. sdf_slice.png  — XY / XZ / YZ 三个中间切面的 SDF 值热力图
      3. hist.png       — SDF 值直方图（诊断 SDF 场是否 well-formed）
    """
    os.makedirs(out_dir, exist_ok=True)

    # ------- 1. Mesh 三视角 -------
    fig = plt.figure(figsize=(15, 5))
    azimuths = [30, 120, 210]
    for idx, az in enumerate(azimuths):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        poly = Poly3DCollection(verts[faces], alpha=0.4, linewidths=0.1,
                                edgecolors="steelblue", facecolors="lightsteelblue")
        ax.add_collection3d(poly)
        lim = bound * 1.1
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=az)
        ax.set_title(f"azim={az}°")
    plt.suptitle(f"Extracted Mesh  ({len(verts)} verts, {len(faces)} faces)", y=1.01)
    plt.tight_layout()
    mesh_path = os.path.join(out_dir, "mesh_view.png")
    plt.savefig(mesh_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Mesh 视图: {mesh_path}")

    # ------- 2. SDF slice -------
    R = sdf_grid.shape[0]
    mid = R // 2
    slices = [
        (sdf_grid[mid, :, :], "XY (Z=0)"),
        (sdf_grid[:, mid, :], "XZ (Y=0)"),
        (sdf_grid[:, :, mid], "YZ (X=0)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (sl, title) in zip(axes, slices):
        vmax = max(abs(sl.min()), abs(sl.max()))
        im = ax.imshow(sl.T, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       extent=[-bound, bound, -bound, bound])
        ax.contour(sl.T, levels=[0.0], colors="black",
                   linewidths=1.5, extent=[-bound, bound, -bound, bound])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("")
    plt.suptitle("SDF Slice（黑线=零等值面）")
    plt.tight_layout()
    slice_path = os.path.join(out_dir, "sdf_slice.png")
    plt.savefig(slice_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  SDF slice: {slice_path}")

    # ------- 3. SDF 直方图 -------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sdf_grid.ravel(), bins=200, color="steelblue", alpha=0.8)
    ax.axvline(0, color="red", linewidth=1.5, label="SDF=0（表面）")
    ax.set_xlabel("SDF value"); ax.set_ylabel("count (log)")
    ax.set_yscale("log"); ax.legend()
    ax.set_title("SDF Field Distribution")
    hist_path = os.path.join(out_dir, "sdf_hist.png")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  SDF 直方图: {hist_path}")

    return mesh_path, slice_path, hist_path


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SDF Mesh Extractor")
    parser.add_argument("--ckpt",        type=str, required=True,
                        help="训练权重路径，如 ./runs/demo01/sdf_final.pth")
    parser.add_argument("--out_dir",     type=str, default=None,
                        help="输出目录，默认与 ckpt 同目录")
    parser.add_argument("--resolution",  type=int, default=256,
                        help="体素分辨率（越大越精细，内存消耗 O(R^3)，默认 256）")
    parser.add_argument("--bound",       type=float, default=1.0,
                        help="提取空间范围 [-bound, bound]^3，默认 1.0")
    parser.add_argument("--level",       type=float, default=0.0,
                        help="SDF 零等值面阈值，默认 0.0")
    parser.add_argument("--fmt",         type=str, default="obj",
                        choices=["obj", "ply"],
                        help="保存格式，默认 obj")
    parser.add_argument("--init_radius", type=float, default=0.5,
                        help="与训练时保持一致，默认 0.5")
    parser.add_argument("--device",      type=str, default="auto")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                               "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # 输出目录
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.ckpt))

    # 加载网络
    sdf_net = SDFNetwork(
        d_in=3, d_out=257, d_hidden=256, n_layers=8,
        skip_in=(4,), multires=6, bias=args.init_radius, scale=1.0,
        geometric_init=True, weight_norm=True
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    sdf_net.load_state_dict(ckpt["sdf_net"])
    sdf_net.eval()
    print(f"Checkpoint loaded: {args.ckpt}")

    # 提取
    print(f"\n[1/3] Marching Cubes (resolution={args.resolution}, bound={args.bound}) ...")
    verts, faces, sdf_grid = extract_mesh_from_sdf(
        sdf_net, device,
        resolution=args.resolution,
        bound=args.bound,
        level=args.level,
    )
    print(f"  → {len(verts)} vertices, {len(faces)} faces")

    # 保存 mesh
    print(f"\n[2/3] 保存 Mesh ...")
    mesh_file = os.path.join(out_dir, f"mesh.{args.fmt}")
    save_mesh(verts, faces, mesh_file)

    # 可视化
    print(f"\n[3/3] 生成可视化图像 ...")
    visualize(verts, faces, sdf_grid, args.bound, out_dir)

    print(f"\n✅ 完成！输出目录: {out_dir}")


if __name__ == "__main__":
    main()
