import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import imageio
from tqdm import tqdm
from model import InstantNGP
from dataloader import TinyNeRFDataset


# 复用 train.py 中的核心函数
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    i, j = i.t(), j.t()
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    dirs = dirs.to(c2w)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def render_rays(model, rays_o, rays_d, near, far, n_samples):
    model.eval()  # 确保在 eval 模式
    t_vals = torch.linspace(near, far, n_samples).to(rays_o.device)
    z_vals = t_vals.expand(rays_o.shape[:-1] + (n_samples,))
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts_flat = pts.reshape(-1, 3)
    d_flat = rays_d[..., None, :].expand(pts.shape).reshape(-1, 3)

    chunk = 1024 * 64
    raw = []
    for i in range(0, pts_flat.shape[0], chunk):
        # 注意模型 forward 返回的是 [rgb, sigma]
        raw.append(model(pts_flat[i:i + chunk], d_flat[i:i + chunk]))
    raw = torch.cat(raw, 0)
    raw = raw.reshape(pts.shape[0], pts.shape[1], n_samples, 4)

    rgb = raw[..., :3]
    sigma = F.relu(raw[..., 3])

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e10]).to(z_vals.device).expand(z_vals[..., :1].shape)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    alpha = 1. - torch.exp(-sigma * dists)
    transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1)[..., :-1]
    weights = alpha * transmittance
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    return rgb_map


# 生成旋转相机位姿的辅助函数
def pose_spherical(theta, phi, radius):
    """
    生成绕中心旋转的相机位姿 (c2w 矩阵)
    """

    def trans_t(t): return torch.tensor([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]
    ], dtype=torch.float32)

    def rot_phi(phi): return torch.tensor([
        [1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0], [0, np.sin(phi), np.cos(phi), 0], [0, 0, 0, 1]
    ], dtype=torch.float32)

    def rot_theta(th): return torch.tensor([
        [np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0], [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]
    ], dtype=torch.float32)

    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w

    # 调整坐标系以匹配 tiny_nerf (OpenGL -> World)
    c2w = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32) @ c2w
    return c2w


@torch.no_grad()
def run_inference():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/tiny_nerf_data.npz")
    parser.add_argument("--exp_dir", type=str, default="./runs/exp1", help="包含 nerf_final.pth 的目录")
    parser.add_argument("--n_samples", type=int, default=192)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if args.device != "auto":
        device = torch.device(args.device)

    os.makedirs(os.path.join(args.exp_dir, "result"), exist_ok=True)

    # 1. 加载模型
    model = InstantNGP().to(device)
    ckpt_path = os.path.join(args.exp_dir, "nerf_final.pth")
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到权重文件 {ckpt_path}")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"✅ 成功加载模型: {ckpt_path}")

    # 2. 加载测试数据
    test_dataset = TinyNeRFDataset(args.data_path, mode='test')
    H, W, focal = test_dataset.H, test_dataset.W, test_dataset.focal

    # --- 功能 1: 推理 4 张对比图 ---
    print("正在生成 4 组测试对比图 (第一行 GT，第二行 Pred)...")

    # 增加 figsize 的高度（从 8 增加到 12），为 Title 留出空间
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))

    # 存储预测图，稍后统一画，避免 axes 索引混乱
    preds = []
    psnrs = []
    gts = []

    # 1. 先进行推理计算
    for i in range(4):
        target_img, target_pose = test_dataset[i]
        target_img, target_pose = target_img.to(device), target_pose.to(device)

        rays_o, rays_d = get_rays(H, W, focal, target_pose)
        rgb_pred = render_rays(model, rays_o, rays_d, 2.5, 5.5, args.n_samples)

        mse = F.mse_loss(rgb_pred, target_img)
        psnr = -10. * torch.log10(mse).item()

        gts.append(target_img.cpu().numpy())
        preds.append(rgb_pred.cpu().numpy())
        psnrs.append(psnr)

    # 2. 统一绘制
    for i in range(4):
        # --- 第一行：Ground Truth ---
        ax_gt = axes[0, i]
        ax_gt.imshow(gts[i])
        ax_gt.set_title(f"Sample {i}\n[ Gold Answer ]", fontsize=14, pad=15)
        ax_gt.axis('off')

        # --- 第二行：Prediction ---
        ax_pred = axes[1, i]
        ax_pred.imshow(preds[i])
        # 使用 color 参数突出 PSNR，pad 增加标题与图片的距离
        ax_pred.set_title(f"Sample {i} Predicted\nPSNR: {psnrs[i]:.2f}",
                          fontsize=14, pad=15, color='darkblue', fontweight='bold')
        ax_pred.axis('off')

    # --- 关键：调整布局参数 ---
    # hspace: 控制行与行之间的高度间距 (0.5 已经很大了)
    # wspace: 控制列与列之间的宽度间距
    plt.subplots_adjust(hspace=0.6, wspace=0.2, top=0.9, bottom=0.1)

    # 确保输出目录存在
    save_dir = os.path.join(args.exp_dir, "result")
    os.makedirs(save_dir, exist_ok=True)

    comparison_path = os.path.join(save_dir, "comparison_4.png")
    # 使用 bbox_inches='tight' 自动裁剪多余白边
    plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 对比图（上下排列版）已保存至: {comparison_path}")

    # --- 功能 2: 360 度旋转 GIF ---
    print("正在渲染 360 度旋转动画 (共 40 帧)...")
    frames = []
    # 设定一个固定的俯仰角和半径 (根据数据统计，半径 4.03, 俯仰 ~-30度比较好看)
    radius = 4.03
    phi = -30.0

    for angle in tqdm(np.linspace(0, 360, 40, endpoint=False)):
        c2w = pose_spherical(angle, phi, radius).to(device)
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rgb_pred = render_rays(model, rays_o, rays_d, 2.5, 5.5, args.n_samples)

        # 转换为 uint8 字节流
        frame = (rgb_pred.cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)

    gif_path = os.path.join(args.exp_dir, "result", "rotation.gif")
    imageio.mimsave(gif_path, frames, fps=20, loop=0)
    print(f"✅ 360度旋转 GIF 已保存至: {gif_path}")


if __name__ == "__main__":
    run_inference()