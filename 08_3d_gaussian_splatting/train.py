import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataloader import GSDataLoader
from model import GaussianModel


def project_points(means3D, w2c, K, H, W):
    """将 3D 点投影到 2D 像素坐标"""
    # 转换到相机坐标系: [N, 3]
    points_cam = (w2c[:3, :3] @ means3D.t()).t() + w2c[:3, 3]

    # 深度 z
    depths = points_cam[:, 2:3]

    # 投影到屏幕: [N, 2]
    points_2d = (K[:2, :2] @ (points_cam[:, :2] / depths).t()).t() + K[:2, 2]

    # 简单的半径估算 (根据深度和初始缩放)
    # 在真实 3DGS 中这是由协方差矩阵计算的，这里简化处理
    radii = (100.0 / (depths + 1e-5))  # 随距离衰减的半径

    return points_2d, depths, radii


def simple_rasterizer(gaussians, w2c, K, H, W):
    """
    简易可微栅格化 (Pure PyTorch)
    注意：为了演示，这里使用了简化版的 Point-based 渲染
    """
    means2D, depths, radii = project_points(gaussians["means"], w2c, K, H, W)
    colors = gaussians["colors"]
    opacities = gaussians["opacity"]

    # 1. 排序 (Depth Sorting) - 3DGS 的核心，从远到近
    indices = torch.argsort(depths.squeeze(), descending=True)

    # 2. 初始化画布
    canvas = torch.zeros((H, W, 3), device=means2D.device)

    # 3. 渲染循环 (简化版：将高斯点绘制为圆形)
    # 注意：在 100x100 下，我们直接用向量化方式模拟
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).to(means2D.device).float()  # [H, W, 2]

    # 为了速度，我们只渲染前 500 个有效的、最近的高斯点（简化版演示）
    # 真实的 3DGS 使用 Tile-based 栅格化，非常快
    render_idx = indices[-500:]

    for idx in render_idx:
        mu = means2D[idx]
        color = colors[idx]
        alpha = opacities[idx]
        r = radii[idx]

        # 计算像素到中心的距离
        dist_sq = torch.sum((grid - mu) ** 2, dim=-1)
        # 高斯分布公式
        g = torch.exp(-dist_sq / (2 * (r ** 2) + 1e-5))

        # Alpha Blending (Over operator)
        weighted_alpha = alpha * g
        canvas = canvas * (1 - weighted_alpha.unsqueeze(-1)) + color * weighted_alpha.unsqueeze(-1)

    return canvas


def train():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    loader = GSDataLoader()
    model = GaussianModel(num_points=2000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    pbar = tqdm(range(2000))
    for step in pbar:
        # 随机选一个视角
        idx = np.random.randint(len(loader.images))
        gt_image, w2c, K = loader.get_view_params(idx)
        gt_image, w2c, K = gt_image.to(device), w2c.to(device), K.to(device)

        # 前向传播
        gaussians = model()
        out_image = simple_rasterizer(gaussians, w2c, K, loader.H, loader.W)

        # Loss
        loss = F.mse_loss(out_image, gt_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            psnr = -10. * torch.log10(loss)
            pbar.set_postfix({"PSNR": f"{psnr.item():.2f}"})
            plt.imsave(f"gs_iter_{step}.png", out_image.detach().cpu().numpy().clip(0, 1))


if __name__ == "__main__":

    train()