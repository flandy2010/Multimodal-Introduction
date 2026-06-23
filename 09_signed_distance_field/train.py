import torch
import torch.nn.functional as F
import numpy as np
from model import SDFNetwork, ColorNetwork
from dataloader import TinySDFLoader
from logger import SDFLogger
from tqdm import tqdm


def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    i, j = i.t(), j.t()
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1).to(c2w.device)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def sdf_to_alpha(sdf, s=10.0):
    # 将 SDF 转换为不透明度 (简单版 NeuS 公式)
    # 当 sdf=0 时，alpha 最大；sdf > 0 逐渐透明
    return torch.sigmoid(-sdf * s)


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    loader = TinySDFLoader()
    images, poses, focal = loader.get_all()
    images, poses = images.to(device), poses.to(device)

    sdf_net = SDFNetwork().to(device)
    color_net = ColorNetwork().to(device)
    optimizer = torch.optim.Adam(list(sdf_net.parameters()) + list(color_net.parameters()), lr=1e-3)
    logger = SDFLogger("./sdf_runs")

    for step in tqdm(range(5001)):
        idx = np.random.randint(len(images))
        gt_img = images[idx]
        rays_o, rays_d = get_rays(loader.H, loader.W, focal, poses[idx])

        # 1. 在射线上采样
        n_samples = 64
        z_vals = torch.linspace(2.0, 6.0, n_samples).to(device)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts.requires_grad = True  # 必须开启，为了算 Eikonal Loss

        # 2. 查询 SDF 和 颜色
        flat_pts = pts.reshape(-1, 3)
        sdf, features = sdf_net(flat_pts)
        rgb = color_net(flat_pts, features)

        # 3. 计算 Eikonal Loss (强制梯度模长为 1)
        grad = torch.autograd.grad(outputs=sdf, inputs=flat_pts,
                                   grad_outputs=torch.ones_like(sdf),
                                   create_graph=True)[0]
        loss_eikonal = torch.mean((torch.norm(grad, dim=-1) - 1) ** 2)

        # 4. 体渲染积分
        sdf = sdf.reshape(loader.H, loader.W, n_samples)
        rgb = rgb.reshape(loader.H, loader.W, n_samples, 3)

        alpha = sdf_to_alpha(sdf, s=50.0)  # s越大表面越硬
        weights = alpha * torch.cumprod(1 - alpha + 1e-7, dim=-1)
        rgb_pred = torch.sum(weights[..., None] * rgb, dim=-2)

        # 5. Loss 与 优化
        loss_color = F.mse_loss(rgb_pred, gt_img)
        loss = loss_color + 0.1 * loss_eikonal

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            psnr = -10. * torch.log10(loss_color)
            logger.report(step, loss.item(), psnr.item())
            logger.save_preview(step, rgb_pred, gt_img)
            logger.visualize_sdf_slice(step, sdf_net, device)


if __name__ == "__main__":
    train()