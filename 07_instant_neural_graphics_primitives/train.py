import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from model import InstantNGP
from dataloader import TinyNeRFDataset
from logger import NeRFLogger # 导入新日志系统


# [get_rays 保持不变]
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    i, j = i.t(), j.t()
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    dirs = dirs.to(c2w)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


# [render_rays 保持不变，包含 Jittering]
def render_rays(model, rays_o, rays_d, near, far, n_samples):
    t_vals = torch.linspace(0., 1., steps=n_samples).to(rays_o.device)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand(rays_o.shape[:-1] + (n_samples,))

    if model.training:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape).to(rays_o.device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts_flat = pts.reshape(-1, 3)
    d_flat = rays_d[..., None, :].expand(pts.shape).reshape(-1, 3)

    chunk = 1024 * 64
    raw = []
    for i in range(0, pts_flat.shape[0], chunk):
        raw.append(model(pts_flat[i:i + chunk], d_flat[i:i + chunk]))
    raw = torch.cat(raw, 0)
    raw = raw.reshape(pts.shape[0], pts.shape[1], n_samples, 4)

    sigma = raw[..., 3]
    if model.training:
        # 加入强度为 1.0 的随机噪声，强迫模型放弃那些“不稳固”的孤立噪点
        sigma = sigma + torch.randn_like(sigma) * 1.0
    sigma = F.relu(sigma)

    rgb = raw[..., :3]

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e10]).to(z_vals.device).expand(z_vals[..., :1].shape)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # 计算透射率和实际权重
    alpha = 1. - torch.exp(-sigma * dists)
    transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1)[..., :-1]
    weights = alpha * transmittance
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    return rgb_map, raw, sigma

@torch.no_grad()
def evaluate(args, model, test_dataset, device, i, logger, current_lr):
    model.eval()
    target_img, target_pose = test_dataset[0]
    target_img, target_pose = target_img.to(device), target_pose.to(device)
    H, W, focal = test_dataset.H, test_dataset.W, test_dataset.focal

    rays_o, rays_d = get_rays(H, W, focal, target_pose)
    rgb_pred, _, _ = render_rays(model, rays_o, rays_d, near=2.5, far=5.5, n_samples=args.n_samples)

    # 利用 Logger 计算详细指标
    metrics = logger.calculate_image_metrics(rgb_pred, target_img)

    # 记录到 Markdown
    logger.log_evaluation(i, metrics, current_lr)
    # 打印分析报告
    logger.print_analysis(i, metrics, current_lr)

    # 可视化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(target_img.cpu().numpy())
    ax1.set_title("Ground Truth")
    ax1.axis('off')

    pred_np = rgb_pred.detach().cpu().numpy()
    ax2.imshow(pred_np)
    ax2.set_title(f"Iter {i} (PSNR: {metrics['psnr']:.2f})")
    ax2.axis('off')

    # 新增：误差热力图（直观看到哪里没练好）
    err_map = np.abs(pred_np - target_img.cpu().numpy()).mean(axis=-1)
    im = ax3.imshow(err_map, cmap='jet')
    ax3.set_title("Error Heatmap")
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    save_path = os.path.join(args.exp_dir, f"iter{i}_metrics.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    model.train()
    return metrics['psnr']

def train(args, model, train_dataset, test_dataset, device, logger):

    H, W, focal = train_dataset.H, train_dataset.W, train_dataset.focal
    optimizer = torch.optim.Adam([
        {'params': model.hash_encoder.parameters(), 'weight_decay': 1e-6},  # 给哈希表加约束
        {'params': model.sigma_net.parameters(), 'lr': args.lr},
        {'params': model.color_net.parameters(), 'lr': args.lr},
    ], lr=args.lr, eps=1e-15)
    pbar = tqdm(range(args.n_iters), desc="Training")

    moving_avg_loss = None
    ema_alpha = 0.9

    def update_lr(step):
        decay_rate = 0.01
        new_lrate = args.lr * (decay_rate ** (step / args.n_iters))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        return new_lrate

    for i in pbar:
        current_lr = update_lr(i)

        model.train()
        idx = np.random.randint(len(train_dataset))
        target_img, target_pose = train_dataset[idx]
        target_img, target_pose = target_img.to(device), target_pose.to(device)

        rays_o, rays_d = get_rays(H, W, focal, target_pose)
        rgb_pred, raw, sigma = render_rays(model, rays_o, rays_d, near=2.5, far=5.5, n_samples=args.n_samples)

        optimizer.zero_grad()

        loss_mse = F.mse_loss(rgb_pred, target_img)

        # 核心：加入稀疏性正则化 (Sparsity Loss)
        # 它的含义是：除非万不得已（为了凑颜色），否则空间里的密度越少越好，从而强行杀掉背景里的浮片
        loss_sparse = torch.mean(torch.abs(raw[..., 3])) * 1e-4

        # 核心：加入 TV Loss (全变分) —— 抹平局部噪点
        # 惩罚相邻采样点之间的密度突变
        sigmas = raw[..., 3]  # [num_rays, n_samples]
        loss_tv = torch.mean(torch.abs(sigmas[:, 1:] - sigmas[:, :-1])) * 1e-5

        loss = loss_mse + loss_sparse + loss_tv
        loss.backward()

        optimizer.step()

        current_loss_val = loss.item()
        if moving_avg_loss is None:
            moving_avg_loss = current_loss_val
        else:
            moving_avg_loss = ema_alpha * moving_avg_loss + (1 - ema_alpha) * current_loss_val

        pbar.set_postfix({"LR": f"{current_lr:.1e}", "AvgL": f"{moving_avg_loss:.4f}"})

        if i % args.display_int == 0:
            evaluate(args, model, test_dataset, device, i, logger, current_lr)

    torch.save(model.state_dict(), os.path.join(args.exp_dir, "nerf_final.pth"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/tiny_nerf_data.npz")
    parser.add_argument("--exp_dir", type=str, default="./runs/exp1")
    parser.add_argument("--n_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n_samples", type=int, default=192)
    parser.add_argument("--display_int", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if args.device != "auto": device = torch.device(args.device)

    # 初始化日志系统
    logger = NeRFLogger(args.exp_dir, args)

    train_dataset = TinyNeRFDataset(args.data_path, mode='train')
    test_dataset = TinyNeRFDataset(args.data_path, mode='test')

    model = InstantNGP().to(device)
    train(args, model, train_dataset, test_dataset, device, logger)


if __name__ == "__main__":
    main()