import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from model import InstantNGP
from dataloader import TinyNeRFDataset

# [get_rays 和 render_rays 保持不变]
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    i, j = i.t(), j.t()
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    dirs = dirs.to(c2w)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def render_rays(model, rays_o, rays_d, near, far, n_samples):
    # 1. 生成基础的采样分布 [n_samples]
    t_vals = torch.linspace(0., 1., steps=n_samples).to(rays_o.device)
    # 将 0~1 映射到实标距离 near~far
    z_vals = near * (1. - t_vals) + far * t_vals
    # 扩展形状为 [num_rays, n_samples]
    z_vals = z_vals.expand(rays_o.shape[:-1] + (n_samples,))

    # --- 核心改进：增加随机扰动 (Jittering / Stratified Sampling) ---
    if model.training:
        # 计算采样点之间的中点作为边界
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)

        # 在每个区间内随机取一个点
        t_rand = torch.rand(z_vals.shape).to(rays_o.device)
        z_vals = lower + (upper - lower) * t_rand
    # -----------------------------------------------------------

    # 计算 3D 采样点坐标: P = O + tD
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    pts_flat = pts.reshape(-1, 3)
    d_flat = rays_d[..., None, :].expand(pts.shape).reshape(-1, 3)

    # 分块处理神经网络查询
    chunk = 1024 * 64  # 稍微调大一点，Instant-NGP 很轻量
    raw = []
    for i in range(0, pts_flat.shape[0], chunk):
        raw.append(model(pts_flat[i:i + chunk], d_flat[i:i + chunk]))
    raw = torch.cat(raw, 0)
    raw = raw.reshape(pts.shape[0], pts.shape[1], n_samples, 4)

    # 提取密度 sigma 和颜色 rgb
    sigma = F.relu(raw[..., 3])
    rgb = raw[..., :3]

    # 计算相邻采样点之间的距离 (用于体渲染积分)
    # 注意：由于增加了抖动，现在的 dists 也是非均匀的，这更真实
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # 最后一段距离设为无穷大
    dists = torch.cat([dists, torch.tensor([1e10]).to(z_vals.device).expand(z_vals[..., :1].shape)], -1)
    # 距离需要乘以射线方向向量的长度（如果是单位向量则不变）
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # 计算透明度 alpha
    alpha = 1. - torch.exp(-sigma * dists)

    # 计算透射率 weights (基于积信号的权重)
    # weights = alpha * transmittance
    transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1)[..., :-1]
    weights = alpha * transmittance

    # 合成最终像素颜色
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    return rgb_map

@torch.no_grad()
def evaluate(args, model, test_dataset, device, i, axes):
    model.eval()
    target_img, target_pose = test_dataset[0]
    target_img, target_pose = target_img.to(device), target_pose.to(device)
    H, W, focal = test_dataset.H, test_dataset.W, test_dataset.focal
    rays_o, rays_d = get_rays(H, W, focal, target_pose)
    rgb_pred = render_rays(model, rays_o, rays_d, near=3.0, far=5.0, n_samples=args.n_samples)
    mse = F.mse_loss(rgb_pred, target_img)
    psnr = -10. * torch.log10(mse)
    ax1, ax2 = axes
    ax1.clear()
    ax1.imshow(target_img.cpu().numpy())
    ax1.set_title("Test GT")
    ax1.axis('off')
    ax2.clear()
    ax2.imshow(rgb_pred.detach().cpu().numpy())
    ax2.set_title(f"Iter {i} Test PSNR: {psnr:.2f}")
    ax2.axis('off')
    save_path = os.path.join(args.exp_dir, f"iter{i}_testpsnr{psnr:.2f}.png")
    plt.savefig(save_path, bbox_inches='tight')
    model.train()
    return psnr.item()

def train(args, model, train_dataset, test_dataset, device):
    os.makedirs(args.exp_dir, exist_ok=True)
    H, W, focal = train_dataset.H, train_dataset.W, train_dataset.focal

    # 推荐：使用更高的初始学习率来配合 Hash Grid
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-15)

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    pbar = tqdm(range(args.n_iters), desc="Training")

    moving_avg_loss = None
    ema_alpha = 0.9

    # --- 改进的学习率衰减函数 ---
    def update_lr(step):
        # 指数衰减：从 args.lr 衰减到 args.lr * 0.1
        # 公式: lr = lr_init * (final_lr_factor ^ (step / total_steps))
        decay_rate = 0.1
        new_lrate = args.lr * (decay_rate ** (step / args.n_iters))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        return new_lrate

    for i in pbar:
        model.train()

        # 1. 更新学习率并获取当前值
        current_lr = update_lr(i)

        # 2. 随机抽取样本训练
        idx = np.random.randint(len(train_dataset))
        target_img, target_pose = train_dataset[idx]
        target_img, target_pose = target_img.to(device), target_pose.to(device)

        rays_o, rays_d = get_rays(H, W, focal, target_pose)
        rgb_pred = render_rays(model, rays_o, rays_d, near=2.0, far=6.0, n_samples=args.n_samples)

        loss = F.mse_loss(rgb_pred, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. 统计滑动平均 Loss
        current_loss_val = loss.item()
        if moving_avg_loss is None:
            moving_avg_loss = current_loss_val
        else:
            moving_avg_loss = ema_alpha * moving_avg_loss + (1 - ema_alpha) * current_loss_val

        # 4. 进度条增加 LR 显示
        pbar.set_postfix({
            "LR": f"{current_lr:.2e}",
            "Loss": f"{current_loss_val:.4f}",
            "Avg": f"{moving_avg_loss:.4f}"
        })

        if i % args.display_int == 0:
            evaluate(args, model, test_dataset, device, i, axes)

    torch.save(model.state_dict(), os.path.join(args.exp_dir, "nerf_final.pth"))
    plt.ioff()
    print(f"✅ 训练完成！最终 Avg Loss: {moving_avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/tiny_nerf_data.npz")
    parser.add_argument("--exp_dir", type=str, default="./runs")

    # --- 推荐的超参数修改 ---
    parser.add_argument("--n_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lrate_decay", type=int, default=250)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--display_int", type=int, default=50)

    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(args.device)
    print(f"🚀 Using device: {device}")

    train_dataset = TinyNeRFDataset(args.data_path, mode='train')
    test_dataset = TinyNeRFDataset(args.data_path, mode='test')

    model = InstantNGP().to(device)
    train(args, model, train_dataset, test_dataset, device)

if __name__ == "__main__":
    main()