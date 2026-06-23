import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from model import NeRF
from dataloader import TinyNeRFDataset
from logger import NeRFLogger


def get_rays(H, W, focal, c2w, use_random_offset=False):

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    # i, j是一个[W, H]矩阵M，其中M[i, :] = i, M[:, j] = j

    if use_random_offset:
        # 给像素坐标 i, j 增加 [-0.5, 0.5] 的随机偏移
        # 这一步能强迫模型学习像素内部的颜色变化，从而“挤出”积木的颗粒感
        i = i + torch.rand_like(i) - 0.5
        j = j + torch.rand_like(j) - 0.5

    i, j = i.t(), j.t()  # 这里转置了
    # dirs.shape = [H, W, 3]，看起来根据焦距挪到距离相机单位距离处的坐标体系，dirs[i][j]对应图中(i, j)位置像素点变化后的空间坐标
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    dirs = dirs.to(c2w)
    # rays_d.shape = [H, W, 3]，表示光线的方向，先相乘再求和，本质上是矩阵乘法（即坐标系变换）
    # [H, W, 3] -> [H, W, 1, 3] -> [H, W 3, 3] -> [H, W, 3]
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    # rays_o.shape = [H, W, 3], 表示光线的原点
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def render_rays(model, rays_o, rays_d, near, far, n_samples, use_random_sample=False, use_density_noise=False):

    # near, far表示物体距离相机的距离范围
    t_vals = torch.linspace(near, far, n_samples).to(rays_o.device)

    # 添加采样点随机偏离
    if use_random_sample:
        mids = .5 * (t_vals[...,1:] + t_vals[...,:-1])
        upper = torch.cat([mids, t_vals[...,-1:]], -1)
        lower = torch.cat([t_vals[...,:1], mids], -1)
        t_rand = torch.rand(t_vals.shape).to(rays_o.device)
        t_vals = lower + (upper - lower) * t_rand

    z_vals = t_vals.expand(rays_o.shape[:-1] + (n_samples,))

    # 从rays_o出发，沿着rays_d方向，每隔一段进行一次采样
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts_flat = pts.reshape(-1, 3)
    d_flat = rays_d[..., None, :].expand(pts.shape).reshape(-1, 3)

    # 分chunk统计模型的预测结果
    chunk = 1024 * 32
    raw = []
    for i in range(0, pts_flat.shape[0], chunk):
        raw.append(model(pts_flat[i:i + chunk], d_flat[i:i + chunk]))
    raw = torch.cat(raw, 0)
    raw = raw.reshape(pts.shape[0], pts.shape[1], n_samples, 4)

    # 加入Density Noise正则化，用于消除“雾气”的散点
    raw_sigma = raw[..., 3]
    if use_density_noise:
        # 注入标准差为 1.0 的高斯噪声
        noise = torch.randn_like(raw_sigma) * 1.0
        sigma = F.relu(raw_sigma + noise)
    else:
        sigma = F.relu(raw_sigma)

    rgb = raw[..., :3]
    dists_pad = torch.tensor([1e10]).to(z_vals.device).expand(z_vals[..., :1].shape)
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], dists_pad], -1)

    # exp(-sigma * dists)表示光穿过介质后的剩余能量
    alpha = 1. - torch.exp(-sigma * dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1)[..., :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    return rgb_map


@torch.no_grad()
def evaluate(args, model, test_dataset, device, i, logger, current_lr):

    model.eval()
    target_img, target_pose = test_dataset[0]
    target_img, target_pose = target_img.to(device), target_pose.to(device)
    H, W, focal = test_dataset.H, test_dataset.W, test_dataset.focal

    rays_o, rays_d = get_rays(H, W, focal, target_pose)
    rgb_pred = render_rays(model, rays_o, rays_d, near=args.near, far=args.far, n_samples=args.n_samples)

    # 使用 logger 完成所有评测、记录、可视化
    psnr = logger.evaluate_and_log(i, rgb_pred, target_img, current_lr)

    model.train()
    return psnr


def train(args, model, train_dataset, test_dataset, device):
    os.makedirs(args.exp_dir, exist_ok=True)
    H, W, focal = train_dataset.H, train_dataset.W, train_dataset.focal
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 初始化 Logger
    logger = NeRFLogger(args.exp_dir, args)

    pbar = tqdm(range(args.n_iters), desc="Training")

    # --- 修改后的学习率衰减：100倍衰减 (从 5e-4 降到 5e-6) ---
    def update_lr(step):
        # 指数衰减公式: lr = lr_init * (gamma ^ (step / total_steps))
        decay_rate = 0.01
        new_lrate = args.lr * (decay_rate ** (step / args.n_iters))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        return new_lrate

    for i in pbar:
        model.train()
        idx = np.random.randint(len(train_dataset))
        target_img, target_pose = train_dataset[idx]
        target_img, target_pose = target_img.to(device), target_pose.to(device)

        lr = update_lr(i)

        rays_o, rays_d = get_rays(H, W, focal, target_pose, use_random_offset=True)

        # 训练时开启随机抖动采样 (rand=True)
        rgb_pred = render_rays(model,
                               rays_o,
                               rays_d,
                               near=args.near,
                               far=args.far,
                               n_samples=args.n_samples,
                               use_random_sample=True,
                               use_density_noise=(lr > float(1e-5))
                               )

        loss = F.mse_loss(rgb_pred, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 进度条增加 PSNR 实时显示
        psnr_val = -10. * torch.log10(loss.detach())
        pbar.set_postfix({
            "LR": f"{lr:.2e}",
            "Loss": f"{loss.item():.4f}",
            "PSNR": f"{psnr_val.item():.2f}"
        })

        if i % args.display_int == 0:
            evaluate(args, model, test_dataset, device, i, logger, lr)

    torch.save(model.state_dict(), os.path.join(args.exp_dir, "nerf_final.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/tiny_nerf_data.npz")
    parser.add_argument("--exp_dir", type=str, default="./runs")
    parser.add_argument("--n_iters", type=int, default=10000)  # 调大步数
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_samples", type=int, default=128)  # 调大采样
    parser.add_argument("--display_int", type=int, default=500)

    # --- 新增参数 ---
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)

    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) \
        if args.device == "auto" else torch.device(args.device)

    # 加载两个数据集
    train_dataset = TinyNeRFDataset(args.data_path, mode='train')
    test_dataset = TinyNeRFDataset(args.data_path, mode='test')

    model = NeRF().to(device)
    train(args, model, train_dataset, test_dataset, device)


if __name__ == "__main__":
    main()