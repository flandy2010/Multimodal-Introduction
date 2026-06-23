import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from model import NeRF
from dataloader import TinyNeRFDataset


def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    # i, j是一个[W, H]矩阵M，其中M[i, :] = i, M[:, j] = j

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


def render_rays(model, rays_o, rays_d, near, far, n_samples):

    # near, far表示物体距离相机的距离范围
    t_vals = torch.linspace(near, far, n_samples).to(rays_o.device)
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

    sigma = F.relu(raw[..., 3])
    rgb = raw[..., :3]
    dists_pad = torch.tensor([1e10]).to(z_vals.device).expand(z_vals[..., :1].shape)
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], dists_pad], -1)

    # exp(-sigma * dists)表示光穿过介质后的剩余能量
    alpha = 1. - torch.exp(-sigma * dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1)[..., :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    return rgb_map


@torch.no_grad()
def evaluate(args, model, test_dataset, device, i, axes):
    """
    在测试集上评估模型
    """
    model.eval()  # 切换到评估模式

    # 始终选择测试集的第一张图作为固定观察视角
    target_img, target_pose = test_dataset[0]
    target_img, target_pose = target_img.to(device), target_pose.to(device)
    H, W, focal = test_dataset.H, test_dataset.W, test_dataset.focal

    # 渲染
    rays_o, rays_d = get_rays(H, W, focal, target_pose)
    rgb_pred = render_rays(model, rays_o, rays_d, near=2.0, far=6.0, n_samples=args.n_samples)

    # 计算 PSNR
    mse = F.mse_loss(rgb_pred, target_img)
    psnr = -10. * torch.log10(mse)

    # 可视化输出
    ax1, ax2 = axes
    ax1.clear()
    ax1.imshow(target_img.cpu().numpy())
    ax1.set_title("Test GT")
    ax1.axis('off')

    ax2.clear()
    ax2.imshow(rgb_pred.detach().cpu().numpy())
    ax2.set_title(f"Iter {i} Test PSNR: {psnr:.2f}")
    ax2.axis('off')

    # 保存图片
    save_path = os.path.join(args.exp_dir, f"iter{i}_testpsnr{psnr:.2f}.png")
    plt.savefig(save_path, bbox_inches='tight')
    # plt.pause(0.01)

    model.train()  # 切换回训练模式
    return psnr.item()


def train(args, model, train_dataset, test_dataset, device):

    os.makedirs(args.exp_dir, exist_ok=True)
    H, W, focal = train_dataset.H, train_dataset.W, train_dataset.focal
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    pbar = tqdm(range(args.n_iters), desc="Training")

    def update_lr(iter):
        return args.lr * (0.1 ** (iter / (args.n_iters * args.lrate_decay)))

    for i in pbar:
        # 训练采样
        idx = np.random.randint(len(train_dataset))
        target_img, target_pose = train_dataset[idx]
        target_img, target_pose = target_img.to(device), target_pose.to(device)

        # 学习率更新
        lr = update_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向与优化
        rays_o, rays_d = get_rays(H, W, focal, target_pose)
        # rays_o.shape = [100, 100, 3]
        # rays_d.shape = [100, 100, 3]
        rgb_pred = render_rays(model, rays_o, rays_d, near=2.0, far=6.0, n_samples=args.n_samples)

        loss = F.mse_loss(rgb_pred, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 进度条显示
        pbar.set_postfix({"LR": f"{lr:.2e}", "Loss": f"{loss.item():.4f}"})

        # 定期调用评估函数
        if i % args.display_int == 0:
            evaluate(args, model, test_dataset, device, i, axes)

    torch.save(model.state_dict(), os.path.join(args.exp_dir, "nerf_final.pth"))
    plt.ioff()
    print("✅ 训练完成！")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/tiny_nerf_data.npz")
    parser.add_argument("--exp_dir", type=str, default="./runs")
    parser.add_argument("--n_iters", type=int, default=4000)  # 建议增加到1万次
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lrate_decay", type=int, default=250)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--display_int", type=int, default=200)
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