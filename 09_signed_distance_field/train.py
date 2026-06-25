import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm
from model import SDFNetwork, ColorNetwork, LearnableVariance
from dataloader import TinySDFDataset
from logger import SDFLogger


def render_rays_sdf(sdf_net, color_net, rays_o, rays_d, near, far, n_samples, s_val,
                    use_random_sample=False, compute_eikonal=True):
    """
    SDF 体渲染（对标 06 NeRF 的 render_rays，核心区别是 SDF→alpha 的转换 + Eikonal Loss）

    返回: rgb_pred [H, W, 3], loss_eikonal (标量，evaluate 时为 0)
    """
    t_vals = torch.linspace(near, far, n_samples).to(rays_o.device)

    # 随机抖动采样（与 06 NeRF 一致）
    if use_random_sample:
        mids = .5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(t_vals.shape).to(rays_o.device)
        t_vals = lower + (upper - lower) * t_rand

    z_vals = t_vals.expand(rays_o.shape[:-1] + (n_samples,))

    # 归一化射线方向（SDF 体渲染必须！z_vals 要对应真实欧氏距离）
    rays_d_norm = rays_d / (rays_d.norm(dim=-1, keepdim=True) + 1e-8)

    # 采样 3D 点（用归一化后的方向）
    pts = rays_o[..., None, :] + rays_d_norm[..., None, :] * z_vals[..., :, None]
    H_img, W_img = rays_o.shape[0], rays_o.shape[1]

    # 视角方向（已归一化）
    dirs_expanded = rays_d_norm[..., None, :].expand_as(pts)

    # Flatten
    flat_pts = pts.reshape(-1, 3)
    flat_dirs = dirs_expanded.reshape(-1, 3)

    # --- 主干前向（不带梯度追踪，快速）---
    chunk = 1024 * 32
    sdf_list, rgb_list = [], []
    for i in range(0, flat_pts.shape[0], chunk):
        p = flat_pts[i:i + chunk]
        d = flat_dirs[i:i + chunk]
        sdf_i, feat_i = sdf_net(p)
        rgb_i = color_net(p, d, feat_i)
        sdf_list.append(sdf_i)
        rgb_list.append(rgb_i)

    sdf_all = torch.cat(sdf_list, 0)  # [H*W*n_samples, 1]
    rgb_all = torch.cat(rgb_list, 0)  # [H*W*n_samples, 3]

    # --- Eikonal Loss ---
    # 混合策略：50% 表面附近点（从射线采样点里抽） + 50% 全局随机点
    # 这样既约束表面处梯度为 1，又约束远离表面的空间
    if compute_eikonal:
        n_eik = 5000
        # 一半来自射线上的采样点（表面附近质量更高）
        n_surface = n_eik // 2
        perm = torch.randperm(flat_pts.shape[0], device=rays_o.device)[:n_surface]
        surface_pts = flat_pts[perm]
        # 一半随机空间点（覆盖射线有效范围，而非只在原点附近）
        n_random = n_eik - n_surface
        # 根据 flat_pts 的实际范围动态决定采样范围
        with torch.no_grad():
            pts_min = flat_pts.min(dim=0)[0]
            pts_max = flat_pts.max(dim=0)[0]
        random_pts = torch.rand(n_random, 3, device=rays_o.device) * (pts_max - pts_min) + pts_min
        # 合并
        eik_pts = torch.cat([surface_pts, random_pts], dim=0)
        eik_pts.requires_grad_(True)
        eik_sdf, _ = sdf_net(eik_pts)
        grad = torch.autograd.grad(
            outputs=eik_sdf, inputs=eik_pts,
            grad_outputs=torch.ones_like(eik_sdf),
            create_graph=True
        )[0]
        loss_eikonal = torch.mean((torch.norm(grad, dim=-1) - 1) ** 2)
    else:
        loss_eikonal = torch.tensor(0.0, device=rays_o.device)

    # Reshape
    sdf_r = sdf_all.reshape(H_img, W_img, n_samples)  # [H, W, S]
    rgb_r = rgb_all.reshape(H_img, W_img, n_samples, 3)  # [H, W, S, 3]

    # 计算alpha
    def get_alpha_volsdf(sdf, s_val, dists):
        # 方式一：VolSDF 风格（基于密度转换，收敛稳）
        sigma = s_val * torch.sigmoid(-sdf * s_val)
        return 1.0 - torch.exp(-sigma * dists)

    def get_alpha_neus(sdf, s_val):
        # 方式二：NeuS 风格（基于 CDF 差值，表面锐利）
        phi = torch.sigmoid(sdf * s_val)
        alpha = torch.clamp((phi[..., :-1] - phi[..., 1:]) / (phi[..., :-1] + 1e-10), min=0.0)
        return torch.cat([alpha, torch.zeros_like(alpha[..., :1])], dim=-1)

    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.full_like(z_vals[..., :1], 1e-2)], -1)
    alpha = get_alpha_volsdf(sdf_r, s_val, dists)  # 推荐用于训练初期快速出形状
    # alpha = get_alpha_neus(sdf_r, s_val)            # 推荐用于精细模型导出

    # 正确的 Transmittance 公式（与 06 NeRF 一致）
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1 - alpha + 1e-7], dim=-1),
        dim=-1
    )[..., :-1]
    weights = alpha * transmittance

    # 体渲染积分
    rgb_pred = torch.sum(weights[..., None] * rgb_r, dim=-2)

    extra_info = {
        "min_sdf": torch.min(sdf_all).item(),
        "max_sdf": torch.max(sdf_all).item(),
        "mean_alpha": torch.mean(alpha).item(),

    }

    return rgb_pred, loss_eikonal, extra_info


@torch.no_grad()
def evaluate(args, sdf_net, color_net, test_dataset, device, step, logger, lr, loss_eikonal_val, s_curr=None):
    sdf_net.eval()
    color_net.eval()

    train_item = test_dataset[0]
    target_img = train_item["image"].to(device)
    rays_o = train_item["rays_o"].to(device)
    rays_d = train_item["rays_d"].to(device)

    # 使用当前训练的 s 值（而非最终目标 s_val），保证 evaluate 和 train 一致
    eval_s = s_curr if s_curr is not None else args.s_val

    rgb_pred, _, _ = render_rays_sdf(
        sdf_net, color_net, rays_o, rays_d,
        near=args.near, far=args.far, n_samples=args.n_samples, s_val=eval_s,
        compute_eikonal=False
    )

    psnr = logger.evaluate_and_log(
        step, rgb_pred, target_img, lr,
        loss_eikonal=loss_eikonal_val,
        s_val=eval_s if isinstance(eval_s, (int, float)) else eval_s.item(),
        sdf_net=sdf_net, device=device
    )

    sdf_net.train()
    color_net.train()
    return psnr


def train(args):
    device = torch.device(args.device) if args.device != "auto" else \
        torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"SDF Training | device: {device}")

    # 数据（与 06 NeRF 相同的 train/test 分离）
    train_dataset = TinySDFDataset(args.data_path, mode='train')
    test_dataset = TinySDFDataset(args.data_path, mode='test')

    # 模型
    sdf_net = SDFNetwork(init_radius=args.init_radius).to(device)
    color_net = ColorNetwork().to(device)
    variance = LearnableVariance(init_val=args.s_val_init).to(device)  # 可学习的 s 参数

    # 优化器：s 参数用独立的较高学习率（NeuS 论文做法）
    optimizer = torch.optim.Adam([
        {'params': sdf_net.parameters(), 'lr': args.lr},
        {'params': color_net.parameters(), 'lr': args.lr},
        {'params': variance.parameters(), 'lr': args.lr * 10.0},  # s 需要快速适应
    ])

    # Logger
    logger = SDFLogger(args.exp_dir, args)
    os.makedirs(args.exp_dir, exist_ok=True)

    def update_lr(step):
        decay_rate = 0.01
        new_lr = args.lr * (decay_rate ** (step / args.n_iters))
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr
        return new_lr

    last_eikonal = 0.0
    pbar = tqdm(range(args.n_iters), desc="SDF Training")

    for step in pbar:
        sdf_net.train()
        color_net.train()

        lr = update_lr(step)
        s_curr = variance.s  # 可学习的 s，不再手动退火

        idx = np.random.randint(len(train_dataset))
        train_item = train_dataset[idx]
        target_img = train_item["image"].to(device)
        rays_o = train_item["rays_o"].to(device)
        rays_d = train_item["rays_d"].to(device)

        rgb_pred, loss_eikonal, extra_info = render_rays_sdf(
            sdf_net, color_net, rays_o, rays_d,
            near=args.near, far=args.far,
            n_samples=args.n_samples, s_val=s_curr,
            use_random_sample=True
        )

        # Loss
        loss_color = F.mse_loss(rgb_pred, target_img)
        loss = loss_color + args.eikonal_weight * loss_eikonal

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（宽松值，只防数值爆炸，不阻碍 Eikonal 收敛）
        torch.nn.utils.clip_grad_norm_(sdf_net.parameters(), max_norm=1.0)
        optimizer.step()

        last_eikonal = loss_eikonal.item()
        psnr_val = -10. * torch.log10(loss_color.detach())
        s_val_now = s_curr.item() if isinstance(s_curr, torch.Tensor) else s_curr

        pbar.set_postfix({
            "Loss": f"{loss.item():.3f}",
            "Lc": f"{loss_color.item():.3f}",
            "Le": f"{loss_eikonal.item():.3f}",
            "LR": f"{lr:.2e}",
            "PSNR": f"{psnr_val.item():.2f}",
            "s": f"{s_val_now:.2f}",
            "d": f"[{extra_info['min_sdf']:.2f}, {extra_info['max_sdf']:.2f}]",
            "alpha": f"{extra_info['mean_alpha']:.2f}"
        })

        if step % args.display_int == 0:
            evaluate(args, sdf_net, color_net, test_dataset, device, step, logger, lr, last_eikonal, s_val_now)

    # 保存
    torch.save({
        'sdf_net': sdf_net.state_dict(),
        'color_net': color_net.state_dict(),
        'variance': variance.state_dict(),
    }, os.path.join(args.exp_dir, "sdf_final.pth"))
    print(f"Done! Saved: {args.exp_dir}/sdf_final.pth")


def main():
    parser = argparse.ArgumentParser(description="SDF/NeuS Trainer")
    parser.add_argument("--data_path", type=str, default="../data/tiny_nerf_data.npz")
    parser.add_argument("--exp_dir", type=str, default="./runs/sdf_default")
    parser.add_argument("--n_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=2.2)
    parser.add_argument("--init_radius", type=float, default=0.5, help="初始球体 SDF 半径")
    parser.add_argument("--s_val", type=float, default=50.0, help="SDF→alpha 的 s 参数（最终值）")
    parser.add_argument("--s_val_init", type=float, default=5.0, help="SDF→alpha 的 s 参数（初始值）")
    parser.add_argument("--eikonal_weight", type=float, default=0.1, help="Eikonal Loss 权重")
    parser.add_argument("--display_int", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
