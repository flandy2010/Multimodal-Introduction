import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm
from model import SDFNetwork, ColorNetwork
from dataloader import TinySDFDataset
from logger import SDFLogger


def get_rays(H, W, focal, c2w):
    """与 06 NeRF 相同的射线生成"""
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    i, j = i.t(), j.t()
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1).to(c2w.device)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


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

    # 采样 3D 点
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    H_img, W_img = rays_o.shape[0], rays_o.shape[1]

    # 归一化视角方向（用于颜色网络）
    dirs_norm = rays_d / (rays_d.norm(dim=-1, keepdim=True) + 1e-8)
    dirs_expanded = dirs_norm[..., None, :].expand_as(pts)

    # Flatten
    flat_pts = pts.reshape(-1, 3)
    if compute_eikonal:
        flat_pts.requires_grad_(True)  # Eikonal Loss 需要梯度
    flat_dirs = dirs_expanded.reshape(-1, 3)

    # 分 chunk 前向（防止 OOM）
    chunk = 1024 * 32
    sdf_list, rgb_list = [], []
    for i in range(0, flat_pts.shape[0], chunk):
        p = flat_pts[i:i + chunk]
        d = flat_dirs[i:i + chunk]
        sdf_i, feat_i = sdf_net(p)
        rgb_i = color_net(p, d, feat_i)
        sdf_list.append(sdf_i)
        rgb_list.append(rgb_i)

    sdf_all = torch.cat(sdf_list, 0)   # [H*W*n_samples, 1]
    rgb_all = torch.cat(rgb_list, 0)    # [H*W*n_samples, 3]

    # --- Eikonal Loss（仅训练时计算）---
    if compute_eikonal:
        grad = torch.autograd.grad(
            outputs=sdf_all, inputs=flat_pts,
            grad_outputs=torch.ones_like(sdf_all),
            create_graph=True
        )[0]
        loss_eikonal = torch.mean((torch.norm(grad, dim=-1) - 1) ** 2)
    else:
        loss_eikonal = torch.tensor(0.0, device=rays_o.device)

    # Reshape
    sdf_r = sdf_all.reshape(H_img, W_img, n_samples)       # [H, W, S]
    rgb_r = rgb_all.reshape(H_img, W_img, n_samples, 3)     # [H, W, S, 3]

    # 计算alpha
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e-2]).expand(dists[..., :1].shape).to(rays_o.device)], -1)

    sigma = s_val * torch.sigmoid(-sdf_r * s_val)
    alpha = 1.0 - torch.exp(-sigma * dists)

    # 正确的 Transmittance 公式（与 06 NeRF 一致）
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1 - alpha + 1e-7], dim=-1),
        dim=-1
    )[..., :-1]
    weights = alpha * transmittance

    # 体渲染积分
    rgb_pred = torch.sum(weights[..., None] * rgb_r, dim=-2)

    return rgb_pred, loss_eikonal


@torch.no_grad()
def evaluate(args, sdf_net, color_net, test_dataset, device, step, logger, lr, loss_eikonal_val):
    sdf_net.eval()
    color_net.eval()

    target_img, target_pose = test_dataset[0]
    target_img, target_pose = target_img.to(device), target_pose.to(device)
    H, W, focal = test_dataset.H, test_dataset.W, test_dataset.focal

    rays_o, rays_d = get_rays(H, W, focal, target_pose)
    rgb_pred, _ = render_rays_sdf(
        sdf_net, color_net, rays_o, rays_d,
        near=args.near, far=args.far, n_samples=args.n_samples, s_val=args.s_val,
        compute_eikonal=False  # evaluate 时不算 Eikonal（在 no_grad 下无法求梯度）
    )

    psnr = logger.evaluate_and_log(
        step, rgb_pred, target_img, lr,
        loss_eikonal=loss_eikonal_val,
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
    H, W, focal = train_dataset.H, train_dataset.W, train_dataset.focal

    # 模型
    sdf_net = SDFNetwork(init_radius=args.init_radius).to(device)
    color_net = ColorNetwork().to(device)

    # 优化器（与 06 NeRF 类似的指数衰减）
    optimizer = torch.optim.Adam(
        list(sdf_net.parameters()) + list(color_net.parameters()),
        lr=args.lr
    )

    # Logger
    logger = SDFLogger(args.exp_dir, args)
    os.makedirs(args.exp_dir, exist_ok=True)

    def update_lr(step):
        decay_rate = 0.01
        new_lr = args.lr * (decay_rate ** (step / args.n_iters))
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr
        return new_lr

    # --- s 值退火：从软到硬 ---
    def get_s(step):
        # 从 s_val_init 线性增长到 s_val
        progress = min(step / (args.n_iters * 0.8), 1.0)
        return args.s_val_init + (args.s_val - args.s_val_init) * progress

    last_eikonal = 0.0
    pbar = tqdm(range(args.n_iters), desc="SDF Training")

    for step in pbar:
        sdf_net.train()
        color_net.train()

        lr = update_lr(step)
        s_curr = get_s(step)

        idx = np.random.randint(len(train_dataset))
        target_img, target_pose = train_dataset[idx]
        target_img, target_pose = target_img.to(device), target_pose.to(device)

        rays_o, rays_d = get_rays(H, W, focal, target_pose)

        rgb_pred, loss_eikonal = render_rays_sdf(
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
        optimizer.step()

        last_eikonal = loss_eikonal.item()
        psnr_val = -10. * torch.log10(loss_color.detach())

        pbar.set_postfix({
            "LR": f"{lr:.2e}",
            "PSNR": f"{psnr_val.item():.2f}",
            "Eik": f"{last_eikonal:.4f}",
            "s": f"{s_curr:.1f}",
        })

        if step % args.display_int == 0:
            evaluate(args, sdf_net, color_net, test_dataset, device, step, logger, lr, last_eikonal)

    # 保存
    torch.save({
        'sdf_net': sdf_net.state_dict(),
        'color_net': color_net.state_dict(),
    }, os.path.join(args.exp_dir, "sdf_final.pth"))
    print(f"Done! Saved: {args.exp_dir}/sdf_final.pth")


def main():

    parser = argparse.ArgumentParser(description="SDF/NeuS Trainer")
    parser.add_argument("--data_path", type=str, default="../data/tiny_nerf_data.npz")
    parser.add_argument("--exp_dir", type=str, default="./runs/sdf_default")
    parser.add_argument("--n_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--init_radius", type=float, default=4.0, help="初始球体 SDF 半径")
    parser.add_argument("--s_val", type=float, default=50.0, help="SDF→alpha 的 s 参数（最终值）")
    parser.add_argument("--s_val_init", type=float, default=5.0, help="SDF→alpha 的 s 参数（初始值）")
    parser.add_argument("--eikonal_weight", type=float, default=0.1, help="Eikonal Loss 权重")
    parser.add_argument("--display_int", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
