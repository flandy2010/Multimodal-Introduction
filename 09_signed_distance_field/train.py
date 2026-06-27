import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm
from model import SDFNetwork, ColorNetwork, LearnableVariance
from dataloader import TinySDFDataset
from logger import SDFLogger


def compute_normals_autograd(sdf_net, flat_pts, chunk):
    """法线方式一（精确）：autograd 求 SDF 对坐标的梯度。"""
    sdf_list, feat_list, normal_list = [], [], []
    for i in range(0, flat_pts.shape[0], chunk):
        p = flat_pts[i:i + chunk].detach().requires_grad_(True)
        with torch.enable_grad():
            sdf_i, feat_i = sdf_net.sdf_and_feature(p)
            grad_i = torch.autograd.grad(
                outputs=sdf_i, inputs=p,
                grad_outputs=torch.ones_like(sdf_i),
                create_graph=False, retain_graph=True, only_inputs=True
            )[0]
            normals_i = F.normalize(grad_i, p=2, dim=-1)
        sdf_list.append(sdf_i)
        feat_list.append(feat_i)
        normal_list.append(normals_i.detach())
    return (
        torch.cat(sdf_list, 0),
        torch.cat(feat_list, 0),
        torch.cat(normal_list, 0),
    )


def compute_normals_finite_diff(sdf_net, flat_pts, chunk, eps=1e-3):
    """法线方式二（近似）：有限差分，6 次 no_grad 前向近似梯度。"""
    offsets = torch.eye(3, device=flat_pts.device) * eps
    normal_list, feat_list = [], []
    with torch.no_grad():
        for i in range(0, flat_pts.shape[0], chunk):
            p = flat_pts[i:i + chunk]
            _, feat_i = sdf_net.sdf_and_feature(p)
            feat_list.append(feat_i)
            grad_fd = []
            for k in range(3):
                sdf_pos = sdf_net.sdf(p + offsets[k])
                sdf_neg = sdf_net.sdf(p - offsets[k])
                grad_fd.append((sdf_pos - sdf_neg) / (2 * eps))
            normals_i = F.normalize(torch.cat(grad_fd, dim=-1), p=2, dim=-1)
            normal_list.append(normals_i)
    normals_all = torch.cat(normal_list, 0)  # [N, 3]
    feat_all = torch.cat(feat_list, 0)       # [N, W] 无梯度

    # 带梯度的 SDF 单独算一次（feat 已 detach，只为 loss 反传）
    sdf_list = []
    for i in range(0, flat_pts.shape[0], chunk):
        p = flat_pts[i:i + chunk]
        sdf_i = sdf_net.sdf(p)
        sdf_list.append(sdf_i)
    return (
        torch.cat(sdf_list, 0),  # [N, 1] 带梯度
        feat_all,                # [N, W] 无梯度（传给 color_net 够用）
        normals_all,             # [N, 3] 无梯度
    )


def render_rays_sdf(sdf_net, color_net, rays_o, rays_d, near, far, n_samples, s_val,
                    use_random_sample=False, compute_eikonal=True,
                    normal_mode="autograd"):
    """
    SDF 体渲染。支持两种输入格式（自动识别）：
      - 全图模式：rays_o/rays_d 为 [H, W, 3]，输出 rgb_pred [H, W, 3]（评估用）
      - 随机射线模式：rays_o/rays_d 为 [B, 3]，输出 rgb_pred [B, 3]（训练用）

    Eikonal 采样策略（参考 NeuS 官方）：
      - 直接复用渲染中已有的采样点，不额外分配，零额外开销
      - 仅约束 pts_norm < 1.2 的点（单位球内，有效 SDF 区域）

    参数:
        normal_mode: "autograd"（精确，H20 推荐）| "finite_diff"（近似，P800 推荐）
    """
    is_image_mode = rays_o.dim() == 3   # True: [H, W, 3]  False: [B, 3]

    if is_image_mode:
        H_img, W_img = rays_o.shape[0], rays_o.shape[1]
        flat_o = rays_o.reshape(-1, 3)
        flat_d = rays_d.reshape(-1, 3)
    else:
        flat_o = rays_o   # [B, 3]
        flat_d = rays_d

    B = flat_o.shape[0]  # 总射线数

    t_vals = torch.linspace(near, far, n_samples).to(flat_o.device)
    if use_random_sample:
        mids  = .5 * (t_vals[1:] + t_vals[:-1])
        upper = torch.cat([mids, t_vals[-1:]], -1)
        lower = torch.cat([t_vals[:1], mids], -1)
        t_vals = lower + (upper - lower) * torch.rand(B, n_samples, device=flat_o.device)
    else:
        t_vals = t_vals.expand(B, n_samples)   # [B, S]

    # 归一化射线方向
    flat_d_norm = flat_d / (flat_d.norm(dim=-1, keepdim=True) + 1e-8)

    # 采样 3D 点  [B, S, 3]
    pts = flat_o[:, None, :] + flat_d_norm[:, None, :] * t_vals[:, :, None]
    flat_pts = pts.reshape(-1, 3)           # [B*S, 3]
    flat_dirs = flat_d_norm.unsqueeze(1).expand_as(pts).reshape(-1, 3)  # [B*S, 3]

    # --- 主干前向：SDF + feature + 法线 ---
    chunk = 1024 * 32
    if normal_mode == "finite_diff":
        sdf_all, feat_all, normals_all = compute_normals_finite_diff(sdf_net, flat_pts, chunk)
    else:
        sdf_all, feat_all, normals_all = compute_normals_autograd(sdf_net, flat_pts, chunk)

    # 颜色网络
    rgb_list = []
    for i in range(0, flat_pts.shape[0], chunk):
        rgb_i = color_net(flat_pts[i:i+chunk], flat_dirs[i:i+chunk],
                          feat_all[i:i+chunk], normals_all[i:i+chunk])
        rgb_list.append(rgb_i)
    rgb_all = torch.cat(rgb_list, 0)   # [B*S, 3]

    # --- Eikonal Loss（NeuS 官方：直接复用渲染采样点，不额外采样）---
    if compute_eikonal:
        # 在渲染采样点上算 Eikonal，仅约束单位球内的点（pts_norm < 1.2）
        eik_pts = flat_pts.detach().requires_grad_(True)
        eik_sdf = sdf_net.sdf(eik_pts)
        grad = torch.autograd.grad(
            outputs=eik_sdf, inputs=eik_pts,
            grad_outputs=torch.ones_like(eik_sdf),
            create_graph=True
        )[0]  # [B*S, 3]
        pts_norm = flat_pts.detach().norm(dim=-1)              # [B*S]
        inside_sphere = (pts_norm < 1.2).float()               # 仅约束有效区域
        grad_norm_sq = (grad.norm(dim=-1) - 1.0) ** 2         # [B*S]
        denom = inside_sphere.sum() + 1e-5
        loss_eikonal = (inside_sphere * grad_norm_sq).sum() / denom
    else:
        loss_eikonal = torch.tensor(0.0, device=flat_o.device)

    # Reshape 回 [B, S, ...]
    sdf_r = sdf_all.reshape(B, n_samples)        # [B, S]
    rgb_r = rgb_all.reshape(B, n_samples, 3)     # [B, S, 3]

    # SDF → alpha（VolSDF 风格）
    dists = torch.cat([
        t_vals[:, 1:] - t_vals[:, :-1],
        torch.full((B, 1), 1e-2, device=flat_o.device)
    ], dim=-1)  # [B, S]
    sigma = s_val * torch.sigmoid(-sdf_r * s_val)
    alpha = 1.0 - torch.exp(-sigma * dists)      # [B, S]

    # Transmittance & 体渲染
    transmittance = torch.cumprod(
        torch.cat([torch.ones(B, 1, device=flat_o.device), 1 - alpha + 1e-7], dim=-1),
        dim=-1
    )[:, :-1]    # [B, S]
    weights = alpha * transmittance              # [B, S]
    rgb_pred_flat = (weights.unsqueeze(-1) * rgb_r).sum(dim=1)   # [B, 3]

    # 还原形状
    if is_image_mode:
        rgb_pred = rgb_pred_flat.reshape(H_img, W_img, 3)
    else:
        rgb_pred = rgb_pred_flat  # [B, 3]

    # 诊断信息（仅全图模式填充有意义的统计）
    top20_alpha = weights.topk(min(20, n_samples), dim=-1).values.mean().item()
    extra_info = {
        "min_sdf":   sdf_all.min().item(),
        "max_sdf":   sdf_all.max().item(),
        "top20_alpha": top20_alpha,
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
        compute_eikonal=False, normal_mode=args.normal_mode
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
    # NeuS 原版参数：d_in=3, d_out=257(1+256), d_hidden=256, n_layers=8,
    # skip_in=(4,), multires=6, bias=init_radius, scale=1
    sdf_net = SDFNetwork(
        d_in=3, d_out=257, d_hidden=256, n_layers=8,
        skip_in=(4,), multires=6, bias=args.init_radius, scale=1.0,
        geometric_init=True, weight_norm=True
    ).to(device)
    color_net = ColorNetwork().to(device)
    variance = LearnableVariance(init_val=args.s_val_init).to(device)  # 可学习的 s 参数

    # 优化器：s 参数用独立的较高学习率（NeuS 论文做法）
    optimizer = torch.optim.Adam([
        {'params': sdf_net.parameters(), 'lr': args.lr, 'base': 1},
        {'params': color_net.parameters(), 'lr': args.lr, 'base': 1},
        {'params': variance.parameters(), 'lr': args.lr, 'base': 10},  # s 需要快速适应
    ])

    # Logger
    logger = SDFLogger(args.exp_dir, args)
    os.makedirs(args.exp_dir, exist_ok=True)

    def update_lr(step):
        decay_rate = 0.01
        new_lr = args.lr * (decay_rate ** (step / args.n_iters))
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr * pg['base']
        return new_lr

    last_eikonal = 0.0
    n_train = len(train_dataset)
    pbar = tqdm(range(args.n_iters), desc="SDF Training")

    for step in pbar:
        sdf_net.train()
        color_net.train()

        lr = update_lr(step)
        s_curr = variance.s  # 可学习的 s，不再手动退火

        # 随机选一张训练图，再随机采样 batch_size 条射线（NeuS 官方做法）
        idx = np.random.randint(n_train)
        rays_o, rays_d, target_rgb = train_dataset.gen_random_rays(
            idx, args.batch_size, device=device
        )

        rgb_pred, loss_eikonal, extra_info = render_rays_sdf(
            sdf_net, color_net, rays_o, rays_d,
            near=args.near, far=args.far,
            n_samples=args.n_samples, s_val=s_curr,
            use_random_sample=True, normal_mode=args.normal_mode
        )

        # Loss（target_rgb 已是 [B, 3]，与 rgb_pred [B, 3] 对应）
        loss_color = F.mse_loss(rgb_pred, target_rgb)
        # 前 1000 步 Eikonal 热身（较小权重），之后全量
        eik_w = args.eikonal_weight * (0.1 if step < 1000 else 1.0)
        loss = loss_color + eik_w * loss_eikonal

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（宽松值，只防数值爆炸）
        torch.nn.utils.clip_grad_norm_(sdf_net.parameters(), max_norm=1.0)
        optimizer.step()

        last_eikonal = loss_eikonal.item()
        psnr_val = -10. * torch.log10(loss_color.detach())
        s_val_now = s_curr.item() if isinstance(s_curr, torch.Tensor) else s_curr

        pbar.set_postfix({
            "Lc": f"{loss_color.item():.4f}",
            "Le": f"{loss_eikonal.item():.3f}",
            "LR": f"{lr:.2e}",
            "PSNR": f"{psnr_val.item():.2f}",
            "s": f"{s_val_now:.2f}",
            "d": f"[{extra_info['min_sdf']:.2f}, {extra_info['max_sdf']:.2f}]",
            "top20_alpha": f"{extra_info['top20_alpha']:.2f}"
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
    parser.add_argument("--batch_size", type=int, default=512,
                        help="每步随机采样的射线数（NeuS 官方默认 512，越大越慢但梯度更稳）")
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
    parser.add_argument("--normal_mode", type=str, default="autograd",
                        choices=["autograd", "finite_diff"],
                        help="法线计算方式: autograd=精确梯度(H20推荐), finite_diff=有限差分(P800推荐)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
