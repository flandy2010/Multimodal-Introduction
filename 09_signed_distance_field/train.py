import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm
from model import SDFNetwork, ColorNetwork, LearnableVariance
from dataloader import TinySDFDataset
from logger import SDFLogger




def get_alpha_volsdf(sdf, s_val, dists):
    """VolSDF 风格：基于密度转换，收敛稳定。

    sigma = s * sigmoid(-sdf * s)      # 密度
    alpha = 1 - exp(-sigma * dist)     # 透射率

    s 含义：控制密度从表面（sdf=0）向外衰减的陡峭程度。
    s 越大 → 密度越集中在 sdf≈0 → 表面越锐利，但早期容易"过脆"卡在局部最优。
    推荐 s 从 10 起步，训练中让网络自主学习增大（LearnableVariance）。
    """
    sigma = s_val * torch.sigmoid(-sdf * s_val)
    return 1.0 - torch.exp(-sigma * dists)


def get_alpha_neus(sdf, s_val):
    """NeuS 风格：基于 CDF 差值，表面天然锐利。

    Φ_s(x) = sigmoid(s * x)                          # CDF
    alpha = (Φ_s(sdf_i) - Φ_s(sdf_i+1)) / Φ_s(sdf_i)   # 不透明度

    s 含义：1/s 是 SDF 空间中表面模糊区域的宽度（标准差）。
    s 越大 → 表面过渡越窄 → alpha 越集中在 sdf=0 附近。
    推荐 s 从 3~5 起步（对应 σ=0.2~0.33），训练中网络自主学习增大。

    注意：NeuS 公式天然保证 ∑alpha = 1（每条射线总不透明度为 1），
    适合 unbounded 场景和无背景的物体重建。
    """
    phi = torch.sigmoid(sdf * s_val)                              # [B, S]
    alpha = (phi[..., :-1] - phi[..., 1:]) / (phi[..., :-1] + 1e-10)
    alpha = torch.clamp(alpha, min=0.0, max=1.0)
    return torch.cat([alpha, torch.zeros_like(alpha[..., :1])], dim=-1)


def render_rays_sdf(sdf_net, color_net, rays_o, rays_d, near, far, n_samples, s_val,
                    use_random_sample=False, compute_eikonal=True,
                    alpha_mode="volsdf"):
    """
    SDF 体渲染。支持两种输入格式（自动识别）：
      - 训练模式：rays_o/rays_d 为 [B, 3]，输出 rgb_pred [B, 3]
      - 评估模式：rays_o/rays_d 为 [H, W, 3]，输出 rgb_pred [H, W, 3]

    alpha_mode: "volsdf"（密度式，收敛稳）| "neus"（CDF式，表面锐利）
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

    # --- 主干前向（单次 sdf_net forward，法线和 Eikonal 出自同一张图）---
    eik_pts = flat_pts.detach().requires_grad_(True)

    if compute_eikonal:
        # 训练：全图支持，create_graph 用于 Eikonal 双次反向
        out = sdf_net(eik_pts)                              # [N, 257]
        sdf_all  = out[:, :1]
        feat_all = out[:, 1:]
        grad_all = torch.autograd.grad(
            sdf_all, eik_pts, torch.ones_like(sdf_all),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_norm  = grad_all.norm(dim=-1)
        pts_norm   = flat_pts.detach().norm(dim=-1)
        inside     = (pts_norm < 1.2).float()
        loss_eikonal = (inside * (grad_norm - 1.0) ** 2).sum() / (inside.sum() + 1e-5)
        eik_diag = {
            "inside_frac": inside.mean().item(),
            "grad_mean":   (inside * grad_norm.detach()).sum().item() / (inside.sum().item() + 1e-5),
            "grad_max":    (inside * grad_norm.detach()).max().item(),
            "count":       int(inside.sum().item()),
        }
    else:
        # 评估（no_grad 上下文）：临时 enable_grad 求法线，不做 Eikonal
        with torch.enable_grad():
            out = sdf_net(eik_pts)
            sdf_all  = out[:, :1]
            feat_all = out[:, 1:]
            grad_all = torch.autograd.grad(
                sdf_all, eik_pts, torch.ones_like(sdf_all),
                create_graph=False, retain_graph=False, only_inputs=True
            )[0]
        loss_eikonal = torch.tensor(0.0, device=flat_o.device)
        eik_diag = {}

    normals_all = F.normalize(grad_all.detach(), p=2, dim=-1)

    # 颜色网络（法线已 detach，不会反向影响 SDF 几何）
    chunk = 1024 * 32
    rgb_list = []
    for i in range(0, flat_pts.shape[0], chunk):
        rgb_i = color_net(flat_pts[i:i+chunk], flat_dirs[i:i+chunk],
                          feat_all[i:i+chunk], normals_all[i:i+chunk])
        rgb_list.append(rgb_i)
    rgb_all = torch.cat(rgb_list, 0)   # [B*S, 3]

    # Reshape 回 [B, S, ...]
    sdf_r = sdf_all.reshape(B, n_samples)        # [B, S]
    rgb_r = rgb_all.reshape(B, n_samples, 3)     # [B, S, 3]

    # SDF → alpha
    dists = torch.cat([
        t_vals[:, 1:] - t_vals[:, :-1],
        torch.full((B, 1), 1e-2, device=flat_o.device)
    ], dim=-1)  # [B, S]

    if alpha_mode == "neus":
        alpha = get_alpha_neus(sdf_r, s_val)
    else:
        alpha = get_alpha_volsdf(sdf_r, s_val, dists)

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
        "eik_diag":  eik_diag,
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

    # 使用当前训练的 s 值，保证 evaluate 和 train 一致
    rgb_pred, _, _ = render_rays_sdf(
        sdf_net, color_net, rays_o, rays_d,
        near=args.near, far=args.far, n_samples=args.n_samples, s_val=s_curr,
        compute_eikonal=False, alpha_mode=args.alpha_mode
    )

    psnr = logger.evaluate_and_log(
        step, rgb_pred, target_img, lr,
        loss_eikonal=loss_eikonal_val,
        s_val=s_curr.item() if isinstance(s_curr, torch.Tensor) else s_curr,
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
        """NeuS 官方 Cosine Annealing + Warmup 学习率调度。

        Warmup 阶段（step < warm_up_end）：lr 从 0 线性增长到 args.lr
        退火阶段（step >= warm_up_end）：Cosine annealing 衰减到 args.lr * lr_alpha

        与官方唯一差异：originally官方 s_cost 组的 base=10，此处保留。
        """
        if step < args.warm_up_end:
            factor = step / args.warm_up_end                         # 线性增长 0→1
        else:
            progress = (step - args.warm_up_end) / (args.n_iters - args.warm_up_end)
            factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - args.lr_alpha) + args.lr_alpha

        new_lr = args.lr * factor
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr * pg['base']
        return new_lr

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
            use_random_sample=True, alpha_mode=args.alpha_mode
        )

        # Loss（target_rgb 已是 [B, 3]，与 rgb_pred [B, 3] 对应）
        loss_color = F.mse_loss(rgb_pred, target_rgb)
        # 前 1000 步 Eikonal 热身（较小权重），之后全量
        eik_w = args.eikonal_weight * (0.1 if step < 1000 else 1.0)
        loss = loss_color + eik_w * loss_eikonal

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_eikonal = loss_eikonal.item()
        psnr_val = -10. * torch.log10(loss_color.detach())
        s_val_now = s_curr.item() if isinstance(s_curr, torch.Tensor) else s_curr

        diag = extra_info.get("eik_diag", {})
        pbar.set_postfix({
            "Lc": f"{loss_color.item():.4f}",
            "Le": f"{loss_eikonal.item():.3f}",
            "in%": f"{diag.get('inside_frac', 0)*100:.0f}" if diag else "-",
            "ḡ": f"{diag.get('grad_mean', 0):.2f}" if diag else "-",
            "g↑": f"{diag.get('grad_max', 0):.1f}" if diag else "-",
            "LR": f"{lr:.2e}",
            "PSNR": f"{psnr_val.item():.2f}",
            "s": f"{s_val_now:.2f}",
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
    parser.add_argument("--batch_size", type=int, default=512, help="每步随机采样的射线数")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warm_up_end", type=int, default=5000, help="前n步从0线性增加到args.lr")
    parser.add_argument("--lr_alpha", type=float, default=0.05, help="Cosine annealing最终衰减比例（lr * lr_alpha）")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=2.2)
    parser.add_argument("--init_radius", type=float, default=0.5, help="初始球体 SDF 半径")
    parser.add_argument("--alpha_mode", type=str, default="volsdf",
                        choices=["volsdf", "neus"],
                        help="SDF→alpha 方式: volsdf=密度式(收敛稳), neus=CDF式(表面锐利)")
    parser.add_argument("--s_val_init", type=float, default=10.0,
                        help="s 参数初始值（volsdf 推荐 10，neus 推荐 3~5）")
    parser.add_argument("--eikonal_weight", type=float, default=0.1, help="Eikonal Loss 权重")
    parser.add_argument("--display_int", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
