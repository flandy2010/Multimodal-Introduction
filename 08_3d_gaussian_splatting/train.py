import os
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from dataloader import GSDataLoader
from model import GaussianModel
from logger import GSLogger
from renderer import simple_rasterizer, gsplat_rasterizer
from strategy import GaussianStrategy


def train(args):

    # 1. 环境准备
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"3DGS Training | device: {device}")

    # 2. 数据准备
    loader = GSDataLoader(args.data_path, factor=args.factor)
    norm_params = loader.get_normalization_params()
    scene_radius = norm_params["radius"]
    initial_pcd = loader.get_initial_pcd()

    # 3. 日志
    logger = GSLogger(args.exp_dir, args)
    logger.log_dataset_stats(loader)

    # 4. 模型
    model = GaussianModel(
        fx=loader.focal,
        fy=loader.focal,
        num_points=args.num_points,
        radius=scene_radius,
        sh_degree=args.sh_degree,
        pcd=initial_pcd
    ).to(device)

    # 5. 策略（论文标准参数）
    strategy = GaussianStrategy(
        densify_from_iter=500,
        densify_until_iter=args.n_iters // 2,  # 在总步数的一半处停止增殖
        densification_interval=100,
        opacity_reset_interval=3000,
        grad_threshold=args.grad_threshold,
        max_points=args.max_points,
    )

    # 6. 优化器（论文推荐的绝对学习率，不依赖外部 lr）
    optimizer = torch.optim.Adam(model.get_optimizer_groups(), eps=1e-15)

    # Means LR 指数衰减：从 0.00016 → 0.0000016 (衰减 100 倍)
    means_lr_init = 0.00016
    means_lr_final = 0.0000016

    moving_avg_loss = None
    ema_alpha = 0.9

    pbar = tqdm(range(args.n_iters), desc="3DGS Training")
    for step in pbar:
        # --- A. Means LR 指数衰减（只衰减 means，其他保持恒定）---
        lr_ratio = (means_lr_final / means_lr_init) ** (step / args.n_iters)
        means_lr = means_lr_init * lr_ratio
        for param_group in optimizer.param_groups:
            if param_group['name'] == 'means':
                param_group['lr'] = means_lr

        # --- B. 渲染 ---
        idx = np.random.randint(len(loader.images))
        gt_image, c2w, w2c, K, camera_pos = loader.get_view_params(idx)
        gt_image, c2w, w2c, K, camera_pos = gt_image.to(device), c2w.to(device), w2c.to(device), K.to(device), camera_pos.to(device)

        gaussians = model(camera_pos=camera_pos)
        # out_image = simple_rasterizer(gaussians, w2c, K, loader.H, loader.W, tile_size=args.tile_size)
        out_image = gsplat_rasterizer(gaussians, w2c, K, loader.H, loader.W, tile_size=args.tile_size)

        # --- C. 损失 ---
        viewspace_points = gaussians.get("viewspace_points", None)
        loss = strategy.get_loss(out_image, gt_image, model, step)

        if loss.grad_fn is None:
            continue
        optimizer.zero_grad()
        loss.backward()

        # --- D. 密度策略 ---
        # optimizer = strategy.step(step, model, optimizer)
        optimizer = strategy.step(step, model, optimizer, c2w=c2w, viewspace_points=viewspace_points)

        # --- E. 更新 + 约束 ---
        optimizer.step()
        model.apply_constraints()

        # --- F. 日志 ---
        loss_val = loss.item()
        moving_avg_loss = loss_val if moving_avg_loss is None else ema_alpha * moving_avg_loss + (
                    1 - ema_alpha) * loss_val

        pbar.set_postfix({
            "Pts": model.num_points,
            "Loss": f"{moving_avg_loss:.4f}",
            "mLR": f"{means_lr:.2e}"
        })

        if step % args.display_int == 0 or step == args.n_iters - 1:
            psnr, ssim = logger.log_combined(model, step, out_image, gt_image, means_lr)
            tqdm.write(f"[Step {step}] PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | Points: {model.num_points}")

    # 7. 保存
    save_path = os.path.join(args.exp_dir, "gs_final.pth")
    model.save_model(save_path)
    print(f"Done! Points: {model.num_points}, saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="3DGS Trainer")
    parser.add_argument("--data_path", type=str, default="../data/360_extra_scenes/flowers")
    parser.add_argument("--exp_dir", type=str, default="./runs/gs_default")

    parser.add_argument("--factor", type=int, default=8, help="Image downscale factor")
    parser.add_argument("--num_points", type=int, default=50000, help="Max initial points")
    parser.add_argument("--max_points", type=int, default=1000000, help="Max total points")
    parser.add_argument("--n_iters", type=int, default=30000)
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree (0=DC, 3=full)")
    parser.add_argument("--tile_size", type=int, default=128, help="Render tile size")
    parser.add_argument("--grad_threshold", type=float, default=0.0002, help="Densify grad threshold")
    parser.add_argument("--display_int", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
