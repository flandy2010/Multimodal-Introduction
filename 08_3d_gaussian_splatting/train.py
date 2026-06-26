import os
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from dataloader import GSDataLoader
from model import GaussianModel
from logger import GSLogger
from renderer import simple_rasterizer, gsplat_rasterizer, auto_rasterizer
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
    loader = GSDataLoader(args.data_path, factor=args.factor, max_init_points=args.num_points)
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
        # --- A. Means LR 指数衰减 + SH 渐进激活 ---
        lr_ratio = (means_lr_final / means_lr_init) ** (step / args.n_iters)
        means_lr = means_lr_init * lr_ratio
        for param_group in optimizer.param_groups:
            if param_group['name'] == 'means':
                param_group['lr'] = means_lr

        # 每 1000 步开放一阶 SH（论文标准：0→1→2→3 阶逐步激活）
        if step > 0 and step % 1000 == 0:
            model.oneupSHdegree()

        # --- B. 渲染 ---
        idx = np.random.randint(len(loader.images))
        gt_image, c2w, w2c, K, camera_pos = loader.get_view_params(idx)
        gt_image, c2w, w2c, K, camera_pos = gt_image.to(device), c2w.to(device), w2c.to(device), K.to(device), camera_pos.to(device)

        gaussians = model(camera_pos=camera_pos)

        # 提前 zero_grad，simple_rasterizer 训练模式会在内部逐 tile backward 累积梯度
        optimizer.zero_grad()

        raster_result = auto_rasterizer(gaussians, w2c, K, loader.H, loader.W,
                                        tile_size=args.tile_size,
                                        gt_image=gt_image,
                                        loss_fn=None)  # simple路径内部直接用 L1-sum，loss_fn 不再需要

        # --- C. 损失 & Backward ---
        if isinstance(raster_result, tuple):
            # simple_rasterizer 训练模式：内部已逐 tile backward 完毕
            # raster_result = (out_image_detach, total_l1_loss)
            out_image, loss_l1 = raster_result
            # SSIM 在 detach 图上计算（仅用于监控，不参与梯度）
            with torch.no_grad():
                loss_ssim = strategy.get_loss(out_image, gt_image, model, step)[1]
            loss = loss_l1  # 梯度已通过 tile backward 累积，这里仅作日志用
            if loss.grad_fn is not None:
                pass  # 不再 backward
        else:
            # gsplat / mps 路径：正常 backward
            out_image = raster_result
            loss_l1, loss_ssim = strategy.get_loss(out_image, gt_image, model, step)
            loss = 0.8 * loss_l1 + 0.1 * loss_ssim
            if loss.grad_fn is None:
                gaussians.clear()
                continue
            loss.backward()

        viewspace_points = gaussians.get("viewspace_points", None)

        # --- D. 密度策略 ---
        optimizer = strategy.step(step, model, optimizer, c2w=c2w,
                                  viewspace_points=viewspace_points,
                                  image_hw=(loader.H, loader.W))

        # 主动清理 gaussians 字典
        gaussians.clear()

        # --- E. 更新 + 约束 ---
        optimizer.step()
        model.apply_constraints()

        # --- F. 日志 ---
        stats = model.get_diagnostics()

        pbar.set_postfix({
            "L1":    f"{loss_l1.item():.4f}",
            "Pts":   model.num_points,
            "SH":    f"{model.active_sh_degree}/{model.sh_degree}",
            # op：top10% / 全量 不透明度均值
            "op":    f"{stats['op_top10_mean']:.2f}/{stats['op_mean']:.2f}",
            # ab_op：不透明度过低（< 0.005）的椭球比例
            "ab_op": f"{stats['ab_op']:.2f}",
            # r：top10% / 全量 平均半径均值
            "r":     f"{stats['r_top10_mean']:.4f}/{stats['r_mean']:.4f}",
            # ab_r：半径超过阈值（radius*0.01）的椭球比例
            "ab_r":  f"{stats['ab_r']:.2f}",
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
    parser.add_argument("--num_points", type=int, default=15000, help="Max initial points")
    parser.add_argument("--max_points", type=int, default=1000000, help="Max total points")
    parser.add_argument("--n_iters", type=int, default=30000)
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree (0=DC, 3=full)")
    parser.add_argument("--tile_size", type=int, default=128, help="Render tile size")
    parser.add_argument("--grad_threshold", type=float, default=0.0002, help="Densify grad threshold")
    parser.add_argument("--display_int", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
