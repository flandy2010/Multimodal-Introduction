import os
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from dataloader import GSDataLoader
from model import GaussianModel
from logger import GSLogger
from renderer import simple_rasterizer
from strategy import GaussianStrategy


def train(args):
    # 1. 环境准备
    if args.device != "auto":
        device = torch.device(args.device)
    else:
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 3DGS 真实场景训练启动 | 设备: {device}")

    # 2. 数据准备与场景解析
    # factor=16 时分辨率约 314x207，渲染速度合理；factor=8 精度更高但慢 4 倍
    loader = GSDataLoader(args.data_path, factor=args.factor)

    # 获取 Parser 算出的场景参数
    norm_params = loader.get_normalization_params()
    scene_radius = norm_params["radius"]

    # 获取初始点云 (关键！这是 3DGS 的灵魂)
    initial_pcd = loader.get_initial_pcd()

    # 3. 日志初始化
    logger = GSLogger(args.exp_dir, args)
    logger.log_dataset_stats(loader)

    # 4. 模型初始化
    model = GaussianModel(
        num_points=args.num_points,
        radius=scene_radius,
        sh_degree=args.sh_degree,
        pcd=initial_pcd
    ).to(device)

    # 5. 策略初始化
    strategy = GaussianStrategy(grad_threshold=args.grad_threshold)

    # 6. 初始优化器 (使用 model 内部定义的分组学习率)
    optimizer = torch.optim.Adam(model.get_optimizer_groups(args.lr), eps=1e-15)

    moving_avg_loss = None
    ema_alpha = 0.9

    pbar = tqdm(range(args.n_iters), desc="3DGS Training")
    for step in pbar:
        # --- A. 学习率调度 (Cosine 衰减，温和下降) ---
        progress = step / args.n_iters
        decay_factor = 0.5 * (1 + math.cos(math.pi * progress))  # 从1.0平滑下降到0
        decay_factor = max(decay_factor, 0.01)  # 最低保留 1%
        curr_lr = args.lr * decay_factor
        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = param_group["lr"]
            param_group['lr'] = param_group["initial_lr"] * decay_factor

        # --- B. 渲染 ---
        idx = np.random.randint(len(loader.images))
        gt_image, w2c, K, camera_pos = loader.get_view_params(idx)
        gt_image, w2c, K, camera_pos = gt_image.to(device), w2c.to(device), K.to(device), camera_pos.to(device)

        gaussians = model(camera_pos=camera_pos)
        out_image = simple_rasterizer(gaussians, w2c, K, loader.H, loader.W, tile_size=args.tile_size)

        # --- C. 损失与反向传播 ---
        loss = strategy.get_loss(out_image, gt_image, model, step)

        if loss.grad_fn is None:
            continue
        optimizer.zero_grad()
        loss.backward()

        # --- D. 密度策略介入 (Cloning/Pruning) ---
        # 注意：这里可能会重置 optimizer，所以必须接收返回值
        optimizer = strategy.step(step, model, optimizer)

        # --- E. 参数更新与硬约束 ---
        optimizer.step()

        # 执行 model 内部的 clamp 约束（防止点飞走或变巨大）
        model.apply_constraints()

        # --- F. 日志记录 ---
        loss_val = loss.item()
        moving_avg_loss = loss_val if moving_avg_loss is None else ema_alpha * moving_avg_loss + (
                    1 - ema_alpha) * loss_val

        pbar.set_postfix({
            "Points": model.num_points,
            "AvgL": f"{moving_avg_loss:.4f}",
            "LR": f"{curr_lr:.1e}"
        })

        if step % args.display_int == 0 or step == args.n_iters - 1:
            psnr, ssim = logger.log_combined(model, step, out_image, gt_image, curr_lr)
            tqdm.write(f"🔔 [Step {step}] PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | Points: {model.num_points}")

    # 7. 保存结果
    save_path = os.path.join(args.exp_dir, "gs_final.pth")
    model.save_model(save_path)
    print(f"✅ 训练完成！最终点数: {model.num_points}, 模型保存在: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Modular 3DGS Trainer")
    # 推荐路径
    parser.add_argument("--data_path", type=str, default="../data/360_extra_scenes/flowers")
    parser.add_argument("--exp_dir", type=str, default="./gs_runs/final_dance")

    parser.add_argument("--factor", type=int, default=16, help="图像下采样倍率，16=快速迭代，4=高精度")
    parser.add_argument("--num_points", type=int, default=15000, help="初始点数上限")
    parser.add_argument("--n_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--sh_degree", type=int, default=3, help="SH 球谐阶数 (0=DC only, 3=full)")
    parser.add_argument("--tile_size", type=int, default=64, help="渲染 tile 大小（越小越省显存）")
    parser.add_argument("--grad_threshold", type=float, default=0.0004, help="密度控制梯度阈值（越高越保守）")
    parser.add_argument("--display_int", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()