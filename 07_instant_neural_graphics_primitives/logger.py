import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func
import datetime


class NeRFLogger:
    def __init__(self, exp_dir, args):
        self.exp_dir = exp_dir
        self.vis_dir = os.path.join(exp_dir, "visuals")
        os.makedirs(self.vis_dir, exist_ok=True)

        # 记录超参数
        self.config = vars(args)
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

        # 初始化实验记录文件
        self.log_file = os.path.join(exp_dir, "train_log.md")
        with open(self.log_file, "w") as f:
            f.write(f"# NeRF Training Report - {datetime.datetime.now()}\n\n")
            f.write("## Hyperparameters\n")
            f.write(" | Param | Value |\n | :--- | :--- |\n")
            for k, v in self.config.items():
                f.write(f" | {k} | {v} |\n")
            f.write("\n## Evaluation Records\n")

    def calculate_image_metrics(self, pred, gt):
        """
        深度分析图片质量
        pred, gt: [H, W, 3] torch.Tensor (0-1 range)
        """
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        # 1. 基础 MSE 和 PSNR
        mse = np.mean((pred_np - gt_np) ** 2)
        psnr = -10. * np.log10(mse + 1e-10)

        # 2. SSIM (衡量结构是否完整，如果低，说明物体边缘太糊)
        # 使用较大的 win_size 处理小图
        ssim_val = ssim_func(gt_np, pred_np, channel_axis=-1, data_range=1.0)

        # 3. 误差分布统计
        abs_diff = np.abs(pred_np - gt_np)
        max_error = np.max(abs_diff)
        mean_error = np.mean(abs_diff)

        # 4. 分析图片缺陷类型
        # 如果 MSE 低但 SSIM 也低 -> 边缘不对，几何有问题
        # 如果 MSE 低但 Max Error 极高 -> 存在 Floaters (孤立噪点)
        return {
            "psnr": psnr,
            "ssim": ssim_val,
            "mse": mse,
            "max_error": max_error,
            "mean_error": mean_error,
            "error_map": np.abs(pred_np - gt_np).mean(axis=-1),
        }

    def log_evaluation(self, iter_idx, metrics, lr):
        """记录评估统计量到日志文件"""
        with open(self.log_file, "a") as f:
            if iter_idx == 0:
                f.write("\n| Iter | PSNR | SSIM | Max Err | Mean Err | LR |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")

            f.write(f"| {iter_idx} | {metrics['psnr']:.2f} | {metrics['ssim']:.4f} | "
                    f"{metrics['max_error']:.4f} | {metrics['mean_error']:.4f} | {lr:.2e} |\n")

    def print_analysis(self, iter_idx, metrics, lr):
        """在终端输出瓶颈分析"""
        print(f"\n--- 🧪 Iteration {iter_idx} Analysis ---")
        print(f"📈 PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}")
        print(f"🔍 Max Pixel Error: {metrics['max_error']:.4f}")

        # 瓶颈智能分析逻辑
        if metrics['psnr'] < 20:
            print("🚨 状态: 严重未收敛。原因可能: 学习率过低或坐标越界。")
        elif metrics['ssim'] < 0.8:
            print("⚠️ 瓶颈: 几何结构模糊。建议: 增加 Hash Grid 分辨率或加大 --n_samples。")
        elif metrics['max_error'] > 0.5:
            print("⚠️ 瓶颈: 存在孤立噪点(Floaters)。建议: 检查 Near/Far 范围或加入 TV Loss 正则化。")
        elif lr > 1e-3 and iter_idx > 2000:
            print("💡 建议: 学习率仍较高，建议加大衰减力度（减小 decay_rate）以磨平噪点。")
        else:
            print("✨ 状态: 正常收敛中。")
        print("-" * 40)

    def save_visualization(self, iter_idx, pred, gt, metrics):
        """保存三联对比图：GT | Rendered | Error Heatmap"""
        pred_np = pred.detach().cpu().numpy().clip(0, 1)
        gt_np = gt.detach().cpu().numpy().clip(0, 1)
        error_map = metrics["error_map"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(gt_np)
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')

        axes[1].imshow(pred_np)
        axes[1].set_title(f"Rendered (PSNR: {metrics['psnr']:.2f})")
        axes[1].axis('off')

        im = axes[2].imshow(error_map, cmap='jet', vmin=0, vmax=max(0.1, error_map.max()))
        axes[2].set_title(f"Error (mean: {metrics['mean_error']:.4f})")
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        save_path = os.path.join(self.vis_dir, f"iter_{iter_idx:05d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return save_path

    def evaluate_and_log(self, iter_idx, pred, gt, lr):
        """一站式：计算指标 + 写日志 + 打印分析 + 保存可视化"""
        metrics = self.calculate_image_metrics(pred, gt)
        self.log_evaluation(iter_idx, metrics, lr)
        self.print_analysis(iter_idx, metrics, lr)
        self.save_visualization(iter_idx, pred, gt, metrics)
        return metrics['psnr']