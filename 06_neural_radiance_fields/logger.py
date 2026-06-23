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

        # 保存超参数
        self.config = vars(args)
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

        # 初始化日志文件
        self.log_file = os.path.join(exp_dir, "train_log.md")
        with open(self.log_file, "w") as f:
            f.write(f"# NeRF Training Report\n")
            f.write(f"生成时间: {datetime.datetime.now()}\n\n")
            f.write("## 超参数\n")
            f.write("| Param | Value |\n| :--- | :--- |\n")
            for k, v in self.config.items():
                f.write(f"| {k} | {v} |\n")
            f.write("\n")

        self._table_started = False

    def calculate_image_metrics(self, pred, gt):
        """
        计算图像质量指标
        pred, gt: [H, W, 3] torch.Tensor (0-1 range)
        """
        pred_np = pred.detach().cpu().numpy().clip(0, 1)
        gt_np = gt.detach().cpu().numpy().clip(0, 1)

        # MSE & PSNR
        mse = np.mean((pred_np - gt_np) ** 2)
        psnr = -10. * np.log10(mse + 1e-10)

        # SSIM
        min_dim = min(pred_np.shape[0], pred_np.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        ssim_val = ssim_func(gt_np, pred_np, channel_axis=-1, data_range=1.0, win_size=win_size)

        # 误差分布
        abs_diff = np.abs(pred_np - gt_np)
        error_map = abs_diff.mean(axis=-1)

        return {
            "psnr": psnr,
            "ssim": ssim_val,
            "mse": mse,
            "max_error": np.max(abs_diff),
            "mean_error": np.mean(abs_diff),
            "error_map": error_map,
        }

    def log_evaluation(self, iter_idx, metrics, lr):
        """记录评估指标到连续表格"""
        with open(self.log_file, "a") as f:
            if not self._table_started:
                f.write("## 训练进度\n\n")
                f.write("| Iter | PSNR | SSIM | MaxErr | MeanErr | LR |\n")
                f.write("| ---: | ---: | ---: | ---: | ---: | :--- |\n")
                self._table_started = True
            f.write(f"| {iter_idx} | {metrics['psnr']:.2f} | {metrics['ssim']:.4f} | "
                    f"{metrics['max_error']:.4f} | {metrics['mean_error']:.4f} | {lr:.2e} |\n")

    def print_analysis(self, iter_idx, metrics, lr):
        """终端输出瓶颈分析"""
        psnr = metrics['psnr']
        ssim = metrics['ssim']
        max_err = metrics['max_error']

        print(f"\n--- Iteration {iter_idx} ---")
        print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | MaxErr: {max_err:.4f}")

        if psnr < 18:
            print("  [!] 严重未收敛。检查学习率、near/far 范围、采样数。")
        elif ssim < 0.7:
            print("  [!] 结构模糊。建议：增加 n_samples 或网络宽度。")
        elif max_err > 0.6:
            print("  [!] 存在噪点(Floaters)。建议：加大 Density Noise 或检查 near/far。")
        elif psnr > 25:
            print("  [ok] 收敛良好。")
        else:
            print("  [..] 正常收敛中。")
        print("-" * 40)

    def save_visualization(self, iter_idx, pred, gt, metrics):
        """
        保存三联对比图：GT | Rendered | Error Heatmap
        """
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
        """一站式调用：计算指标 + 写日志 + 打印分析 + 保存可视化"""
        metrics = self.calculate_image_metrics(pred, gt)
        self.log_evaluation(iter_idx, metrics, lr)
        self.print_analysis(iter_idx, metrics, lr)
        self.save_visualization(iter_idx, pred, gt, metrics)
        return metrics['psnr']
