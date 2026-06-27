import os
import json
import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func
import datetime


class SDFLogger:
    """与 06 NeRF 的 NeRFLogger 统一风格，额外支持 SDF 切面可视化"""

    def __init__(self, exp_dir, args):
        self.exp_dir = exp_dir
        self.vis_dir = os.path.join(exp_dir, "visuals")
        os.makedirs(self.vis_dir, exist_ok=True)

        # 保存超参数
        config = vars(args)
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # markdown 日志
        self.log_file = os.path.join(exp_dir, "train_log.md")
        with open(self.log_file, "w") as f:
            f.write(f"# SDF Training Report\n")
            f.write(f"时间: {datetime.datetime.now()}\n\n")
            f.write("## 超参数\n")
            f.write("| Param | Value |\n| :--- | :--- |\n")
            for k, v in config.items():
                f.write(f"| {k} | {v} |\n")
            f.write("\n")

        self._table_started = False

    def calculate_image_metrics(self, pred, gt):
        pred_np = pred.detach().cpu().numpy().clip(0, 1)
        gt_np = gt.detach().cpu().numpy().clip(0, 1)

        mse = np.mean((pred_np - gt_np) ** 2)
        psnr = -10. * np.log10(mse + 1e-10)

        min_dim = min(pred_np.shape[0], pred_np.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        ssim_val = ssim_func(gt_np, pred_np, channel_axis=-1, data_range=1.0, win_size=win_size)

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

    def log_evaluation(self, step, metrics, lr, loss_eikonal=0.0, s_val=None):
        with open(self.log_file, "a") as f:
            if not self._table_started:
                f.write("## 训练进度\n\n")
                f.write("| Step | PSNR | SSIM | MaxErr | MeanErr | Eikonal | s | LR |\n")
                f.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n")
                self._table_started = True
            s_str = f"{s_val:.2f}" if s_val is not None else "-"
            f.write(f"| {step} | {metrics['psnr']:.2f} | {metrics['ssim']:.4f} | "
                    f"{metrics['max_error']:.4f} | {metrics['mean_error']:.4f} | "
                    f"{loss_eikonal:.4f} | {s_str} | {lr:.2e} |\n")

    def print_analysis(self, step, metrics, lr):
        psnr = metrics['psnr']
        ssim = metrics['ssim']
        print(f"\n--- Step {step} ---")
        print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | MaxErr: {metrics['max_error']:.4f}")

        if psnr < 15:
            print("  [!] 严重未收敛。检查 SDF 初始化半径、s 参数、near/far 范围。")
        elif ssim < 0.6:
            print("  [!] 几何结构模糊。建议增加 n_samples 或降低 Eikonal 权重。")
        elif psnr > 25:
            print("  [ok] 收敛良好。")
        else:
            print("  [..] 正常收敛中。")
        print("-" * 40)

    def save_visualization(self, step, pred, gt, metrics):
        """三联图：GT | Rendered | Error Heatmap"""
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

        save_path = os.path.join(self.vis_dir, f"render_{step:05d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)

    @torch.no_grad()
    def visualize_sdf_slice(self, step, sdf_net, device):
        """绘制三个坐标轴平面的 SDF 切面图（XY, XZ, YZ）"""
        res = 128
        grid = torch.linspace(-1.5, 1.5, res)
        x, y = torch.meshgrid(grid, grid, indexing='ij')

        # ---- 1. XY平面 (Z=0) ----
        pts_xy = torch.stack([x, y, torch.zeros_like(x)], dim=-1).to(device).reshape(-1, 3)
        sdf_xy = sdf_net.sdf(pts_xy)
        sdf_xy = sdf_xy.reshape(res, res).cpu().numpy()

        # ---- 2. XZ平面 (Y=0) ----
        pts_xz = torch.stack([x, torch.zeros_like(x), y], dim=-1).to(device).reshape(-1, 3)
        sdf_xz = sdf_net.sdf(pts_xz)
        sdf_xz = sdf_xz.reshape(res, res).cpu().numpy()

        # ---- 3. YZ平面 (X=0) ----
        pts_yz = torch.stack([torch.zeros_like(x), x, y], dim=-1).to(device).reshape(-1, 3)
        sdf_yz = sdf_net.sdf(pts_yz)
        sdf_yz = sdf_yz.reshape(res, res).cpu().numpy()

        # ---- 绘图 ----
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        titles = ['XY (Z=0)', 'XZ (Y=0)', 'YZ (X=0)']
        data = [sdf_xy, sdf_xz, sdf_yz]

        for ax, dat, title in zip(axes, data, titles):
            im = ax.imshow(dat, cmap='seismic', origin='lower',
                           extent=[-1.5, 1.5, -1.5, 1.5])
            # 绘制零等值线（黑色实线）
            ax.contour(np.linspace(-1.5, 1.5, res), np.linspace(-1.5, 1.5, res),
                       dat, levels=[0], colors='black', linewidths=2)
            ax.set_title(f"{title}  Step {step}")
            ax.set_xlabel('X' if 'X' in title else 'Y' if 'Y' in title else 'Z')
            ax.set_ylabel('Y' if 'Y' in title else 'Z' if 'Z' in title else 'X')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, f"sdf_slices_{step:05d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)

    def evaluate_and_log(self, step, pred, gt, lr, loss_eikonal=0.0, s_val=None, sdf_net=None, device=None):
        """一站式：指标 + 日志 + 可视化 + SDF 切面"""
        metrics = self.calculate_image_metrics(pred, gt)
        self.log_evaluation(step, metrics, lr, loss_eikonal, s_val=s_val)
        self.print_analysis(step, metrics, lr)
        self.save_visualization(step, pred, gt, metrics)

        if sdf_net is not None and device is not None:
            self.visualize_sdf_slice(step, sdf_net, device)

        return metrics['psnr']
