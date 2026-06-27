import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func
import datetime
from model import SH_C0


class GSLogger:
    def __init__(self, exp_dir, args):
        self.exp_dir = exp_dir
        self.vis_dir = os.path.join(exp_dir, "visuals")
        os.makedirs(self.vis_dir, exist_ok=True)

        # 保存配置
        with open(os.path.join(exp_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        # 两个独立文件：汇总表 + 模型状态详情
        self.summary_path = os.path.join(exp_dir, "summary.md")
        self.detail_path = os.path.join(exp_dir, "model_states.md")

        # 初始化汇总表（连续表格，不被打断）
        with open(self.summary_path, "w") as f:
            f.write(f"# 3DGS 训练汇总\n")
            f.write(f"生成时间: {datetime.datetime.now()}\n\n")

        # 初始化模型状态详情
        with open(self.detail_path, "w") as f:
            f.write(f"# 3DGS 模型状态详情\n\n")

        self._table_started = False

    def log_dataset_stats(self, loader):
        """记录数据集分布（写入汇总表头部）"""
        sample_img, *_ = loader.get_view_params(0)
        img_mean = sample_img.mean().item()
        img_std = sample_img.std().item()
        centers = loader.poses[:, :3, 3]
        dist_mean = torch.norm(centers, dim=1).mean().item()

        stats = f"## 数据集\n" \
                f"- 分辨率: {loader.W}x{loader.H} | 视角: {len(loader.images)} | " \
                f"像素均值(采样): {img_mean:.4f} | 相机距离: {dist_mean:.4f}\n\n"

        with open(self.summary_path, "a") as f:
            f.write(stats)
        print(f"📊 数据统计已记录: 均值 {img_mean:.4f}, 距离 {dist_mean:.4f}")

    def log_combined(self, model, step, pred, gt, current_lr, tag="Training"):
        """一站式调用：评测 + 模型状态，输出为汇总表的一行 + 详情的一个 block"""
        # --- 渲染指标 ---
        pred_np = pred.detach().cpu().numpy().clip(0, 1)
        gt_np   = gt.detach().cpu().numpy()
        mse      = np.mean((pred_np - gt_np) ** 2)
        psnr     = -10. * np.log10(mse + 1e-10)
        ssim_val = ssim_func(gt_np, pred_np, channel_axis=-1, data_range=1.0)
        error_map = np.abs(pred_np - gt_np).mean(axis=-1)

        # --- 模型指标 ---
        means   = model.means.detach()
        scales  = torch.exp(model.scales.detach())
        opacity = torch.sigmoid(model.opacity.detach())

        scale_max     = scales.max().item()
        n_large       = (scales.mean(dim=-1) > 0.02).sum().item()
        n_tiny        = (scales.mean(dim=-1) < 0.003).sum().item()
        n_transparent = (opacity.squeeze() < 0.05).sum().item()

        # --- 写汇总表 ---
        with open(self.summary_path, "a") as f:
            if not self._table_started:
                f.write("## 训练进度\n\n")
                f.write("| Step | PSNR | SSIM | Points | ScaleMax | Large | Opacity | LR |\n")
                f.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n")
                self._table_started = True
            f.write(f"| {step} | {psnr:.2f} | {ssim_val:.4f} | {len(means)} "
                    f"| {scale_max:.5f} | {n_large} | {opacity.mean().item():.3f} "
                    f"| {current_lr:.2e} |\n")

        # --- 写详情 block ---
        block = f"## Step {step}\n" \
                f"- PSNR: {psnr:.2f} | SSIM: {ssim_val:.4f}\n" \
                f"- Points: {len(means)} | Scale max: {scale_max:.5f}\n" \
                f"- 大点(>0.02): {n_large} ({n_large/len(means)*100:.1f}%) | 小点(<0.003): {n_tiny} ({n_tiny/len(means)*100:.1f}%)\n" \
                f"- Opacity: mean={opacity.mean().item():.4f} | 透明球(<0.05): {n_transparent} ({n_transparent/len(means)*100:.1f}%)\n\n"

        with open(self.detail_path, "a") as f:
            f.write(block)

        # --- 可视化 ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(gt_np)
        axes[0].set_title("Ground Truth")
        axes[1].imshow(pred_np)
        axes[1].set_title(f"Rendered (PSNR: {psnr:.2f})")
        im = axes[2].imshow(error_map, cmap='jet')
        axes[2].set_title(f"Error (mean: {np.mean(error_map):.4f})")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        for ax in axes: ax.axis('off')
        save_path = os.path.join(self.vis_dir, f"step_{step:04d}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)

        # 告警
        if scale_max > model.radius * 0.1:
            print(f"⚠️ [Step {step}] Scale Max={scale_max:.4f}，接近上限")

        return psnr, ssim_val