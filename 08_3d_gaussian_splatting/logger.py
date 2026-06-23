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

        # 记录初始状态用于对比
        self._initial_means = None
        self._initial_scales = None
        self._initial_colors = None
        self._table_started = False

    def _format_list(self, lst):
        return "[" + ", ".join([f"{x:.4f}" for x in lst]) + "]"

    def log_dataset_stats(self, loader):
        """记录数据集分布（写入汇总表头部）"""
        img_mean = loader.images.mean().item()
        img_std = loader.images.std().item()
        centers = loader.poses[:, :3, 3]
        dist_mean = torch.norm(centers, dim=1).mean().item()

        stats = f"## 数据集\n" \
                f"- 分辨率: {loader.W}x{loader.H} | 视角: {len(loader.images)} | " \
                f"像素均值: {img_mean:.4f} | 相机距离: {dist_mean:.4f}\n\n"

        with open(self.summary_path, "a") as f:
            f.write(stats)
        print(f"📊 数据统计已记录: 均值 {img_mean:.4f}, 距离 {dist_mean:.4f}")

    def _snapshot_initial(self, model):
        """在第一次调用时快照初始参数"""
        if self._initial_means is None:
            self._initial_means = model.means.detach().clone()
            self._initial_scales = torch.exp(model.scales.detach()).clone()
            # SH 模型：用 DC 分量计算初始颜色
            self._initial_colors = (SH_C0 * model.sh_coeffs[:, 0].detach() + 0.5).clamp(0, 1).clone()

    def log_model_params(self, model, step=0, tag="Training"):
        """监控高斯点的物理状态 → 写入 model_states.md"""
        self._snapshot_initial(model)

        means = model.means.detach()
        scales = torch.exp(model.scales.detach())
        opacity = torch.sigmoid(model.opacity.detach())
        colors = (SH_C0 * model.sh_coeffs[:, 0].detach() + 0.5).clamp(0, 1)

        # 椭球形状
        scale_ratio = scales.max(dim=-1)[0] / (scales.min(dim=-1)[0] + 1e-8)
        n_sphere = (scale_ratio < 1.5).sum().item()
        n_ellipsoid = ((scale_ratio >= 1.5) & (scale_ratio < 3.0)).sum().item()
        n_needle = (scale_ratio >= 3.0).sum().item()

        # 尺度分布
        scale_mean = scales.mean().item()
        scale_max = scales.max().item()
        scale_std = scales.std().item()
        n_large = (scales.mean(dim=-1) > 0.02).sum().item()
        n_tiny = (scales.mean(dim=-1) < 0.003).sum().item()

        # 透明度
        n_visible = (opacity.squeeze() > 0.1).sum().item()
        n_opaque = (opacity.squeeze() > 0.8).sum().item()
        n_transparent = (opacity.squeeze() < 0.05).sum().item()

        # 位置变化
        if self._initial_means is not None and len(means) == len(self._initial_means):
            drift = (means - self._initial_means.to(means.device)).norm(dim=-1)
            drift_mean, drift_max = drift.mean().item(), drift.max().item()
        else:
            drift_mean = drift_max = -1.0

        # 颜色变化
        if self._initial_colors is not None and len(colors) == len(self._initial_colors):
            color_change = (colors - self._initial_colors.to(colors.device)).abs().mean().item()
        else:
            color_change = -1.0

        # Scale 变化
        if self._initial_scales is not None and len(scales) == len(self._initial_scales):
            scale_change = (scales - self._initial_scales.to(scales.device)).abs().mean().item()
        else:
            scale_change = -1.0

        # 写入详情文件
        block = f"## Step {step} [{tag}]\n" \
                f"| 指标 | 值 |\n| :--- | :--- |\n" \
                f"| 点数 | {len(means)} |\n" \
                f"| 位置范围 | {self._format_list(means.min(0)[0].tolist())} ~ {self._format_list(means.max(0)[0].tolist())} |\n" \
                f"| Scale mean/std/max | {scale_mean:.5f} / {scale_std:.5f} / {scale_max:.5f} |\n" \
                f"| 大点(>0.02) / 小点(<0.003) | {n_large} ({n_large/len(means)*100:.1f}%) / {n_tiny} ({n_tiny/len(means)*100:.1f}%) |\n" \
                f"| 形状 球/椭球/针 | {n_sphere} / {n_ellipsoid} / {n_needle} |\n" \
                f"| Opacity mean | {opacity.mean().item():.4f} |\n" \
                f"| 可见/不透明/透明 | {n_visible} / {n_opaque} / {n_transparent} |\n" \
                f"| 颜色 RGB | {self._format_list(colors.mean(0).tolist())} |\n" \
                f"| 位置漂移 mean/max | {drift_mean:.5f} / {drift_max:.5f} |\n" \
                f"| 颜色变化 | {color_change:.5f} |\n" \
                f"| 缩放变化 | {scale_change:.5f} |\n\n"

        with open(self.detail_path, "a") as f:
            f.write(block)

        # 同时在汇总表追加一行紧凑摘要（不打断表格）
        summary_row = f"| {step} | {len(means)} | {scale_mean:.4f} | {scale_max:.4f} | " \
                      f"{n_large} | {opacity.mean().item():.3f} | " \
                      f"{drift_mean:.4f} | {scale_change:.5f} |\n"
        with open(self.summary_path, "a") as f:
            f.write(summary_row)

        # 告警
        if n_large > len(means) * 0.1:
            print(f"⚠️ [Step {step}] {n_large} 个大点 (>10%)，可能光晕")
        if scale_max > 0.025:
            print(f"⚠️ [Step {step}] Scale Max={scale_max:.4f}，光晕风险")

    def evaluate_and_analyze(self, step, pred, gt, current_lr):
        """渲染质量评测 → 写入 summary.md 的连续表格"""
        pred_np = pred.detach().cpu().numpy().clip(0, 1)
        gt_np = gt.detach().cpu().numpy()

        # 核心指标
        mse = np.mean((pred_np - gt_np) ** 2)
        psnr = -10. * np.log10(mse + 1e-10)
        ssim_val = ssim_func(gt_np, pred_np, channel_axis=-1, data_range=1.0)

        # 误差
        error_map = np.abs(pred_np - gt_np).mean(axis=-1)
        max_err = np.max(error_map)

        # 覆盖率
        coverage = (pred_np.sum(axis=-1) > 0.01).mean()

        # 写入汇总表（连续行，不被模型状态打断）
        with open(self.summary_path, "a") as f:
            if not self._table_started:
                f.write("## 训练进度\n\n")
                f.write("| Step | PSNR | SSIM | MaxErr | Coverage | LR | Points | ScaleAvg | ScaleMax | LargePts | OpacAvg | Drift | ScaleΔ |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                self._table_started = True
            # 这里先写渲染指标，模型状态指标由 log_model_params 补充
            # 为了保持一行完整，我们在这里把两者合并

        # 暂存本 step 的渲染指标，等 log_model_params 时一起写
        self._pending_eval = {
            "step": step,
            "psnr": psnr,
            "ssim": ssim_val,
            "max_err": max_err,
            "coverage": coverage,
            "lr": current_lr,
        }

        # 可视化
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

        return psnr, ssim_val

    def log_combined(self, model, step, pred, gt, current_lr, tag="Training"):
        """一站式调用：评测 + 模型状态，输出为汇总表的一行 + 详情的一个 block"""
        self._snapshot_initial(model)

        # --- 渲染指标 ---
        pred_np = pred.detach().cpu().numpy().clip(0, 1)
        gt_np = gt.detach().cpu().numpy()
        mse = np.mean((pred_np - gt_np) ** 2)
        psnr = -10. * np.log10(mse + 1e-10)
        ssim_val = ssim_func(gt_np, pred_np, channel_axis=-1, data_range=1.0)
        error_map = np.abs(pred_np - gt_np).mean(axis=-1)
        max_err = np.max(error_map)
        coverage = (pred_np.sum(axis=-1) > 0.01).mean()

        # --- 模型指标 ---
        means = model.means.detach()
        scales = torch.exp(model.scales.detach())
        opacity = torch.sigmoid(model.opacity.detach())
        colors = (SH_C0 * model.sh_coeffs[:, 0].detach() + 0.5).clamp(0, 1)

        scale_mean = scales.mean().item()
        scale_max = scales.max().item()
        scale_std = scales.std().item()
        n_large = (scales.mean(dim=-1) > 0.02).sum().item()
        n_tiny = (scales.mean(dim=-1) < 0.003).sum().item()

        scale_ratio = scales.max(dim=-1)[0] / (scales.min(dim=-1)[0] + 1e-8)
        n_sphere = (scale_ratio < 1.5).sum().item()
        n_ellipsoid = ((scale_ratio >= 1.5) & (scale_ratio < 3.0)).sum().item()
        n_needle = (scale_ratio >= 3.0).sum().item()

        n_visible = (opacity.squeeze() > 0.1).sum().item()
        n_opaque = (opacity.squeeze() > 0.8).sum().item()
        n_transparent = (opacity.squeeze() < 0.05).sum().item()

        if self._initial_means is not None and len(means) == len(self._initial_means):
            drift = (means - self._initial_means.to(means.device)).norm(dim=-1)
            drift_mean, drift_max = drift.mean().item(), drift.max().item()
        else:
            drift_mean = drift_max = -1.0

        if self._initial_colors is not None and len(colors) == len(self._initial_colors):
            color_change = (colors - self._initial_colors.to(colors.device)).abs().mean().item()
        else:
            color_change = -1.0

        if self._initial_scales is not None and len(scales) == len(self._initial_scales):
            scale_change = (scales - self._initial_scales.to(scales.device)).abs().mean().item()
        else:
            scale_change = -1.0

        # --- 写汇总表（一行包含全部信息，连续不打断）---
        with open(self.summary_path, "a") as f:
            if not self._table_started:
                f.write("## 训练进度\n\n")
                f.write("| Step | PSNR | SSIM | Coverage | Points | ScaleAvg | ScaleMax | Large | Opacity | Drift | ColorΔ | ScaleΔ | LR |\n")
                f.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n")
                self._table_started = True
            f.write(f"| {step} | {psnr:.2f} | {ssim_val:.4f} | {coverage:.3f} | {len(means)} "
                    f"| {scale_mean:.5f} | {scale_max:.5f} | {n_large} | {opacity.mean().item():.3f} "
                    f"| {drift_mean:.4f} | {color_change:.4f} | {scale_change:.5f} | {current_lr:.2e} |\n")

        # --- 写详情 block ---
        block = f"## Step {step}\n" \
                f"- PSNR: {psnr:.2f} | SSIM: {ssim_val:.4f} | Coverage: {coverage:.3f}\n" \
                f"- Scale: mean={scale_mean:.5f}, std={scale_std:.5f}, max={scale_max:.5f}\n" \
                f"- 大点(>0.02): {n_large} ({n_large/len(means)*100:.1f}%) | 小点(<0.003): {n_tiny} ({n_tiny/len(means)*100:.1f}%)\n" \
                f"- 形状: 球={n_sphere} 椭球={n_ellipsoid} 针={n_needle}\n" \
                f"- Opacity: mean={opacity.mean().item():.4f} | 可见={n_visible} 不透明={n_opaque} 透明={n_transparent}\n" \
                f"- 漂移: mean={drift_mean:.5f} max={drift_max:.5f}\n" \
                f"- 颜色变化: {color_change:.5f} | 缩放变化: {scale_change:.5f}\n\n"

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
        if n_large > len(means) * 0.1:
            print(f"⚠️ [Step {step}] {n_large} 个大点 (>10%)，光晕风险")
        if scale_max > 0.025:
            print(f"⚠️ [Step {step}] Scale Max={scale_max:.4f}，接近上限")

        return psnr, ssim_val
