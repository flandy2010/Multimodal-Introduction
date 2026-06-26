import torch
import torch.nn.functional as F


def ssim_loss(pred, gt, window_size=11):
    """简化版 SSIM Loss（单尺度），输入 [H, W, 3] 格式"""
    pred_t = pred.permute(2, 0, 1).unsqueeze(0)
    gt_t = gt.permute(2, 0, 1).unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device) / (window_size ** 2)

    ssim_val = 0.0
    for c in range(3):
        p = pred_t[:, c:c+1]
        g = gt_t[:, c:c+1]

        mu_p = F.conv2d(p, kernel, padding=pad)
        mu_g = F.conv2d(g, kernel, padding=pad)

        mu_p_sq = mu_p ** 2
        mu_g_sq = mu_g ** 2
        mu_pg = mu_p * mu_g

        sigma_p_sq = F.conv2d(p * p, kernel, padding=pad) - mu_p_sq
        sigma_g_sq = F.conv2d(g * g, kernel, padding=pad) - mu_g_sq
        sigma_pg = F.conv2d(p * g, kernel, padding=pad) - mu_pg

        numerator = (2 * mu_pg + C1) * (2 * sigma_pg + C2)
        denominator = (mu_p_sq + mu_g_sq + C1) * (sigma_p_sq + sigma_g_sq + C2)

        ssim_val += (numerator / (denominator + 1e-8)).mean()

    return 1.0 - ssim_val / 3.0


class GaussianStrategy:
    """
    论文标准密度控制策略
    """
    def __init__(self,
                 densify_from_iter=500,
                 densify_until_iter=15000,
                 densification_interval=100,
                 opacity_reset_interval=3000,
                 grad_threshold=0.0002,
                 max_points=80000):
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.grad_threshold = grad_threshold
        self.max_points = max_points

    # def step(self, step, model, optimizer):
    #     """每步调用：密度控制 + 透明度重置"""
    #
    #     # 密度控制：每 100 步执行
    #     if (step >= self.densify_from_iter and
    #             step < self.densify_until_iter and
    #             step % self.densification_interval == 0 and
    #             model.num_points < self.max_points):
    #         max_scale = model.radius * 0.1  # 剔除场景级巨球
    #         optimizer = model.densify_and_prune(
    #             optimizer,
    #             grad_threshold=self.grad_threshold,
    #             min_opacity=0.005,
    #             max_scale=max_scale,
    #         )
    #         print(f"📊 [Step {step}] Densify: {model.num_points} points")
    #
    #     # 透明度重置：每 3000 步
    #     if step > 0 and step % self.opacity_reset_interval == 0:
    #         model.reset_opacity()
    #         print(f"🧹 [Step {step}] Opacity reset")
    #
    #     return optimizer

    def step(self, step, model, optimizer, c2w, viewspace_points=None, image_hw=None):
        """每步调用：增加梯度累积 + 密度控制 + 透明度重置"""

        # --- 新增：梯度累积 (SDF 训练不需要，但 3DGS 必须有) ---
        # 只有在致密化区间内，才需要累积 2D 梯度
        if step < self.densify_until_iter and viewspace_points is not None:
            model.update_densification_stats(viewspace_points, image_hw=image_hw)

        # 密度控制：每 100 步执行 (保持不变)
        if (self.densify_from_iter <= step < self.densify_until_iter
            and step % self.densification_interval == 0
            and model.num_points < self.max_points
        ):
            max_scale = model.radius * 0.1
            # 执行真正的分裂/克隆/剔除
            optimizer = model.densify_and_prune(
                optimizer,
                grad_threshold=self.grad_threshold,
                min_opacity=0.005,
                max_scale=max_scale,
                c2w=c2w,
            )
            print(f"📊 [Step {step}] Densify: {model.num_points} points")

        # 透明度重置 (保持不变)
        if step > 0 and step % self.opacity_reset_interval == 0:
            model.reset_opacity()
            print(f"🧹 [Step {step}] Opacity reset")

        return optimizer

    def get_loss(self, out_image, gt_image, model, step):
        """论文标准 Loss: 0.8 * L1 + 0.2 * (1 - SSIM)"""
        loss_l1 = F.l1_loss(out_image, gt_image)
        loss_ssim = ssim_loss(out_image, gt_image)
        return loss_l1, loss_ssim
