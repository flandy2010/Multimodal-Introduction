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

    def step(self, step, model, optimizer, c2w, viewspace_points=None, viewspace_gids=None, image_hw=None):
        """每步调用：梯度累积 + 密度控制（clone/split/prune）+ 透明度重置"""

        # 梯度累积（仅 densify 区间需要）
        if step < self.densify_until_iter and viewspace_points is not None:
            model.update_densification_stats(viewspace_points, image_hw=image_hw, gids=viewspace_gids)

        # clone/split/prune：仅在 densify 区间内（原论文做法）
        if (self.densify_from_iter <= step < self.densify_until_iter
            and step % self.densification_interval == 0
            and model.num_points < self.max_points
        ):
            max_scale = model.radius * 0.1
            # 原论文：step > opacity_reset_interval(3000) 后才启用屏幕空间剪枝（20像素）
            max_screen_size = 20 if step > self.opacity_reset_interval else None
            optimizer = model.densify_and_prune(
                optimizer,
                grad_threshold=self.grad_threshold,
                min_opacity=0.005,
                max_scale=max_scale,
                c2w=c2w,
                max_screen_size=max_screen_size,
            )
            print(f"📊 [Step {step}] Densify: {model.num_points} points")

        # 透明度重置（全程有效，原论文做法）
        if step > 0 and step % self.opacity_reset_interval == 0:
            model.reset_opacity()
            print(f"🧹 [Step {step}] Opacity reset")

        return optimizer

    def get_loss(self, out_image, gt_image, model, step, scale_reg=0.01, opacity_reg=0.01):
        """
        论文标准 Loss: 0.8 * L1 + 0.2 * (1 - SSIM)
        + 正则化项（gsplat MCMCStrategy 默认值，对超大球有抑制作用）：
          scale_reg:   惩罚过大的 scale（L1 正则），默认 0.01
          opacity_reg: 惩罚非 0/1 的 opacity（熵正则），默认 0.01
        设为 0 可完全关闭正则化。
        """
        loss_l1   = F.l1_loss(out_image, gt_image)
        loss_ssim = ssim_loss(out_image, gt_image)

        loss = 0.8 * loss_l1 + 0.2 * loss_ssim

        # scale 正则：L1 on exp(log_scales)，即物理尺度的均值，惩罚大球
        if scale_reg > 0.0:
            scales = model.get_scaling()                      # [N, 3] 物理尺度
            loss_scale = scales.mean()
            loss = loss + scale_reg * loss_scale

        # opacity 正则：熵正则，鼓励 opacity 趋向 0 或 1（避免半透明大球积累）
        if opacity_reg > 0.0:
            opacities = model.get_opacity().squeeze()         # [N]
            loss_opa = -(opacities * torch.log(opacities + 1e-8)
                         + (1 - opacities) * torch.log(1 - opacities + 1e-8)).mean()
            loss = loss + opacity_reg * loss_opa

        return loss_l1, loss_ssim, loss