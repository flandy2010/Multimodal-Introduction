import torch
import torch.nn.functional as F


def ssim_loss(pred, gt, window_size=11):
    """简化版 SSIM Loss（单尺度），输入 [H, W, 3] 格式"""
    # 转为 [1, 3, H, W]
    pred_t = pred.permute(2, 0, 1).unsqueeze(0)
    gt_t = gt.permute(2, 0, 1).unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 均值滤波
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device) / (window_size ** 2)

    # 按通道分别计算
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
    def __init__(self,
                 densify_from_iter=500,
                 densify_until_iter=15000,
                 densification_interval=500,
                 opacity_reset_interval=3000,
                 grad_threshold=0.0004,
                 max_points=100000):
        """
        控制高斯密度和训练策略
        """
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.grad_threshold = grad_threshold
        self.max_points = max_points

    def step(self, step, model, optimizer):
        """
        在每个训练步调用，决定是否触发 Densification 或 Opacity Reset
        """
        # 密度控制
        if (step > self.densify_from_iter and
                step < self.densify_until_iter and
                step % self.densification_interval == 0 and
                model.num_points < self.max_points):
            optimizer = model.densify_and_prune(optimizer, grad_threshold=self.grad_threshold)

        # 透明度重置（论文核心 trick：清理膨胀的幽灵球）
        if step > 0 and step % self.opacity_reset_interval == 0:
            model.reset_opacity()
            print(f"🧹 [Step {step}] 透明度重置完成，清理光团")

        return optimizer

    def get_loss(self, out_image, gt_image, model, step):
        """
        复合 Loss: MSE + SSIM + 稀疏性正则
        SSIM 抗模糊，MSE 保证整体收敛，稀疏性正则防止点堆积
        """
        # L1 Loss（比 MSE 对异常值更鲁棒，不会过度惩罚小误差）
        loss_l1 = F.l1_loss(out_image, gt_image)

        # SSIM Loss（鼓励结构性匹配，抵抗"均匀色块"捷径）
        loss_ssim = ssim_loss(out_image, gt_image)

        # 混合比例：论文标准 0.8 L1 + 0.2 SSIM
        loss = 0.8 * loss_l1 + 0.2 * loss_ssim

        # 稀疏性惩罚：鼓励透明度向两极分化（要么全透明要么全不透明）
        if step > 500:
            opacity = torch.sigmoid(model.gauss_params["opacities"])
            # 熵正则：让 opacity 尽量接近 0 或 1
            entropy = -(opacity * torch.log(opacity + 1e-8) + (1 - opacity) * torch.log(1 - opacity + 1e-8))
            loss = loss + 1e-4 * entropy.mean()

        return loss
