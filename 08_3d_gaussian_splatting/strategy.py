import torch
import torch.nn.functional as F


class GaussianStrategy:
    def __init__(self,
                 densify_from_iter=300,
                 densify_until_iter=4000,
                 densification_interval=300,
                 grad_threshold=0.0002):
        """
        控制高斯密度控制的策略类
        """
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.grad_threshold = grad_threshold

    def step(self, step, model, optimizer):
        """
        在每个训练步调用，决定是否触发 Densification
        """
        if step > self.densify_from_iter and \
                step < self.densify_until_iter and \
                step % self.densification_interval == 0:
            # 调用 model.py 中定义的复杂 Tensor 拼接逻辑
            optimizer = model.densify_and_prune(optimizer, grad_threshold=self.grad_threshold)

        return optimizer

    def get_loss(self, out_image, gt_image, model, step):
        """
        封装复合 Loss 计算
        """
        loss_mse = F.mse_loss(out_image, gt_image)

        # 稀疏性惩罚：后期才开启，防止背景太脏
        sparsity_weight = 1e-7 if step > 1000 else 0.0
        loss_opacity = torch.mean(torch.sigmoid(model.gauss_params["opacities"]))

        total_loss = loss_mse + sparsity_weight * loss_opacity
        return total_loss