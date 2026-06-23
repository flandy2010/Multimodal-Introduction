import torch
import torch.nn as nn
import numpy as np


def inverse_sigmoid(x):
    """Sigmoid 的反函数，用于初始化"""
    return torch.log(x / (1 - x + 1e-7) + 1e-7)


class GaussianModel(nn.Module):

    def __init__(self, num_points=2000, radius=1.5, pcd=None):
        super().__init__()
        if pcd is not None:
            means, colors_raw = pcd  # 直接拿来用
            colors = torch.log(colors_raw / (1 - colors_raw + 1e-7) + 1e-7)
        else:
            means = torch.rand(num_points, 3) * 2 - 1
            colors = torch.zeros(num_points, 3)

        self.num_points = means.shape[0]  # 使用实际点云数量，而非传入参数
        self.radius = radius

        self.gauss_params = nn.ParameterDict({
            "means": nn.Parameter(means),
            "scales": nn.Parameter(torch.log(torch.ones(means.shape[0], 3) * 0.005)),
            "rotations": nn.Parameter(torch.tile(torch.tensor([1.0, 0, 0, 0]), (means.shape[0], 1))),
            "opacities": nn.Parameter(torch.ones(means.shape[0], 1) * 0.0),
            "colors": nn.Parameter(colors)
        })

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def rotations(self):
        return self.gauss_params["rotations"]

    @property
    def opacity(self):
        return self.gauss_params["opacities"]

    @property
    def colors(self):
        return self.gauss_params["colors"]

    def get_scaling(self):
        return torch.exp(self.scales)

    def get_rotation(self):
        return torch.nn.functional.normalize(self.rotations)

    def get_opacity(self):
        return torch.sigmoid(self.opacity)

    def get_color(self):
        return torch.sigmoid(self.colors)

    def forward(self):
        return {
            "means": self.means,
            "scales": self.get_scaling(),
            "rotations": self.get_rotation(),
            "opacity": self.get_opacity(),
            "colors": self.get_color()
        }

    @torch.no_grad()
    def apply_constraints(self):
        # 严格限制缩放：scale 上限 0.03（约 8px 半径 @focal=540, depth=2）
        # 这是防止光晕的核心约束
        self.gauss_params["scales"].clamp_(max=np.log(0.03))
        # 限制位置，防止点飞走
        limit = self.radius * 2.0
        self.gauss_params["means"].clamp_(-limit, limit)

    def get_optimizer_groups(self, lr):
        """
        返回优化器参数组配置
        3DGS 论文推荐：means lr 最小（因为 SfM 点云初始位置已经很准），
        colors/opacity 稍大（需要从 0 开始学），scales 适中
        """
        return [
            {'params': [self.gauss_params["means"]], 'lr': lr * 0.1, 'name': 'means'},
            {'params': [self.gauss_params["colors"]], 'lr': lr * 2.0, 'name': 'colors'},
            {'params': [self.gauss_params["opacities"]], 'lr': lr * 2.0, 'name': 'opacities'},
            {'params': [self.gauss_params["scales"]], 'lr': lr * 1.0, 'name': 'scales'},
            {'params': [self.gauss_params["rotations"]], 'lr': lr * 0.5, 'name': 'rotations'},
        ]

    @torch.no_grad()
    def densify_and_prune(self, optimizer, grad_threshold=0.0004, min_opacity=0.01):
        """密度控制：克隆、分裂与剪枝"""
        if self.means.grad is None: return optimizer

        # 1. 提取梯度
        grads = torch.norm(self.means.grad, dim=-1)

        # 2. 掩码计算
        densify_mask = grads > grad_threshold
        opacities = self.get_opacity().squeeze()
        prune_mask = opacities < min_opacity

        # 3. 构造新参数
        new_params = {}
        for name, param in self.gauss_params.items():
            remain_param = param[~prune_mask]
            added_param = param[densify_mask]
            new_params[name] = nn.Parameter(torch.cat([remain_param, added_param], dim=0))

        # 4. 更新模型
        self.gauss_params = nn.ParameterDict(new_params)
        self.num_points = self.gauss_params["means"].shape[0]

        # 5. 重建优化器
        current_lr = optimizer.param_groups[0]['lr']
        new_optimizer = torch.optim.Adam(self.get_optimizer_groups(current_lr), eps=1e-15)

        return new_optimizer

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))