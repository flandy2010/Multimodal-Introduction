import torch
import torch.nn as nn


class GaussianModel(nn.Module):
    def __init__(self, num_points=5000):
        super().__init__()
        self.num_points = num_points

        # 1. 中心位置 (XYZ) - 随机初始化在 [-1.5, 1.5] 范围内
        self.means = nn.Parameter(torch.rand(num_points, 3) * 3.0 - 1.5)

        # 2. 缩放 (Scaling) - 初始设为较小的值，经过 exp 确保为正
        self.scales = nn.Parameter(torch.log(torch.ones(num_points, 3) * 0.05))

        # 3. 旋转 (Rotation) - 使用四元数 (w, x, y, z)
        self.rotations = nn.Parameter(torch.zeros(num_points, 4))
        self.rotations.data[:, 0] = 1.0  # 初始无旋转

        # 4. 不透明度 (Opacity) - 经过 sigmoid 映射到 [0, 1]
        self.opacity = nn.Parameter(torch.zeros(num_points, 1))

        # 5. 颜色 (RGB) - 简单起见不使用球谐函数，直接存 RGB
        self.colors = nn.Parameter(torch.rand(num_points, 3))

    def get_scaling(self):
        return torch.exp(self.scales)

    def get_rotation(self):
        return torch.nn.functional.normalize(self.rotations)

    def get_opacity(self):
        return torch.sigmoid(self.opacity)

    def get_color(self):
        return torch.sigmoid(self.colors)

    def forward(self):
        # 3DGS 模型本身只是返回其参数
        return {
            "means": self.means,
            "scales": self.get_scaling(),
            "rotations": self.get_rotation(),
            "opacity": self.get_opacity(),
            "colors": self.get_color()
        }