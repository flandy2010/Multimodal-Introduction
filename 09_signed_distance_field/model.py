import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class PositionalEncoding(nn.Module):
    """与 06 NeRF 相同的位置编码"""

    def __init__(self, L=10):
        super().__init__()
        self.L = L

    def forward(self, x):
        out = [x]
        for i in range(self.L):
            for func in [torch.sin, torch.cos]:
                out.append(func(2 ** i * x))
        return torch.cat(out, dim=-1)  # 输出维度: 3 + 3 * 2 * L


class SDFNetwork(nn.Module):
    def __init__(self, D=8, W=256, L_pos=6, init_radius=0.5):
        super().__init__()
        # 注意：如果你做了单位球归一化，init_radius 设为 0.5 最合适
        self.L_pos = L_pos
        self.pos_enc = PositionalEncoding(L_pos)
        in_dim = 3 + 3 * 2 * L_pos

        def make_layer(in_f, out_f, is_sdf_head=False):
            layer = nn.Linear(in_f, out_f)
            if is_sdf_head:
                # --- 关键修改 1：最后一层权重均值设为 0，标准差极小 ---
                # 这样初始输出就完全由 bias 控制
                nn.init.normal_(layer.weight, mean=0.0, std=1e-5)
                nn.init.constant_(layer.bias, -init_radius)
            else:
                # --- 关键修改 2：隐藏层使用更小的初始化，降低正偏置累积 ---
                nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_f))
                # 初始 bias 稍微给一点点负值，抵消 Softplus 的正偏置
                nn.init.constant_(layer.bias, -0.01)

            return nn.utils.weight_norm(layer)

        # 几何主干
        layers = []
        layers.append(make_layer(in_dim, W))
        for i in range(D - 1):
            if i == 4:
                layers.append(make_layer(W + in_dim, W))
            else:
                layers.append(make_layer(W, W))
        self.pts_linears = nn.ModuleList(layers)

        # 确保只定义一次，且使用 weight_norm
        self.sdf_linear = make_layer(W, 1, is_sdf_head=True)
        self.feature_linear = nn.Linear(W, W)

    def forward(self, x):
        x_embed = self.pos_enc(x)
        h = x_embed
        for i, l in enumerate(self.pts_linears):
            if i == 5:
                h = torch.cat([x_embed, h], dim=-1)
            # beta = 20 是对的
            h = F.softplus(l(h), beta=20)

        sdf = self.sdf_linear(h)
        features = self.feature_linear(h)
        return sdf, features


class ColorNetwork(nn.Module):
    """
    颜色网络：输入 (3D 位置, 视角方向, 几何特征) → RGB
    与 06 NeRF 的外观部分对应，但额外输入几何特征
    """

    def __init__(self, d_feature=256, L_dir=4):
        super().__init__()
        self.dir_enc = PositionalEncoding(L_dir)
        in_dim_dir = 3 + 3 * 2 * L_dir

        self.layers = nn.Sequential(
            nn.Linear(3 + d_feature + in_dim_dir, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, pts, dirs, features):
        """
        pts: [N, 3] 位置
        dirs: [N, 3] 归一化视角方向
        features: [N, W] 几何特征
        """
        d_embed = self.dir_enc(dirs)
        return self.layers(torch.cat([pts, d_embed, features], dim=-1))


class LearnableVariance(nn.Module):
    """
    NeuS 的可学习 s 参数（论文中称为 variance / inv_s）
    s = exp(log_s)，用 exp 保证 s > 0
    初始值通常设为 ln(init_val)，如 init_val=3 → log_s=ln(3)≈1.1
    """

    def __init__(self, init_val=3.0):
        super().__init__()
        # 存储 log(s)，用 exp 取出 s，保证 s > 0
        self.log_s = nn.Parameter(torch.tensor(np.log(init_val), dtype=torch.float32))

    @property
    def s(self):
        return torch.exp(self.log_s)

    def forward(self):
        return self.s


if __name__ == '__main__':
    sdf_net = SDFNetwork()
    color_net = ColorNetwork()
    variance = LearnableVariance(init_val=3.0)

    pts = torch.randn(100, 3)
    dirs = torch.randn(100, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    sdf, feats = sdf_net(pts)
    rgb = color_net(pts, dirs, feats)
    s = variance()
    print(f"SDF shape: {sdf.shape}, s = {s.item():.4f}")
    print(f"SDF shape: {sdf.shape}, range: [{sdf.min():.3f}, {sdf.max():.3f}]")
    print(f"RGB shape: {rgb.shape}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")
