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

    def __init__(self, D=8, W=256, L_pos=10, init_radius=0.5):
        super().__init__()
        self.L_pos = L_pos
        self.init_radius = init_radius          # 保存下来，forward 要用
        self.pos_enc = PositionalEncoding(L_pos)
        in_dim = 3 + 3 * 2 * L_pos

        def make_layer(in_f, out_f, is_sdf_head=False):
            layer = nn.Linear(in_f, out_f)
            if is_sdf_head:
                # ---- 修改点：让网络初始输出接近 0，不喧宾夺主 ----
                # 权重用极小标准差，偏置设为 0，这样 self.sdf_linear(h) ≈ 0
                nn.init.normal_(layer.weight, mean=0.0, std=1e-5)
                nn.init.constant_(layer.bias, 0.0)
            else:
                nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_f))
                nn.init.constant_(layer.bias, -0.01)
            return nn.utils.weight_norm(layer)

        # 几何主干（不变）
        layers = []
        layers.append(make_layer(in_dim, W))
        for i in range(D - 1):
            if i == 4:
                layers.append(make_layer(W + in_dim, W))
            else:
                layers.append(make_layer(W, W))
        self.pts_linears = nn.ModuleList(layers)

        # 输出头：sdf 和 feature
        self.sdf_linear = nn.Linear(W, 1)   # 不使用 weight_norm
        nn.init.normal_(self.sdf_linear.weight, 0.0, std=1e-4)
        nn.init.constant_(self.sdf_linear.bias, 1e-4)

        self.feature_linear = nn.Linear(W, W)
        # feature 的初始化可以保持原样，或简单的 xavier
        nn.init.xavier_uniform_(self.feature_linear.weight)
        nn.init.constant_(self.feature_linear.bias, 0.0)

    def forward(self, x):

        x_embed = self.pos_enc(x)
        h = x_embed
        for i, l in enumerate(self.pts_linears):
            if i == 5:
                h = torch.cat([x_embed, h], dim=-1)
            h = F.softplus(l(h), beta=20)

        # 原始网络输出（初始≈0）
        sdf_raw = self.sdf_linear(h)

        # ---- 关键：加上球体距离先验 ----
        # ||x|| - init_radius，保持维度一致
        sdf = sdf_raw + torch.norm(x, dim=-1, keepdim=True) - self.init_radius

        features = self.feature_linear(h)
        return sdf, features


class ColorNetwork(nn.Module):
    """
    颜色网络：输入 (3D 位置, 法线, 视角方向, 几何特征) → RGB
    """
    def __init__(self, d_feature=256, L_dir=4):
        super().__init__()
        self.dir_enc = PositionalEncoding(L_dir)
        in_dim_dir = 3 + 3 * 2 * L_dir

        # 输入：位置(3) + 法线(3) + 方向编码(in_dim_dir) + 特征(d_feature)
        self.layers = nn.Sequential(
            nn.Linear(3 + 3 + in_dim_dir + d_feature, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, pts, dirs, features, normals):
        """
        pts: [N, 3] 位置
        dirs: [N, 3] 归一化视角方向
        features: [N, W] 几何特征
        normals: [N, 3] 表面法线（通常从 SDF 梯度归一化得到）
        """
        d_embed = self.dir_enc(dirs)
        # 为节省显存，法线可以 detach（如果不希望颜色损失影响几何）
        # 如果你想保留法线的梯度，去掉 .detach()
        normals = normals.detach()  # 可选，视需求决定
        return self.layers(torch.cat([pts, normals, d_embed, features], dim=-1))


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
