import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    """
    SDF 网络：输入 3D 位置，输出 SDF 值 + 几何特征
    架构参考 06 NeRF 的 pts_linears（8 层 + Skip Connection）
    区别：激活函数用 Softplus（保证 SDF 平滑）；几何初始化为球体
    """
    def __init__(self, D=8, W=256, L_pos=6, init_radius=4.0):
        super().__init__()
        self.L_pos = L_pos
        self.pos_enc = PositionalEncoding(L_pos)

        in_dim = 3 + 3 * 2 * L_pos

        # 几何主干（与 NeRF 的 pts_linears 结构对齐）
        self.pts_linears = nn.ModuleList([nn.Linear(in_dim, W)])
        for i in range(D - 1):
            if i == 4:  # 第 5 层 Skip Connection
                self.pts_linears.append(nn.Linear(W + in_dim, W))
            else:
                self.pts_linears.append(nn.Linear(W, W))

        # SDF 输出头（1 维 SDF 值）
        self.sdf_linear = nn.Linear(W, 1)

        # 几何特征输出（供颜色网络使用，与 NeRF 的 feature_linear 对应）
        self.feature_linear = nn.Linear(W, W)

        # --- 几何初始化：让初始输出近似球体 SDF ---
        self._geometric_init(init_radius)

    def _geometric_init(self, radius):
        """
        几何初始化为近似球体 SDF（IDR/NeuS 标准做法）

        策略：让 sdf_linear 的权重非常小 → 初始输出 ≈ bias = -radius（常数）
        然后由 Eikonal Loss 逐步把网络推向真正的 ||x|| - radius 形态。
        这比试图通过特殊初始化直接得到球体更稳定。
        """
        with torch.no_grad():
            for i, layer in enumerate(self.pts_linears):
                nn.init.constant_(layer.bias, 0.0)
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

            # SDF 输出层：极小权重 + bias=-radius
            # 初始 SDF ≈ -radius 对所有点（全在内部）
            # Eikonal Loss 会驱动其快速学习空间结构
            nn.init.normal_(self.sdf_linear.weight, 0.0, 0.0001)
            nn.init.constant_(self.sdf_linear.bias, -radius)

            # 特征层
            nn.init.constant_(self.feature_linear.bias, 0.0)
            nn.init.kaiming_normal_(self.feature_linear.weight, nonlinearity='relu')

    def forward(self, x):
        """
        输入: x [N, 3]
        输出: sdf [N, 1], features [N, W]
        """
        x_embed = self.pos_enc(x)

        h = x_embed
        for i, l in enumerate(self.pts_linears):
            if i == 5:  # Skip Connection（对应 D-1 中 i==4 的那层之后）
                h = torch.cat([x_embed, h], dim=-1)
            h = F.softplus(l(h), beta=100)

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


if __name__ == '__main__':
    sdf_net = SDFNetwork()
    color_net = ColorNetwork()

    pts = torch.randn(100, 3)
    dirs = torch.randn(100, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    sdf, feats = sdf_net(pts)
    rgb = color_net(pts, dirs, feats)
    print(f"SDF shape: {sdf.shape}, range: [{sdf.min():.3f}, {sdf.max():.3f}]")
    print(f"RGB shape: {rgb.shape}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")
