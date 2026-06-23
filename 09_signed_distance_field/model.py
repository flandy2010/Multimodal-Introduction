import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SDFNetwork(nn.Module):
    def __init__(self, d_in=3, d_out=1, d_hidden=256, n_layers=8):
        super().__init__()
        # 增加位置编码 (与 NeRF 相同，用于抓细节)
        self.n_freqs = 6
        dims = [d_in + d_in * self.n_freqs * 2] + [d_hidden] * n_layers + [d_out + d_hidden]

        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

        # --- 核心：几何初始化 ---
        # 强制让 MLP 初始输出一个球体 (SDF = sqrt(x^2+y^2+z^2) - radius)
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if i == len(self.layers) - 1:
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[i]), std=0.0001)
                    nn.init.constant_(layer.bias, -0.5)  # 初始半径 0.5
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(dims[i + 1]))

    def forward(self, x):
        # 位置编码
        input_x = x
        for i in range(self.n_freqs):
            for func in [torch.sin, torch.cos]:
                input_x = torch.cat([input_x, func(2 ** i * np.pi * x)], dim=-1)

        h = input_x
        for i, l in enumerate(self.layers):
            h = l(h)
            if i < len(self.layers) - 1:
                h = F.softplus(h, beta=100)  # SDF 通常使用 softplus 保证平滑

        return h[..., :1], h[..., 1:]  # 返回 (SDF值, 几何特征)


class ColorNetwork(nn.Module):
    def __init__(self, d_feature=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3 + d_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, pts, features):
        return self.layers(torch.cat([pts, features], dim=-1))