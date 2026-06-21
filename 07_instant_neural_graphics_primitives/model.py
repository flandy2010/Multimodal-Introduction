import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HashEncoder(nn.Module):
    def __init__(self, num_levels=16, level_dim=2, per_level_scale=1.5,
                 base_resolution=16, log2_hashmap_size=19):
        super().__init__()
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.hashmap_size = 2 ** log2_hashmap_size
        self.resolutions = [int(base_resolution * (per_level_scale ** i)) for i in range(num_levels)]

        self.embeddings = nn.ModuleList([
            nn.Embedding(self.hashmap_size, level_dim) for _ in range(num_levels)
        ])

        # --- 修改 1: 稍微加大初始化范围，确保初期有梯度 ---
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, a=-0.01, b=0.01)  # 从 0.0001 提升到 0.01

        self.output_dim = num_levels * level_dim

    def hash_func(self, coords):
        # --- 修改 2: 显式指定质数为长整型，防止位运算异常 ---
        # 使用更稳健的质数处理
        primes = [1, 2654435761, 805459861]
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

        # 显式使用 torch.bitwise_xor 保证兼容性
        res = x * primes[0]
        res = torch.bitwise_xor(res, y * primes[1])
        res = torch.bitwise_xor(res, z * primes[2])
        return res % self.hashmap_size

    def forward(self, x):
        # --- 修改 3: 强制坐标 Clamp 在 [0, 1-eps] 之间，防止越界 ---
        x = torch.clamp(x, 0.0, 1.0 - 1e-4)

        out_features = []
        for i in range(self.num_levels):
            res = self.resolutions[i]
            x_scaled = x * res
            xi = x_scaled.long()
            xf = x_scaled - xi.float()

            # 这里的 offsets 建议移到构造函数或设为静态以提升性能
            offsets = torch.tensor([
                [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
            ], device=x.device)

            corners = xi.unsqueeze(1) + offsets
            hashed_indices = self.hash_func(corners)
            corner_features = self.embeddings[i](hashed_indices)

            w = xf.unsqueeze(1)
            # 三线性插值
            weights = torch.prod(torch.where(offsets == 1, w, 1 - w), dim=-1, keepdim=True)
            level_feature = torch.sum(corner_features * weights, dim=1)
            out_features.append(level_feature)

        return torch.cat(out_features, dim=-1)


class PositionalEncoding(nn.Module):
    """
    将低维坐标 (x, y, z) 映射到高维，让网络能学到高频细节。
    """

    def __init__(self, L=10):
        super().__init__()
        self.L = L

    def forward(self, x):
        # x 形状: [N, 3]
        out = [x]
        for i in range(self.L):
            for func in [torch.sin, torch.cos]:
                out.append(func(2 ** i * x))
        return torch.cat(out, dim=-1)  # 输出维度: 3 + 3 * 2 * L


class InstantNGP(nn.Module):
    def __init__(self, L_dir=4):
        super().__init__()
        self.hash_encoder = HashEncoder()

        self.dir_dim = 3 + 3 * 2 * L_dir
        self.dir_enc_fn = PositionalEncoding(L=L_dir)

        # --- 修改 4: 几何 MLP 增加初始偏置，防止初期 Sigma 全为负数被 ReLU 杀掉 ---
        self.sigma_net = nn.Sequential(
            nn.Linear(self.hash_encoder.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        # 初始化最后一个线性层，让 Sigma 初期接近一个微小的正数
        nn.init.constant_(self.sigma_net[-1].bias, 0.1)

        self.color_net = nn.Sequential(
            nn.Linear(15 + self.dir_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, x, d):
        # --- 修改 5: 归一化逻辑微调，确保物体处于 [0, 1] 中心 ---
        # TinyNeRF 数据通常在 [-1.5, 1.5] 之间，这里映射到 [0.125, 0.875]
        x = (x + 2.0) / 4.0

        x_embed = self.hash_encoder(x)
        h = self.sigma_net(x_embed)
        sigma = h[..., 0:1]
        geo_feat = h[..., 1:]

        d_embed = self.dir_enc_fn(d)
        color_h = torch.cat([geo_feat, d_embed], dim=-1)
        rgb = self.color_net(color_h)

        return torch.cat([rgb, sigma], dim=-1)


# --------------------------------------------------------------------------------
# 测试代码
# --------------------------------------------------------------------------------
if __name__ == '__main__':

    model = InstantNGP()
    # 模拟 5 条射线，每条射线上 64 个采样点
    test_x = torch.rand(5 * 64, 3)
    test_d = torch.rand(5 * 64, 3)

    ret = model(test_x, test_d)
    print(f"输入形状: {test_x.shape}")
    print(f"输出形状 (RGB+Sigma): {ret.shape}")  # [320, 4]

    # 计算参数量对比
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Instant-NGP 总参数量: {total_params / 1e6:.2f} M (大部分在哈希表里)")