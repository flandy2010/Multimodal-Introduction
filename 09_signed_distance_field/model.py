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
    """
    SDF 网络，完全参照 NeuS 原版实现（borrowed from IDR）。

    关键差异（相比朴素实现）：
    1. skip 层的线性层输出是 W - input_ch（不是 W），然后和输入拼接凑成 W
       → skip 层只学"残差"，输入 pass-through，/ sqrt(2) 保持量级
    2. forward 里 skip 时 x = cat([x, inputs]) / sqrt(2)
    3. 最终 SDF 值除以 scale（坐标归一化系数），其余 feature 不除
    4. IDR 几何初始化：确保 SDF 初始输出 ≈ ||x|| - bias（球体先验）
    """

    def __init__(self, d_in=3, d_out=257, d_hidden=256, n_layers=8,
                 skip_in=(4,), multires=6, bias=0.5, scale=1.0,
                 geometric_init=True, weight_norm=True, inside_outside=False):
        super().__init__()

        # dims[0] 初始为 d_in，有 PE 时替换为 PE 输出维度
        dims = [d_in] + [d_hidden] * n_layers + [d_out]

        self.embed_fn_fine = None
        if multires > 0:
            self.embed_fn_fine = PositionalEncoding(multires)
            dims[0] = 3 + 3 * 2 * multires   # PE 展开后维度

        self.num_layers = len(dims)
        self.skip_in   = skip_in
        self.scale      = scale

        for l in range(self.num_layers - 1):
            # NeuS 原版：skip 层的输出是 W - input_ch，不是 W
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # 最后一层（SDF 头 + feature 头合并输出）
                    if not inside_outside:
                        nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        nn.init.constant_(lin.bias, -bias)
                    else:
                        nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    # 第一层：PE 维度（后 dims[0]-3 列）权重置 0
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.constant_(lin.weight[:, 3:], 0.0)
                    nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    # skip 层：skip 拼接进来的 PE 部分（后 dims[0]-3 列）置 0
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        # 坐标缩放（NeuS 原版用 scale 做归一化）
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            # skip 连接：cat 后 / sqrt(2) 保持量级
            if l in self.skip_in:
                x = torch.cat([x, inputs], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        # 最终输出：SDF 值 / scale，feature 不除
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        """仅返回 SDF 值 [N, 1]"""
        return self.forward(x)[:, :1]

    def sdf_and_feature(self, x):
        """返回 SDF [N, 1] 和 feature [N, d_hidden]"""
        out = self.forward(x)
        return out[:, :1], out[:, 1:]


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
    normals = torch.randn(100, 3)
    normals = normals / normals.norm(dim=-1, keepdim=True)

    sdf, feats = sdf_net(pts)
    rgb = color_net(pts, dirs, feats, normals)
    s = variance()

    print(f"SDF shape: {sdf.shape}, s = {s.item():.4f}")
    print(f"SDF shape: {sdf.shape}, range: [{sdf.min():.3f}, {sdf.max():.3f}]")
    print(f"RGB shape: {rgb.shape}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")
