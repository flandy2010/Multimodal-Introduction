import torch
import torch.nn as nn
import torch.nn.functional as F


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


class NeRF(nn.Module):
    """
    标准 NeRF 网络：输入位置和视角，输出颜色(RGB)和密度(sigma)
    """

    def __init__(self, D=8, W=256, L_pos=10, L_dir=4):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir

        self.pos_enc = PositionalEncoding(L_pos)
        self.dir_enc = PositionalEncoding(L_dir)

        # 几何部分 (只输入位置)
        in_dim_pos = 3 + 3 * 2 * L_pos
        self.pts_linears = nn.ModuleList([nn.Linear(in_dim_pos, W)])
        for i in range(D - 1):
            if i == 4:  # 第 5 层增加残差连接 (Skip Connection)
                self.pts_linears.append(nn.Linear(W + in_dim_pos, W))
            else:
                self.pts_linears.append(nn.Linear(W, W))

        self.alpha_linear = nn.Linear(W, 1)  # 输出密度 sigma

        # 外观部分 (输入特征 + 视角)
        in_dim_dir = 3 + 3 * 2 * L_dir
        self.feature_linear = nn.Linear(W, W)
        self.views_linear = nn.Linear(W + in_dim_dir, W // 2)
        self.rgb_linear = nn.Linear(W // 2, 3)  # 输出颜色 RGB

    def forward(self, x, d):
        # 1. 位置编码
        x_embed = self.pos_enc(x)
        d_embed = self.dir_enc(d)

        # 2. 预测密度
        h = x_embed
        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h))
            if i == 4: h = torch.cat([x_embed, h], dim=-1)

        sigma = self.alpha_linear(h)
        feature = self.feature_linear(h)

        # 3. 预测颜色 (结合视角)
        h = torch.cat([feature, d_embed], dim=-1)
        h = F.relu(self.views_linear(h))
        rgb = torch.sigmoid(self.rgb_linear(h))

        return torch.cat([rgb, sigma], dim=-1)  # [N, 4]


if __name__ == '__main__':

    positional_encoding = PositionalEncoding()
    test = torch.randint(1, 10, (5, 3)).long()
    ret = positional_encoding(test)
    print(ret.shape)