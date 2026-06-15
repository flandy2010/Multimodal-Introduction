import torch
import torch.nn as nn
import numpy as np
import math


# --- 1. 2D 正余弦位置编码 ---
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: 补丁网格的大小 (例如 14x14)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    # 实际shape=[2, 1, 14, 14]
    # grid[0]表示了所有patch的横坐标，grid[1]表示了所有patch的纵坐标
    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # 使用一半的维度来编码网格的 H，另一半编码 W
    emb_h = get_1d_sincos_pos_embed_from_values(embed_dim // 2, grid[0].flatten())
    emb_w = get_1d_sincos_pos_embed_from_values(embed_dim // 2, grid[1].flatten())
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    return pos_embed


def get_1d_sincos_pos_embed_from_values(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# --- 2. adaLN-Zero 调制层 ---
def modulate(x, shift, scale):
    # x: [B, T, D], shift/scale: [B, D]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# --- 3. DiT Block: 核心 Transformer 块 ---
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        # adaLN-Zero 预测 6 个参数 (针对两个 LayerNorm 的 shift, scale, gate)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x, c):
        # x: [B, T, D], c: [B, D] (已融合的条件向量)
        # 以下6个变量shape: [B, D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-Attention 部分
        res = modulate(self.norm1(x), shift_msa, scale_msa)
        res, _ = self.attn(res, res, res)
        x = x + gate_msa.unsqueeze(1) * res

        # FFN 部分
        res = modulate(self.norm2(x), shift_mlp, scale_mlp)
        res = self.mlp(res)
        x = x + gate_mlp.unsqueeze(1) * res
        return x


# --- 4. DiT 模型主体 ---
class DiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_size, patch_size, in_channels, hidden_size = cfg.dit.input_size, cfg.dit.patch_size, cfg.dit.in_channels, cfg.dit.hidden_size
        depth, num_heads = cfg.dit.depth, cfg.dit.num_heads
        num_classes = cfg.data.num_classes

        self.in_channels = cfg.dit.in_channels
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2

        # A. Patchify: 将图像切块并投影
        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, hidden_size)

        # B. 时间与类别嵌入
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.c_embedder = nn.Embedding(num_classes, hidden_size)

        # C. 2D 位置编码 (固定，不参与训练)
        pos_embed = get_2d_sincos_pos_embed(hidden_size, input_size // patch_size)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0))

        # D. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # E. Final Layer: Unpatchify
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        self.final_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

        # 初始化
        self.initialize_weights()

    def initialize_weights(self):
        # 依照 DiT 论文进行初始化
        nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def forward(self, x, t, c):
        """
        x: [B, 1, 28, 28]
        t: [B, 1] (经过 PE 预处理后的向量，或者这里直接传入时间步 ID)
        c: [B, 1] (类别 ID)
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # 1. Patchify: [B, 1, 28, 28] -> [B, 196, p*p]
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 3, 5, 1).flatten(3)
        x = x.reshape(B, -1, p * p * C)
        x = self.x_embedder(x)  # [B, T, D]
        x = x + self.pos_embed  # 加上 2D 位置编码

        # 2. 条件融合 (这里简单处理：将 t 和 c 的嵌入相加)
        c_vec = self.c_embedder(c.view(-1))
        t_vec = self.t_embedder(torch.randn(B, 128, device=x.device))  # 占位

        # cond.shape = [B, D]
        cond = t_vec + c_vec

        # 3. Transformer 推理
        for block in self.blocks:
            x = block(x, cond)

        # 4. Final Layer / Unpatchify
        shift, scale = self.final_adaLN_modulation(cond).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.final_linear(x)  # [B, T, p*p*C]

        # 重排回图像形状: [B, 1, 28, 28]
        x = x.reshape(B, H // p, W // p, p, p, C).permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, H, W)

        return x

    @torch.no_grad()
    def forward_with_cfg(self, x, t, c, s, empty_label):
        """
        封装推理时的 CFG 逻辑
        s: 引导强度 (Guidance Scale)
        """
        # 假设标签 10 是空标签
        c_empty = torch.full_like(c, fill_value=empty_label)

        # 合并 Batch 一次推理
        combined_x = torch.cat([x, x], dim=0)
        combined_t = torch.cat([t, t], dim=0)
        combined_c = torch.cat([c_empty, c], dim=0)

        vector_out = self.forward(combined_x, combined_t, combined_c)
        vector_uncond, vector_cond = vector_out.chunk(2, dim=0)

        if isinstance(s, list):
            assert len(s) == vector_uncond.shape[0]
            s = torch.FloatTensor(s).view(-1, 1, 1, 1)
            s = s.to(vector_out.device)

        return vector_uncond + s * (vector_cond - vector_uncond)


# --- 5. 测试 ---
if __name__ == "__main__":

    from types import SimpleNamespace
    import torch

    cfg = SimpleNamespace(
        dit=SimpleNamespace(
            input_size=28,
            patch_size=2,
            in_channels=1,
            hidden_size=128,
            depth=4,
            num_heads=4
        ),
        data=SimpleNamespace(
            num_classes=11,
            img_shape=(1, 28, 28)
        ),
        method=SimpleNamespace(
            n_steps=1000
        ),
        common=SimpleNamespace(
            device="cpu"
        )
    )

    model = DiT(cfg)
    x = torch.randn(8, 1, 28, 28)
    t = torch.randn(8, 128)  # 假设时间已经转为向量
    c = torch.randint(0, 10, (8, 1))

    out = model(x, t, c)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")  # 应该也是 [8, 1, 28, 28]

    out = model.forward_with_cfg(x, t, c, s=1.0, empty_label=10)