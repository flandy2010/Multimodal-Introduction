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


class InstructionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        """
        :param vocab_size: 词表大小（你的字符总数，约 20-30）
        :param embed_dim: 每个字符的编码维度
        :param hidden_dim: GRU 的隐藏层维度（通常与 DiT 的 hidden_size 一致）
        """
        super().__init__()

        # 1. 字符嵌入层：将 [B, L] 转换为 [B, L, embed_dim]
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 2. GRU 层：处理序列信息
        # batch_first=True 保证输入输出形状为 [Batch, Seq, Dim]
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 3. 可选：线性投影层，确保输出维度与 DiT 完美匹配
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        :param x: 字符索引张量，形状 [batch_size, seq_len]
        :return: 指令表达向量，形状 [batch_size, hidden_dim]
        """
        # [B, L] -> [B, L, E]
        x = self.embedding(x)

        # 通过 GRU
        # output: 包含序列中每个时间步的隐藏状态 [B, L, H]
        # hn: 包含最后一个时间步的隐藏状态 [num_layers, B, H]
        output, hn = self.rnn(x)

        # 我们只需要序列结束时的“总结性”向量
        # 对于单层 GRU，hn.squeeze(0) 形状就是 [batch_size, hidden_dim]
        last_hidden = hn.squeeze(0)

        # 经过线性层做最后的特征变换
        out = self.fc(last_hidden)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t):
        # t: [batch_size] 整数索引
        return self.pe[t]


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
        n_steps = cfg.method.n_steps

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
        self.c_embedder = InstructionEncoder(vocab_size=cfg.data.vocab_size, embed_dim=hidden_size, hidden_dim=hidden_size)

        # C1. 2D位置编码 (固定，不参与训练)
        pos_embed = get_2d_sincos_pos_embed(hidden_size, input_size // patch_size)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0))

        # C2. 帧位置编码（参与训练）
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, cfg.video.num_frames, hidden_size))

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

        self.t_pe = PositionalEncoding(n_steps, hidden_size)

        # 初始化
        self.initialize_weights()

    def initialize_weights(self):
        # 参数初始化
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.normal_(self.temporal_pos_embed, std=0.02)
        # 依照 DiT 论文：将 adaLN 的输出层初始化为 0
        nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def forward(self, x, t, c):
        """
        x: [B, F, C, H, W]
        t: [B, 1]
        c: [B, L]
        """
        B, F, C, H, W = x.shape
        p = self.patch_size

        # 1. Patchify
        # [B, F, C, H, W] -> [B, F, C, H//p, p, W//p, p]
        x = x.reshape(B, F, C, H // p, p, W // p, p)
        # 换位并合并：[B, F, H//p, W//p, p, p, C]
        x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()

        num_spatial_patches = self.num_patches
        x = x.reshape(B, F * num_spatial_patches, -1)
        x = self.x_embedder(x)

        # 2. 3D 位置编码
        # 空间部分：[1, 196, D] -> [1, F*196, D] (重复 F 次)
        spatial_pos = self.pos_embed.repeat(1, F, 1)
        # 时间部分：[1, F, D] -> [1, F*196, D] (每个帧 ID 扩充 N 次)
        temporal_pos = self.temporal_pos_embed[:, :F, :].repeat_interleave(num_spatial_patches, dim=1)

        x = x + spatial_pos + temporal_pos

        # 3. 条件融合
        c_vec = self.c_embedder(c)  # [B, D] 来自 GRU
        t_pe = self.t_pe(t.view(-1))  # [B, D]
        t_vec = self.t_embedder(t_pe)  # [B, D]
        cond = t_vec + c_vec

        # 4. Transformer Blocks
        for block in self.blocks:
            x = block(x, cond)

        # 5. Unpatchify
        # 最后一层 adaLN
        res = self.final_adaLN_modulation(cond).chunk(2, dim=1)
        shift, scale = res[0], res[1]
        x = modulate(self.final_norm(x), shift, scale)
        x = self.final_linear(x)

        # 逆向还原形状
        # [B, F, H//p, W//p, p, p, C]
        x = x.reshape(B, F, H // p, W // p, p, p, C)
        # 换位：[B, F, C, H//p, p, W//p, p]
        x = x.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
        x = x.reshape(B, F, C, H, W)

        return x

    @torch.no_grad()
    def forward_with_cfg(self, x, t, c, s, empty_label):
        """
        s: 可以是 float, 也可以是 list/tensor [B]
        """
        # 确保 c_empty 的形状和设备与 c 一致
        c_empty = torch.full_like(c, fill_value=empty_label)

        # 合并推理
        combined_x = torch.cat([x, x], dim=0)
        combined_t = torch.cat([t, t], dim=0)
        combined_c = torch.cat([c_empty, c], dim=0)

        vector_out = self.forward(combined_x, combined_t, combined_c)
        vector_uncond, vector_cond = vector_out.chunk(2, dim=0)

        # 重点修正：处理 s 的广播维度
        if isinstance(s, (list, torch.Tensor)):
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, device=x.device, dtype=torch.float)
            s = s.view(-1, 1, 1, 1, 1)  # 5D 广播：[B, 1, 1, 1, 1]
        else:
            # 如果 s 是标量 float
            s = float(s)

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
            img_shape=(1, 28, 28),
            num_frames=12,
            vocab_size=20,
        ),
        method=SimpleNamespace(
            n_steps=1000
        ),
        common=SimpleNamespace(
            device="cpu"
        )
    )

    model = DiT(cfg)
    x = torch.randn(8, 12, 1, 28, 28)    # [B, F, C, H, W]
    t = torch.randint(0, 1000, (8, 1))   # [B, 1]
    c = torch.randint(1, 20, (8, 10))    # [B, L]

    out = model(x, t, c)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")  # 应该也是 [8, 1, 28, 28]

    out = model.forward_with_cfg(x, t, c, s=1.0, empty_label=10)