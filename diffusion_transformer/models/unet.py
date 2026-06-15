import torch
import torch.nn as nn
import torch.nn.functional as F


# --- PositionalEncoding: 时间步编码 ---
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        assert d_model % 2 == 0

        # --- 原代码逻辑 (使用 Embedding) ---
        # self.embedding = nn.Embedding(max_seq_len, d_model)
        # self.embedding.weight.data = pe
        # self.embedding.requires_grad_(False)

        # --- 优化点：使用 register_buffer 存储固定权重，更符合 PyTorch 规范 ---
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # 计算除数项: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        # 注册为 buffer，不会被优化器更新，且随模型移动设备
        self.register_buffer('pe', pe)

    def forward(self, t):
        # t 形状期望为 [batch_size] 或 [batch_size, 1]
        t = t.view(-1)
        return self.pe[t]


# --- UnetBlock: UNet 的基本卷积单元 ---
class UnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, residual=False):
        super().__init__()
        # shape 用于 LayerNorm
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual

        if residual:
            self.residual_conv = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)

        if self.residual:
            out = out + self.residual_conv(x)

        out = self.activation(out)
        return out


# --- UNet: 核心网络结构 ---
class UNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        n_steps = cfg.method.n_steps
        n_classes = cfg.method.n_classes
        channels = cfg.unet.channels
        pe_dim = cfg.unet.pe_dim
        residual = cfg.unet.residual

        # 假设通过 get_image_shape 得到 (1, 28, 28)
        C, H, W = 1, 28, 28
        layers = len(channels)

        # 计算每一层分辨率
        Hs, Ws = [H], [W]
        cH, cW = H, W
        for _ in range(layers - 1):
            cH //= 2;
            cW //= 2
            Hs.append(cH);
            Ws.append(cW)

        self.pe = PositionalEncoding(n_steps, pe_dim)
        self.pe_c = nn.Embedding(n_classes, pe_dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()
        self.pe_linears_de = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Encoder 路径
        prev_channel = C
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            # 为每一层准备时间+条件的线性映射
            self.pe_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, prev_channel), nn.ReLU(),
                              nn.Linear(prev_channel, prev_channel)))

            self.encoders.append(nn.Sequential(
                UnetBlock((prev_channel, cH, cW), prev_channel, channel, residual=residual),
                UnetBlock((channel, cH, cW), channel, channel, residual=residual)
            ))
            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        # Mid 路径
        self.pe_mid = nn.Linear(pe_dim, prev_channel)
        mid_channel = channels[-1]
        self.mid = nn.Sequential(
            UnetBlock((prev_channel, Hs[-1], Ws[-1]), prev_channel, mid_channel, residual=residual),
            UnetBlock((mid_channel, Hs[-1], Ws[-1]), mid_channel, mid_channel, residual=residual),
        )
        prev_channel = mid_channel

        # Decoder 路径
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            # 优化点：decoder 的 pe 维度应该是当前输入通道数 (channel * 2)
            self.pe_linears_de.append(nn.Linear(pe_dim, prev_channel))
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))

            self.decoders.append(nn.Sequential(
                UnetBlock((channel * 2, cH, cW), channel * 2, channel, residual=residual),
                UnetBlock((channel, cH, cW), channel, channel, residual=residual)
            ))
            prev_channel = channel

        self.conv_out = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t, c):
        n = x.shape[0]
        # 1. 融合时间与条件
        t_emb = self.pe(t).view(n, -1)
        c_emb = self.pe_c(c).view(n, -1)
        condition = t_emb + c_emb

        encoder_outs = []
        # 2. Encoder 路径
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders, self.downs):
            pe = pe_linear(condition).view(n, -1, 1, 1)
            x = encoder(x + pe)
            encoder_outs.append(x)
            x = down(x)

        # 3. Middle 路径
        pe = self.pe_mid(condition).view(n, -1, 1, 1)
        x = self.mid(x + pe)

        # 4. Decoder 路径
        for pe_linear, decoder, up, skip in zip(self.pe_linears_de, self.decoders, self.ups, encoder_outs[::-1]):
            pe = pe_linear(condition).view(n, -1, 1, 1)
            x = up(x)

            # 尺寸对齐 (处理由于奇数导致的 pad 问题)
            pad_x = skip.shape[2] - x.shape[2]
            pad_y = skip.shape[3] - x.shape[3]
            if pad_x != 0 or pad_y != 0:
                x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2))

            x = torch.cat((skip, x), dim=1)
            x = decoder(x + pe)

        return self.conv_out(x)

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

        eps_out = self.forward(combined_x, combined_t, combined_c)
        eps_uncond, eps_cond = eps_out.chunk(2, dim=0)

        return eps_uncond + s * (eps_cond - eps_uncond)


# --- 配置与构建函数 ---
unet_res_cfg = {
    'channels': [16, 32, 64, 128],
    'pe_dim': 128,
    'residual': True
}


def build_network(config: dict, n_steps, n_classes):
    # 此处只处理 UNet 类型
    return UNet(n_steps, n_classes, **config)