import torch
import torch.nn as nn
import random
from tqdm import tqdm
from .base_engine import BaseEngine


class DDPMEngine(BaseEngine):

    def __init__(self, cfg):
        super().__init__(cfg)
        # --- 原代码 ---
        # self.min_beta = cfg.min_beta
        # self.max_beta = cfg.max_beta
        # self.empty_label = cfg.empty_label
        # self.preprare_schedule()

        # --- 优化点：从 cfg 规范化引用参数 ---
        self.min_beta = cfg.method.min_beta
        self.max_beta = cfg.method.max_beta
        self.prepare_schedule()

    def prepare_schedule(self):
        # --- 原代码 ---
        # self.betas = torch.linspace(self.min_beta, self.max_beta, self.n_steps).to(self.device)
        # self.alphas = 1 - self.betas
        # self.alpha_bars = torch.ones_like(self.alphas).to(self.device)
        # product = 1
        # for i in range(len(self.alpha_bars)):
        #     product *= self.alphas[i]
        #     self.alpha_bars[i] = product

        # --- 优化点：使用 torch.cumprod 实现向量化计算，更符合 PyTorch 习惯 ---
        self.betas = torch.linspace(self.min_beta, self.max_beta, self.n_steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_train_data(self, x_real, label):

        batch_size = x_real.shape[0]
        device = x_real.device

        # 随机采样 [0, n_steps-1] 之间的整数，同时增加一个维度
        t_int = torch.randint(0, self.n_steps, (batch_size,), device=device)
        t_int = t_int.unsqueeze(-1)

        eps = torch.randn_like(x_real)
        # 使用 t 进行切片并对齐维度
        a_bar = self.alpha_bars[t_int].view(-1, 1, 1, 1, 1)

        x_t = torch.sqrt(a_bar) * x_real + torch.sqrt(1 - a_bar) * eps

        # 处理指令
        c = self.apply_cfg_dropout(label).to(device)

        return x_t, t_int, eps, c

    @torch.no_grad()
    def sample(self, net, shape, c, scale):
        net.eval()
        # 先从随机初始化开始
        x = torch.randn(shape).to(self.device)
        c = c.to(self.device)

        # 每次给定当前x，t进行一次降噪
        for t_val in tqdm(range(self.n_steps - 1, -1, -1), desc="DDPM Sampling"):
            # 将标量 t_val 传进去，避免在 sample_step 里做多值布尔判断
            x = self.sample_step(x, t_val, net, c, scale)

        return x

    def sample_step(self, x, t, net, c, scale):

        batch_size = x.shape[0]
        # 将标量 t 包装成 Tensor 供网络使用
        t_tensor = torch.full((batch_size, 1), t, device=self.device, dtype=torch.long)

        # 调用模型内部提供的并行推理
        eps = net.forward_with_cfg(x, t_tensor, c, scale, empty_label=self.empty_label)

        # --- 数学参数提取 ---
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]

        # --- 优化点：利用标量 t 进行判断，逻辑更清晰 ---
        if t == 0:
            noise = 0
        else:
            # 后验方差计算
            alpha_bar_prev = self.alpha_bars[t - 1]
            beta_t_pdf = self.betas[t] * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            noise = torch.randn_like(x) * torch.sqrt(beta_t_pdf)

        # DDPM 均值公式
        mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps
        )

        return mean + noise