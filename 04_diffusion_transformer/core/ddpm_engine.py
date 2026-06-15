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
        self.empty_label = cfg.data.num_classes - 1  # 假设最后一个索引是空标签
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
        # --- 原代码 ---
        # batch_size = x_real.shape[0]
        # t = [random.randint(0, self.n_steps - 1) for _ in range(batch_size)]
        # t_tensor = torch.LongTensor(t).to(x_real.device)
        # t_tensor = t_tensor.unsqueeze(-1)
        # eps = torch.randn_like(x_real)
        # alpha_bars = self.alpha_bars[t]
        # alpha_bars = alpha_bars.view(-1, 1, 1, 1)
        # x_t = x_real * torch.sqrt(alpha_bars) + eps * torch.sqrt(1 - alpha_bars)
        # return (x_t, t_tensor, eps)

        # --- 优化点：使用 torch.randint 直接在设备上生成随机数，避开 Python 循环 ---
        batch_size = x_real.shape[0]
        device = x_real.device
        t = torch.randint(0, self.n_steps, (batch_size,), device=device)
        t = t.unsqueeze(-1)

        eps = torch.randn_like(x_real)
        # 使用 t 进行切片并对齐维度
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1)

        x_t = torch.sqrt(a_bar) * x_real + torch.sqrt(1 - a_bar) * eps

        # 处理标签
        c = self.apply_cfg_dropout(label).to(x_real.device)
        c = c.unsqueeze(-1)

        return x_t, t, eps, c

    @torch.no_grad()
    def sample(self, net, shape, c, scale):
        net.eval()
        # 先从随机初始化开始
        x = torch.randn(shape).to(self.device)

        c_tensor = torch.LongTensor(c).to(self.device)
        c_tensor = c_tensor.unsqueeze(-1)

        # 每次给定当前x，t进行一次降噪
        for t_val in tqdm(range(self.n_steps - 1, -1, -1), desc="DDPM Sampling"):
            # 将标量 t_val 传进去，避免在 sample_step 里做多值布尔判断
            x = self.sample_step(x, t_val, net, c_tensor, scale)

        return x

    def sample_step(self, x, t, net, c, scale):
        # --- 原代码逻辑调整 ---
        # 原逻辑：eps_empty = net(x, t, c_empty); eps_guidance = net(x, t, c)
        # 原逻辑：if t == 0: ... 会因为 t 是 Tensor 而报错

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