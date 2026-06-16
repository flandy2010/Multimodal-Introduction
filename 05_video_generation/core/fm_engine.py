import torch
from tqdm import tqdm
from .base_engine import BaseEngine


class FlowMatchingEngine(BaseEngine):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sample_steps = cfg.method.sample_steps  # 推理总步数，可以和训练步数不同

    def get_train_data(self, x_real, label):
        """
        x_real: [B, F, C, H, W] (视频序列)
        label:  [B, L] (指令 ID 序列)
        """
        batch_size = x_real.shape[0]
        device = x_real.device

        # 噪声维度和x_real维度保持一致
        eps = torch.randn_like(x_real)

        # 随机采样 [0, n_steps-1] 之间的整数，同时增加一个维度
        t_int = torch.randint(0, self.n_steps, (batch_size,), device=device)
        t_int = t_int.unsqueeze(-1)
        time_rate = t_int.float().view(batch_size, 1, 1, 1, 1) / self.n_steps

        x_t = time_rate * x_real + (1 - time_rate) * eps
        vector = x_real - eps

        c = self.apply_cfg_dropout(label).to(device)

        # 维度校验
        # x_t, vector: shape=[B, F, C, H, W]
        # t_int, shape=[B, 1]
        # c, shape=[B, L]
        # print(x_t.shape)
        # print(vector.shape)
        # print(t_int.shape)
        # print(c.shape)

        return x_t, t_int, vector, c

    @torch.no_grad()
    def sample(self, net, shape, c, scale):

        net.eval()
        x = torch.randn(shape).to(self.device)
        c = c.to(self.device)

        # 核心修改，这里的dt是基于推理步数算的：1/10
        dt = 1.0 / self.sample_steps

        for i in tqdm(range(self.sample_steps), desc="FlowMatching Sampling"):
            # 1. 计算当前的进度比例 (0.0 -> 1.0)
            progress = i / self.sample_steps

            # 2. 将进度比例映射回训练时的整数尺度
            t_curr = int(progress * self.n_steps)

            x = self.sample_step(x, t_curr, net, c, scale, dt)

        return x

    def sample_step(self, x, t, net, c, scale, dt):
        batch_size = x.shape[0]
        # 传给网络的 t 是映射后的“训练尺度”的 t
        t_tensor = torch.full((batch_size, 1), t, device=self.device, dtype=torch.long)

        vector = net.forward_with_cfg(x, t_tensor, c, scale, empty_label=self.empty_label)

        # 使用当前推理步数对应的 dt 进行更新
        x_next = x + dt * vector

        return x_next
