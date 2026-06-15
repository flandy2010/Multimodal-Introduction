import torch
from tqdm import tqdm
from .base_engine import BaseEngine


class FlowMatchingEngine(BaseEngine):

    def get_train_data(self, x_real, label):

        batch_size = x_real.shape[0]
        device = x_real.device
        t = torch.randint(0, self.n_steps, (batch_size,), device=device)
        t = t.unsqueeze(-1)

        eps = torch.randn_like(x_real)
        time_rate = t.float() / self.n_steps
        time_rate = time_rate.view(batch_size, 1, 1, 1)

        x_t = time_rate * x_real + (1 - time_rate) * eps
        vector = x_real - eps

        # 处理标签
        c = self.apply_cfg_dropout(label).to(x_real.device)
        c = c.unsqueeze(-1)

        return x_t, t, vector, c

    @torch.no_grad()
    def sample(self, net, shape, c, scale):
        net.eval()
        # 先从随机初始化开始
        x = torch.randn(shape).to(self.device)

        c_tensor = torch.LongTensor(c).to(self.device)
        c_tensor = c_tensor.unsqueeze(-1)

        # 每次给定当前x，t进行一次降噪
        for t_val in tqdm(range(self.n_steps), desc="FlowMatching Sampling"):
            # 将标量 t_val 传进去，避免在 sample_step 里做多值布尔判断
            x = self.sample_step(x, t_val, net, c_tensor, scale)

        return x

    def sample_step(self, x, t, net, c, scale):

        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size, 1), t, device=self.device, dtype=torch.long)
        vector = net.forward_with_cfg(x, t_tensor, c, scale, empty_label=self.empty_label)
        x_next = x + 1 / self.n_steps * vector

        return x_next

