import torch
from tqdm import tqdm


class DDPM():

    def __init__(self,
                 n_steps: int,
                 device: torch.device = torch.device('cpu'),
                 ):

        self.device = device
        self.n_steps = n_steps

    def sample_forward(self, x, t, eps=None):
        # x: [batch_size, 1, h, w]
        # t: [batch_size, 1]

        assert len(x.shape) == 4
        assert len(t.shape) == 2 and t.shape[-1] == 1

        if eps is None:
            eps = torch.randn_like(x)

        # time_rate=1表示恢复到原图
        time_rate = t / self.n_steps
        time_rate = time_rate.view(-1, 1, 1, 1)

        x_with_noise = x * time_rate + (1 - time_rate) * eps

        return x_with_noise

    def sample_backward(self, image_shape, net, device, c, s=1.0):
        # 1. 从随机初始化开始 (此时对应 t=0)
        x = torch.randn(image_shape).to(self.device)

        batch_size = x.shape[0]
        if not isinstance(c, list):
            c = [c for _ in range(batch_size)]
        if not isinstance(s, list):
            s = [s for _ in range(batch_size)]

        # 2. 【关键修改】时间从 0 走到 n_steps-1
        # 在 Flow Matching 中，我们是从噪声 (t=0) 向原图 (t=1) 演化
        for t in tqdm(range(self.n_steps), desc="Inference"):
            x = self.sample_backward_step(x, t=t, c=c, s=s, net=net, device=device)

        return x

    def sample_backward_step(self, x_t, t, c, s, net, device):
        batch_size = x_t.shape[0]
        t_tensor = torch.LongTensor([[t] for _ in range(batch_size)]).to(device)

        # 条件和空条件 Tensor 准备
        c_tensor = torch.LongTensor(c).to(device).unsqueeze(1)
        c_tensor_empty = torch.LongTensor([[10] for _ in range(batch_size)]).to(device)

        # 推理速度向量 v
        v_uncond = net(x_t, t_tensor, c_tensor_empty)
        v_cond = net(x_t, t_tensor, c_tensor)

        # CFG 结合
        s_tensor = torch.FloatTensor(s).to(device).view(-1, 1, 1, 1)
        v_final = v_uncond + s_tensor * (v_cond - v_uncond)

        # 3. 【核心更新公式】欧拉积分
        # x_{t+1} = x_t + v * dt
        # 这里的 dt = 1.0 / n_steps
        dt = 1.0 / self.n_steps
        x_t_next = x_t + v_final * dt

        return x_t_next




