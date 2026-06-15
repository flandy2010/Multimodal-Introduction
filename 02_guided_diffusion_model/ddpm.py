import torch
from tqdm import tqdm


class DDPM():

    def __init__(self,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.2,
                 device: torch.device = torch.device('cpu'),
                 ):

        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.ones_like(self.alphas).to(device)

        product = 1
        for i in range(len(self.alpha_bars)):
            product *= self.alphas[i]
            self.alpha_bars[i] = product

        self.device = device
        self.n_steps = n_steps

    def sample_forward(self, x, t, eps=None):
        # x: [batch_size, 1, h, w]
        # t: [batch_size, 1]

        assert len(x.shape) == 4
        assert len(t.shape) == 2 and t.shape[-1] == 1

        if eps is None:
            eps = torch.randn_like(x)
        alpha_bars = self.alpha_bars[t]
        alpha_bars = alpha_bars.view(-1, 1, 1, 1)

        x_with_noise = x * torch.sqrt(alpha_bars) + eps * torch.sqrt(1 - alpha_bars)

        return x_with_noise

    def sample_backward(self, image_shape, net, device, c, s=1.0, simple_var=True):

        # 先从随机初始化开始
        x = torch.randn(image_shape).to(self.device)

        # 采用list形式方便批量生成进行对比
        batch_size = x.shape[0]
        if not isinstance(c, list):
            c = [c for _ in range(batch_size)]
        if not isinstance(s, list):
            s = [s for _ in range(batch_size)]

        # 每次给定当前x，t进行一次降噪
        for t in tqdm(range(self.n_steps - 1, -1, -1), desc="Inference"):
            x = self.sample_backward_step(x, t=t, c=c, s=s, net=net, device=device, simple_var=simple_var)

        return x

    def sample_backward_step(self, x_t, t, c, s, net, device, simple_var):

        batch_size = x_t.shape[0]
        t_tensor = torch.LongTensor([[t] for _ in range(batch_size)]) # [batch_size, 1]
        t_tensor = t_tensor.to(device)

        # 增加【控制条件=目标值】的tensor
        c_tensor = torch.LongTensor(c).to(device)
        c_tensor = c_tensor.unsqueeze(1)

        # 增加【控制条件=空值】的tensor，本例中empty_set对应标签为10
        c_tensor_empty = torch.LongTensor([[10] for _ in range(batch_size)])
        c_tensor_empty = c_tensor_empty.to(device)

        # 推理两次，控制条件分别用空值和目标值
        eps_empty = net(x_t, t_tensor, c_tensor_empty)
        eps_guidance = net(x_t, t_tensor, c_tensor)

        s_tensor = torch.FloatTensor(s).to(device)
        s_tensor = s_tensor.view(-1, 1, 1, 1)

        eps = eps_empty + s_tensor * (eps_guidance - eps_empty)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                # 方差\tilde{\beta_t} = \beta_t
                beta_t = self.betas[t]
            else:
                # 方差\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t}} \cdot \beta_t
                beta_t = self.betas[t] * (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t])

            # 计算用于采样x_{t-1}的方差
            noise = torch.randn_like(x_t)
            noise = noise * torch.sqrt(beta_t)

        # 计算用于采样x_{t-1}的均值
        mean = 1 / torch.sqrt(self.alphas[t]) * (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps)

        x_t = mean + noise

        return x_t




