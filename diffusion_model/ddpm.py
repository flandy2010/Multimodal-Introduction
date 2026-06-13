import torch


class DDPM():

    def __init__(self,
                 n_step: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.2,
                 device: torch.device = torch.device('cpu'),
                 ):

        self.betas = torch.linspace(min_beta, max_beta, n_step).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.ones_like(self.alphas)

        product = 1
        for i in range(len(self.alpha_bars)):
            product *= self.alphas[i]
            self.alpha_bars[i] = product

        self.device = device
        self.n_step = n_step

    def sample_forward(self, x, t, eps=None):
        # x: [batch_size, 1, h, w]
        # t: [batch_size, 1]

        assert len(x.shape) == 4
        assert len(t.shape) == 2 and t.shape[-1] == 1

        if eps is None:
            eps = torch.randn_like(x)
        alpha_bars = self.alpha_bars[t]
        eps = eps.view(-1, 1, 1, 1)
        x_with_noise = x * torch.sqrt(alpha_bars) + eps * torch.sqrt(1 - alpha_bars)

        return x_with_noise

    def sample_backward(self, image_shape, net, device, simple_var=True):

        # 先从随机初始化开始
        x = torch.randn(image_shape).to(self.device)

        # 每次给定当前x，t进行一次降噪
        for t in range(self.n_step, -1, -1):
            x = self.sample_backward_step(x, t=t, net=net, device=device, simple_var=simple_var)

        return x

    def sample_backward_step(self, x_t, t, net, device, simple_var):

        batch_size = x_t.shape[0]
        t_tensor = torch.LongTensor([[t] for _ in range(batch_size)], device=device) # [batch_size, 1]

        eps = net(x_t, t_tensor)

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




