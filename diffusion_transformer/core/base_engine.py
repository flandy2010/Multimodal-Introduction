import torch
from abc import ABC, abstractmethod

class BaseEngine(ABC):

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_steps = cfg.method.n_steps
        self.device = cfg.common.device

        self.drop_rate = cfg.method.drop_rate
        self.empty_label = cfg.data.num_classes - 1  # 假设最后一个是空标签

    @abstractmethod
    def get_train_data(self, x_real, label):
        """
        输入: 原图 x1 (Flow Matching) 或 x0 (DDPM)
        输出: (x_t, t_tensor, target, c)
        """
        pass

    def apply_cfg_dropout(self, label):
        """
        通用的标签丢弃逻辑，供子类调用
        """
        # 复制一份标签，避免修改原始数据
        c = label.clone()
        if self.drop_rate > 0:
            # 生成随机掩码
            mask = torch.rand(c.shape, device=c.device) < self.drop_rate
            c[mask] = self.empty_label
        c = c.long()
        return c

    @abstractmethod
    @torch.no_grad()
    def sample(self, net, shape, c, scale):
        """
        输入: 网络, 图像形状, 类别标签, 引导强度
        输出: 生成的图像
        """
        pass