import torch
from abc import ABC, abstractmethod

class BaseEngine(ABC):

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_steps = cfg.method.n_steps
        self.device = cfg.common.device

        self.drop_rate = cfg.method.drop_rate
        self.empty_label = 0  # 对应<pad>标记

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
        label为文本指令，shape=[batch, seq_len]
        """
        # 复制一份标签，避免修改原始数据
        c = label.clone()
        B, L = label.shape
        if self.drop_rate > 0:
            # 决定哪些 Batch 需要被丢弃
            mask = torch.rand(B, device=label.device) < self.drop_rate
            # 将选中的整个序列替换为<pad>对应的字符
            c[mask] = 0
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