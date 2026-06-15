import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ddpm import DDPM
from model import unet_res_cfg, build_network


# --- 新增：损失统计辅助类 ---
class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 建议扩散模型使用 [-1, 1] 归一化
    ])

    train_dataset = datasets.MNIST(
        root='../data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader


def train(ddpm, net, device, ckpt_path):
    # 基础配置
    batch_size = 512
    n_epochs = 50
    lr = 1e-3

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    print(f"Starting training on {device}...")

    for e in range(1, n_epochs + 1):
        # 初始化本 Epoch 的 Loss 统计
        epoch_loss = AverageMeter()

        # 使用 tqdm 包装 dataloader，并设置显示格式
        pbar = tqdm(dataloader, dynamic_ncols=True)

        net.train()  # 确保开启 Train 模式
        for x, _ in pbar:
            batch_size = x.size(0)
            x = x.to(device)

            # 1. 采样时间步 t
            t = torch.randint(0, n_steps, (batch_size, 1)).to(device)

            # 2. 采样噪声 eps
            eps = torch.randn_like(x).to(device)

            # 3. 前向加噪得到 x_t
            x_t = ddpm.sample_forward(x, t, eps)

            # 4. 模型预测噪声
            pred_eps = net(x_t, t)

            # 5. 反向传播
            optimizer.zero_grad()
            loss = loss_fn(pred_eps, eps)
            loss.backward()
            optimizer.step()

            # 6. 更新统计量
            epoch_loss.update(loss.item(), batch_size)

            # 7. 更新进度条右侧显示内容
            pbar.set_description(f"Epoch {e:03d}/{n_epochs}")
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{epoch_loss.avg:.4f}")

        # Epoch 结束后的输出
        print(f"==> Epoch {e:03d} Final Avg Loss: {epoch_loss.avg:.6f}")

        # 定期保存
        if e % 10 == 0:
            save_name = f"checkpoint_epoch_{e:03d}.pth"
            save_file = os.path.join(ckpt_path, save_name)
            torch.save(net.state_dict(), save_file)
            print(f"[SAVE] Checkpoint saved: {save_file}")


if __name__ == '__main__':

    # 检查是否有 MPS (Mac) 或 CUDA (NVIDIA)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    n_steps = 1000
    config = unet_res_cfg
    net = build_network(config, n_steps)

    ddpm = DDPM(n_steps=n_steps, device=device)

    train(
        ddpm=ddpm,
        net=net,
        device=device,
        ckpt_path='./ckpt',
    )