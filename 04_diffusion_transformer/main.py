import time
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid


from models.unet import UNet
from models.dit import DiT  # 假设以后有了
from core.fm_engine import FlowMatchingEngine
from core.ddpm_engine import DDPMEngine
from config import get_config, update_config
from utils.data import get_dataloader
from utils.logger import setup_experiment


def train(model, engine, cfg, exp_dir):

    # 1. 初始化 TensorBoard
    writer = SummaryWriter(log_dir=exp_dir)
    dataloader = get_dataloader(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.method.lr)
    loss_fn = nn.MSELoss()

    global_step = 0
    for epoch in range(1, cfg.method.epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        epoch_loss = 0

        for x, label in pbar:
            x, label = x.to(cfg.common.device), label.to(cfg.common.device)

            x_t, t, target, c = engine.get_train_data(x, label)
            predict = model(x_t, t, c)

            loss = loss_fn(predict, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录步级 Loss
            global_step += 1
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # 记录 Epoch 级 Loss
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar("Loss/epoch", avg_loss, epoch)

        # --- 每 5 个 Epoch 进行一次视觉采样 ---
        if epoch % 5 == 0:
            model.eval()
            # 固定生成 0-9 各一个数字，外加一个空标签（可选）
            sample_labels = list(range(10))
            # 调用引擎的采样函数
            with torch.no_grad():
                samples = engine.sample(
                    model,
                    shape=(10, *cfg.data.img_shape),
                    c=sample_labels,
                    scale=cfg.method.cfg_scale
                )

            # 逆归一化并保存
            samples = (samples * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(samples, nrow=5)
            save_image(grid, f"{exp_dir}/samples/epoch_{epoch}.png")
            writer.add_image("Visual/Samples", grid, epoch)

            # 保存权重
            torch.save(model.state_dict(), f"{exp_dir}/checkpoints/last.pth")

    writer.close()

def sample():
    pass


def parse_args():

    parser = argparse.ArgumentParser(description="MiniGen Project")
    parser.add_argument("--mode", type=str, choices=["train", "sample"])
    parser.add_argument("--method", type=str, choices=["ddpm", "flow_matching"])
    parser.add_argument("--n_steps", type=int, help="number of iteration steps")
    parser.add_argument("--n_classes", type=int, help="number of classes")
    parser.add_argument("--s", type=float, help="CFG scale")

    parser.add_argument("--model", type=str, choices=["unet", "dit"])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", type=str, default="cpu", help="training device")

    # 允许从命令行对ddpm定义min_beta和max_beta
    parser.add_argument("--min_beta", type=float, default=0.0001)
    parser.add_argument("--max_beta", type=float, default=0.2)

    # 允许从命令行输入列表：--channels 16 32 64 128
    parser.add_argument("--channels", type=int, nargs='+', help="UNet channels list")

    # 支持在推理的时候自定义测试数据
    parser.add_argument("--infer_labels", type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument("--infer_scales", type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.5, 5.0, 10.0])

    return parser.parse_args()


def main():

    cfg = get_config()
    args = parse_args()
    cfg = update_config(cfg, args)

    exp_dir = setup_experiment(cfg)
    print(f"[INFO] Experiment log will be saved to: {exp_dir}")

    if cfg.model_type == "unet":
        model = UNet(cfg)
    elif cfg.model_type == "dit":
        model = DiT(cfg)  # DiT 也可以设计成接收 cfg

    model.to(cfg.common.device)

    # 3. 选择算法引擎
    if cfg.method.type == "flow_matching":
        engine = FlowMatchingEngine(cfg)
    else:
        engine = DDPMEngine(cfg)

    # 4. 执行任务
    if cfg.common.mode == "train":
        train(model, engine, cfg, exp_dir)
    elif cfg.common.mode == "sample":
        sample(model, engine, cfg, exp_dir)


if __name__ == "__main__":
    main()