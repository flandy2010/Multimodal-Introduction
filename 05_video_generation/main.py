import os
import time
import torch
import torch.nn as nn
import argparse
import itertools
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid


from models.unet import UNet
from models.dit import DiT
from core.fm_engine import FlowMatchingEngine
from core.ddpm_engine import DDPMEngine
from config import get_config, update_config
from video_dataloader import get_video_dataloader, CharTokenizer
from video_generator import VideoTransformEngine
from logger import setup_experiment, load_config_from_dir


def train(model, engine, cfg, exp_dir):

    # 1. 初始化 TensorBoard
    writer = SummaryWriter(log_dir=exp_dir)
    dataloader = get_video_dataloader(cfg)
    tokenizer = CharTokenizer()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.method.lr)
    loss_fn = nn.MSELoss()

    global_step = 0
    for epoch in range(1, cfg.method.epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        epoch_loss = 0

        for item in pbar:

            # item: dict, keys=["video", "inst_ids", "inst_text", "label"]
            # video.shape=[B, F, C, H, W], inst_ids.shape=[B, L=10]
            video, inst_ids = item["video"], item["inst_ids"]
            video, inst_ids = video.to(cfg.common.device), inst_ids.to(cfg.common.device)

            video_t, t, target, cond_ids = engine.get_train_data(video, inst_ids)
            predict = model(video_t, t, cond_ids)

            # predict&target维度一致, shape=[B, F, C, H, W]
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

            instructions = ["将2水平翻转", "将4垂直翻转", "将5放大1倍", "将7缩小2倍", "将9旋转30度", "将1旋转120度"]
            inst_ids = tokenizer.encode_batch(instructions, return_tensor=True)

            with torch.no_grad():
                samples = engine.sample(
                    model,
                    shape=(len(instructions), *cfg.video.video_shape),
                    c=inst_ids,
                    scale=cfg.method.cfg_scale
                )
                samples = samples.detach().cpu()
                epoch = 0
                for instruction, sample in zip(instructions, samples):
                    VideoTransformEngine.save_to_grid_image(
                        video_tensor=sample,
                        inst_text=f"epoch_{epoch}_{instruction}",
                        save_dir=f"{exp_dir}/samples/"
                    )

            # 保存权重
            torch.save(model.state_dict(), f"{exp_dir}/checkpoints/last.pth")

    writer.close()

def sample(model, engine, cfg, exp_dir):

    ckpt_path = f"{exp_dir}/checkpoints/last.pth"
    state_dict = torch.load(ckpt_path, map_location=cfg.common.device)

    model.load_state_dict(state_dict)
    model = model.to(cfg.common.device)

    # TODO


def parse_args():

    parser = argparse.ArgumentParser(description="MiniGen Project")
    parser.add_argument("--mode", type=str, choices=["train", "sample"])
    parser.add_argument("--method", type=str, choices=["ddpm", "flow_matching"])
    parser.add_argument("--n_steps", type=int, help="number of training iteration steps")
    parser.add_argument("--n_classes", type=int, help="number of classes")
    parser.add_argument("--s", type=float, help="CFG scale")
    parser.add_argument("--sample_steps", type=int, help="number of inference iteration steps")

    parser.add_argument("--model", type=str, choices=["unet", "dit"])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", type=str, default="cpu", help="training device")

    # 允许从命令行对ddpm定义min_beta和max_beta
    parser.add_argument("--min_beta", type=float, default=0.0001)
    parser.add_argument("--max_beta", type=float, default=0.2)

    # 允许从命令行自定义dit的patch_size
    parser.add_argument("--patch_size", type=int, default=2)

    # 允许从命令行输入列表：--channels 16 32 64 128
    parser.add_argument("--channels", type=int, nargs='+', help="UNet channels list")

    # 支持在推理的时候自定义测试数据
    parser.add_argument("--exp_dir", type=str, help="Model Checkpoint dir")
    parser.add_argument("--infer_mode", type=str, default="product", choices=["zip", "product"])
    parser.add_argument("--infer_labels", type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument("--infer_scales", type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.5, 5.0, 10.0])

    return parser.parse_args()


def run_train(args):

    cfg = get_config()
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
    train(model, engine, cfg, exp_dir)


def run_sample(args):

    cfg = load_config_from_dir(args.exp_dir)
    cfg = update_config(cfg, args)

    if cfg.model_type == "unet":
        model = UNet(cfg)
    elif cfg.model_type == "dit":
        model = DiT(cfg)  # DiT 也可以设计成接收 cfg

    ckpt_path = os.path.join(args.exp_dir, "checkpoints/last.pth")
    state_dict = torch.load(ckpt_path, map_location=cfg.common.device)
    model.load_state_dict(state_dict)
    model.to(cfg.common.device)

    # 3. 选择算法引擎
    if cfg.method.type == "flow_matching":
        engine = FlowMatchingEngine(cfg)
    else:
        engine = DDPMEngine(cfg)

    sample(model, engine, cfg, args.exp_dir)


def main():

    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "sample":
        run_sample(args)


if __name__ == "__main__":
    main()