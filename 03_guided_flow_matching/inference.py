import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ddpm import DDPM
from model import unet_res_cfg, build_network
from download_dataset import get_image_shape, show_images


def inference(ddpm, net, device, c, s, batch_size=10):

    # 增加引导强度系数s以及目标标签cond
    if not isinstance(c, list):
        c = [c for _ in range(batch_size)]
    if not isinstance(s, list):
        s = [s for _ in range(batch_size)]

    image_shape = get_image_shape()
    image_shape = (batch_size, *image_shape)
    ret = ddpm.sample_backward(image_shape, net, device, c=c, s=s)

    return ret


if __name__ == "__main__":

    n_steps = 10
    n_classes = 11
    device = "mps"

    s = [0, 0.5, 1.0, 2.5, 5.0, 0, 0.5, 1.0, 2.5, 5.0, 0, 0.5, 1.0, 2.5, 5.0]
    c = [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8]
    batch_size = len(s)

    config = unet_res_cfg
    net = build_network(config, n_steps, n_classes)

    ddpm = DDPM(n_steps=n_steps, device=device)
    ckpt_dir = "./ckpt"
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_epoch_030.pth")

    state_dict = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state_dict)
    net = net.to(device)

    print(f"[INFO] visualize checkpoint: {os.path.abspath(ckpt_path)}")
    with torch.no_grad():
        ret = inference(ddpm, net, device=device, c=c, s=s, batch_size=batch_size)
    ret = ret.detach().cpu()
    show_images(
        ret,
        num_rows=3,
        num_cols=5,
        labels=[f"Pred={cc}, s={ss}" for cc, ss in zip(c, s)],
    )

