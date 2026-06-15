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


def inference(ddpm, net, device, batch_size=10):

    image_shape = get_image_shape()
    image_shape = (batch_size, *image_shape)
    ret = ddpm.sample_backward(image_shape, net, device, simple_var=True)

    return ret


if __name__ == "__main__":

    n_steps = 1000
    device = "mps"

    config = unet_res_cfg
    net = build_network(config, n_steps)

    ddpm = DDPM(n_steps=n_steps, device=device)
    ckpt_dir = "./ckpt"
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_epoch_050.pth")

    state_dict = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state_dict)
    net = net.to(device)

    print(f"[INFO] visualize checkpoint: {os.path.abspath(ckpt_path)}")
    with torch.no_grad():
        ret = inference(ddpm, net, device=device, batch_size=10)
    ret = ret.detach().cpu()
    show_images(ret, labels=["Pred" for _ in range(n_steps)])

