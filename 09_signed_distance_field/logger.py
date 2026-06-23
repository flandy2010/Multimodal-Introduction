import matplotlib.pyplot as plt
import os
import torch
import numpy as np


class SDFLogger:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        self.log_file = open(os.path.join(exp_dir, "log.txt"), "w")

    def report(self, step, loss, psnr):
        line = f"Step: {step} | Loss: {loss:.6f} | PSNR: {psnr:.2f}\n"
        print(line, end="")
        self.log_file.write(line)
        self.log_file.flush()

    @torch.no_grad()
    def save_preview(self, step, pred_img, gt_img):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(gt_img.cpu().numpy());
        ax[0].set_title("GT")
        ax[1].imshow(pred_img.cpu().numpy());
        ax[1].set_title(f"Step {step}")
        plt.savefig(os.path.join(self.exp_dir, f"render_{step:04d}.png"))
        plt.close()

    @torch.no_grad()
    def visualize_sdf_slice(self, step, sdf_net, device):
        # 绘制 Z=0 处的 SDF 切面图
        res = 100
        grid = torch.linspace(-1.2, 1.2, res)
        x, y = torch.meshgrid(grid, grid, indexing='ij')
        pts = torch.stack([x, y, torch.zeros_like(x)], dim=-1).to(device).reshape(-1, 3)
        sdf, _ = sdf_net(pts)
        sdf = sdf.reshape(res, res).cpu().numpy()

        plt.figure()
        plt.imshow(sdf, cmap='seismic')
        plt.colorbar()
        plt.title(f"SDF Slice Step {step}")
        plt.savefig(os.path.join(self.exp_dir, f"sdf_{step:04d}.png"))
        plt.close()