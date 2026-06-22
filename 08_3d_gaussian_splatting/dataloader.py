import numpy as np
import torch
import os


class GSDataLoader:
    def __init__(self, data_path="../data/tiny_nerf_data.npz"):
        data = np.load(data_path)
        self.images = torch.from_numpy(data['images']).float()  # [106, 100, 100, 3]
        self.poses = torch.from_numpy(data['poses']).float()  # [106, 4, 4]
        self.focal = float(data['focal'])
        self.H, self.W = self.images.shape[1:3]

        # 计算内参矩阵 K
        self.K = torch.tensor([
            [self.focal, 0, self.W / 2],
            [0, self.focal, self.H / 2],
            [0, 0, 1]
        ], dtype=torch.float32)

    def get_view_params(self, index):
        image = self.images[index]
        c2w = self.poses[index]
        # 3DGS 通常使用 world-to-camera 矩阵
        w2c = torch.inverse(c2w)
        return image, w2c, self.K


# 测试代码
if __name__ == "__main__":
    loader = GSDataLoader()
    img, w2c, K = loader.get_view_params(0)
    print(f"Image: {img.shape}, W2C: {w2c.shape}")