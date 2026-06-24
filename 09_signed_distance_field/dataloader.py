import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TinySDFDataset(Dataset):
    """与 06 NeRF 的 TinyNeRFDataset 保持一致的数据加载接口"""

    def __init__(self, data_path="../data/tiny_nerf_data.npz", mode='train', test_idx=101):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据集: {data_path}")

        data = np.load(data_path, allow_pickle=True)
        images = data['images']  # (106, 100, 100, 3)
        poses = data['poses']   # (106, 4, 4)
        self.focal = float(data['focal'])
        self.H, self.W = images.shape[1:3]

        if mode == 'train':
            self.images = images[:test_idx]
            self.poses = poses[:test_idx]
        else:
            self.images = images[test_idx:]
            self.poses = poses[test_idx:]

        print(f"SDF Data [{mode}]: {self.H}x{self.W}, {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()
        pose = torch.from_numpy(self.poses[idx]).float()
        return image, pose
