import numpy as np
import torch
import os

class TinySDFLoader:
    def __init__(self, data_path="../data/tiny_nerf_data.npz"):
        data = np.load(data_path)
        self.images = torch.from_numpy(data['images']).float()
        self.poses = torch.from_numpy(data['poses']).float()
        self.focal = float(data['focal'])
        self.H, self.W = self.images.shape[1:3]
        print(f"✅ Data Loaded: {self.H}x{self.W}, Images: {len(self.images)}")

    def get_all(self):
        return self.images, self.poses, self.focal