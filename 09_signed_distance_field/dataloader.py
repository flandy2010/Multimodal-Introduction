import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TinySDFDataset(Dataset):
    """
    SDF 专用数据加载器
    核心逻辑：
    1. 自动计算场景中心与缩放，将相机/物体归一化到半径为 1.0 的单位球内。
    2. 生成严格单位化的射线方向 (Unit-norm rays_d)。
    """

    def __init__(self, data_path="../data/tiny_nerf_data.npz", mode='train', test_idx=101):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据集: {data_path}")

        data = np.load(data_path, allow_pickle=True)
        images = data['images']  # (106, 100, 100, 3)
        poses = data['poses']  # (106, 4, 4)
        self.focal = float(data['focal'])
        self.H, self.W = images.shape[1:3]

        # --- 1. 全局坐标归一化 (SDF 训练的核心) ---
        # 计算所有相机的中心点
        cam_centers = poses[:, :3, 3]  # (106, 3)
        # 计算相机群的中心，作为世界坐标系原点
        self.center = np.mean(cam_centers, axis=0)
        # 计算相机到中心的最远距离，并将其缩放到约 1.0 左右，这样物体就被自然地包裹在 [-1, 1] 的球体内
        self.scale = 1.0 / np.linalg.norm(cam_centers - self.center, axis=1).max()

        # 记录归一化参数，用于后续推理渲染
        self.normalization_params = {'center': self.center, 'scale': self.scale}

        if mode == 'train':
            self.images = images[:test_idx]
            self.poses = poses[:test_idx]
        else:
            self.images = images[test_idx:]
            self.poses = poses[test_idx:]

        print(f"SDF Data [{mode}]: {self.H}x{self.W}, {len(self.images)} images")
        print(f"Normalization Applied: Center={self.center}, Scale={self.scale:.4f}")

    def get_rays(self, pose):
        """
        生成一幅图对应的所有射线
        输入: 归一化后的相机位姿 [4, 4]
        返回:
            rays_o: [H, W, 3], 射线起点
            rays_d: [H, W, 3], 严格归一化的射线方向
        """
        # 生成像素坐标网格
        i, j = torch.meshgrid(
            torch.linspace(0, self.W - 1, self.W),
            torch.linspace(0, self.H - 1, self.H),
            indexing='ij'
        )
        i, j = i.t(), j.t()  # [H, W]

        # 计算相机坐标系下的方向 (Standard Pinhole Model)
        # z轴朝内为 -1
        dirs = torch.stack([
            (i - self.W * 0.5) / self.focal,
            -(j - self.H * 0.5) / self.focal,
            -torch.ones_like(i)
        ], -1)  # [H, W, 3]

        # 将方向旋转到世界坐标系
        # rays_d = dirs_cam * R^T
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], -1)

        # --- 重要：SDF 必须执行方向归一化 ---
        # 确保每个 rays_d 的模长为 1.0
        rays_d = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8)

        # 射线原点即相机在世界坐标系的位置
        rays_o = pose[:3, 3].expand(rays_d.shape)  # [H, W, 3]

        return rays_o, rays_d

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        返回数据字典，包含送入模型的所有原材料
        """
        # 1. 图像归一化 [H, W, 3], range [0, 1]
        image = torch.from_numpy(self.images[idx]).float()

        # 2. 位姿归一化
        pose = self.poses[idx].copy()
        # 减中心，乘缩放：让相机分布在原点周围的球面上
        pose[:3, 3] = (pose[:3, 3] - self.center) * self.scale
        pose = torch.from_numpy(pose).float()

        # 3. 生成射线
        rays_o, rays_d = self.get_rays(pose)

        return {
            "image": image,  # [100, 100, 3], range [0, 1]
            "rays_o": rays_o,  # [100, 100, 3], 范围约 [-1.2, 1.2]
            "rays_d": rays_d,  # [100, 100, 3], 严格 unit-norm
            "pose": pose  # [4, 4]
        }


if __name__ == "__main__":

    # 测试代码
    dataset = TinySDFDataset(data_path="../data/tiny_nerf_data.npz", mode='train')
    sample = dataset[0]

    print("\n--- Shape 验证 ---")
    print(f"Image:  {sample['image'].shape}")  # [100, 100, 3]
    print(f"Rays_o: {sample['rays_o'].shape}")  # [100, 100, 3]
    print(f"Rays_d: {sample['rays_d'].shape}")  # [100, 100, 3]

    print("\n--- 取值范围验证 ---")
    ro_norm = torch.norm(sample['rays_o'], dim=-1).mean()
    rd_norm = torch.norm(sample['rays_d'], dim=-1).mean()

    # 验证光线方向是否归一化 (SDF 渲染死理)
    print(f"Rays_d Norm (Mean): {rd_norm:.4f} (期望: 1.0000)")
    # 验证相机是否在单位球附近
    print(f"Rays_o Distance:    {ro_norm:.4f} (期望: 1.0 附近)")
    # 验证图像像素
    print(f"Image Pixel Range:  [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")