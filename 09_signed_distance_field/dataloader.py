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

        # 预计算并缓存所有帧的图像 tensor、位姿、射线
        # 避免 __getitem__ 每次重复 get_rays（CPU meshgrid+矩阵乘法，约 1.6s/次）
        self._cache = self._precompute_all()

        print(f"SDF Data [{mode}]: {self.H}x{self.W}, {len(self.images)} images")
        print(f"Normalization Applied: Center={self.center}, Scale={self.scale:.4f}")

    def _precompute_all(self):
        """一次性预计算所有帧的 image/pose/rays_o/rays_d，缓存为 tensor 列表"""
        # 像素方向（相机坐标系），与位姿无关，只算一次
        i, j = torch.meshgrid(
            torch.linspace(0, self.W - 1, self.W),
            torch.linspace(0, self.H - 1, self.H),
            indexing='ij'
        )
        i, j = i.t(), j.t()  # [H, W]
        dirs_cam = torch.stack([
            (i - self.W * 0.5) / self.focal,
            -(j - self.H * 0.5) / self.focal,
            -torch.ones_like(i)
        ], -1)  # [H, W, 3]

        cache = []
        for idx in range(len(self.images)):
            image = torch.from_numpy(self.images[idx]).float()

            pose = self.poses[idx].copy()
            pose[:3, 3] = (pose[:3, 3] - self.center) * self.scale
            pose = torch.from_numpy(pose).float()

            # 将方向旋转到世界坐标系并归一化
            rays_d = torch.sum(dirs_cam[..., None, :] * pose[:3, :3], -1)
            rays_d = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8)
            rays_o = pose[:3, 3].expand(rays_d.shape).contiguous()

            cache.append({
                "image": image,
                "rays_o": rays_o,
                "rays_d": rays_d,
                "pose": pose,
            })
        return cache

    def gen_random_rays(self, img_idx, batch_size, device='cpu'):
        """
        从指定图像随机采样 batch_size 条射线（参考 NeuS 官方 gen_random_rays_at）。
        训练时每步调用，远比全图渲染（H×W 条射线）快。

        返回:
            rays_o:  [B, 3] 射线起点（已归一化坐标系）
            rays_d:  [B, 3] 归一化射线方向
            colors:  [B, 3] 对应像素 GT 颜色 [0, 1]
        """
        cached = self._cache[img_idx]
        H, W = self.H, self.W

        # 随机采样像素坐标
        px = torch.randint(0, W, (batch_size,))
        py = torch.randint(0, H, (batch_size,))

        rays_o = cached["rays_o"][py, px]   # [B, 3]
        rays_d = cached["rays_d"][py, px]   # [B, 3]
        colors = cached["image"][py, px]    # [B, 3]

        return rays_o.to(device), rays_d.to(device), colors.to(device)

    def get_rays(self, pose):
        """
        生成一幅图对应的所有射线（保留接口，供外部单独调用）
        输入: 归一化后的相机位姿 [4, 4]
        返回:
            rays_o: [H, W, 3], 射线起点
            rays_d: [H, W, 3], 严格归一化的射线方向
        """
        i, j = torch.meshgrid(
            torch.linspace(0, self.W - 1, self.W),
            torch.linspace(0, self.H - 1, self.H),
            indexing='ij'
        )
        i, j = i.t(), j.t()
        dirs = torch.stack([
            (i - self.W * 0.5) / self.focal,
            -(j - self.H * 0.5) / self.focal,
            -torch.ones_like(i)
        ], -1)
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], -1)
        rays_d = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8)
        rays_o = pose[:3, 3].expand(rays_d.shape)
        return rays_o, rays_d

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """直接返回预计算缓存，无任何重复计算"""
        return self._cache[idx]


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