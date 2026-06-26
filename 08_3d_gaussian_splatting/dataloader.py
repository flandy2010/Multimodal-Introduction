import os
import torch
import struct
import numpy as np
from PIL import Image
from pathlib import Path


def qvec2rotmat(q):
    return np.array([
        [1 - 2 * q[2] ** 2 - 2 * q[3] ** 2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[1] * q[3] + 2 * q[0] * q[2]],
        [2 * q[1] * q[2] + 2 * q[0] * q[3], 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2, 2 * q[2] * q[3] - 2 * q[0] * q[1]],
        [2 * q[1] * q[3] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2]
    ])


class GSDataLoader:
    def __init__(self, data_path, factor=8):
        self.path = Path(data_path)
        self.factor = factor
        self.sparse_path = self.path / "sparse" / "0"

        # 1. 加载相机内参
        self.load_intrinsics()
        # 2. 加载相机外参
        self.load_extrinsics()
        # 3. 加载点云
        self.initial_points, self.initial_colors = self.load_points3d()
        # 4. 加载图片
        self.images = self.load_images()

        print(f"✅ 对齐完成 | 分辨率: {self.W}x{self.H} | 点数: {len(self.initial_points)}")

    def load_intrinsics(self):

        with open(self.sparse_path / "cameras.bin", "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            cam_id, model_id, width, height = struct.unpack("<IiQQ", f.read(24))
            model_n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}
            n_p = model_n_params.get(model_id, 4)
            params = struct.unpack(f"<{n_p}d", f.read(n_p * 8))
            self.focal = float(params[0] / self.factor)
            self.W, self.H = int(width / self.factor), int(height / self.factor)

            # K矩阵的含义是：将3D点(x, y, z)透视到z=1后得到(x/z, y/z, 1)，然后按焦距和宽高进行缩放和平移
            # 最终实现：从归一化相机平面到像素平面的仿射变换
            self.K = torch.tensor([
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1]
            ], dtype=torch.float32)

    def load_extrinsics(self):

        poses, img_names = [], []
        with open(self.sparse_path / "images.bin", "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                # image_id(I), q(4d), t(3d), cam_id(I) -> 4 + 32 + 24 + 4 = 64 字节
                data = struct.unpack("<IdddddddI", f.read(64))
                qvec, tvec = np.array(data[1:5]), np.array(data[5:8])
                # qvec：四元数（旋转）
                # tvec：世界坐标系到相机坐标系（平移）

                # 读取 null-terminated 文件名
                name = ""
                while True:
                    char = f.read(1).decode("utf-8")
                    if char == "\0": break
                    name += char
                img_names.append(name)

                # 跳过 2D 特征点
                num_points2d = struct.unpack("<Q", f.read(8))[0]
                f.read(num_points2d * 24)

                # W2C -> C2W (保持 COLMAP 坐标系: X右 Y下 Z前)
                # Camera-to-World矩阵，就是一台相机在世界坐标系中的“位置”和“朝向”。它将相机坐标系下的点，映射回世界坐标系中。
                R = qvec2rotmat(qvec)
                R_c2w = R.T
                T_c2w = -R_c2w @ tvec
                c2w = np.concatenate([R_c2w, T_c2w[:, None]], axis=1)
                poses.append(c2w)

        self.img_names = img_names
        poses = np.array(poses)
        # 中心化（下标为3的第4维存放了平移量）
        self.avg_center = np.mean(poses[:, :3, 3], axis=0)
        poses[:, :3, 3] -= self.avg_center
        # 缩放
        self.scene_scale = 1.0 / (np.max(np.abs(poses[:, :3, 3])) + 1e-5)
        poses[:, :3, 3] *= self.scene_scale

        # 最终所有相机都被约束在了3轴[-1, 1]的立方体里面
        self.poses = torch.from_numpy(poses).float()

    def load_points3d(self):
        bin_path = self.sparse_path / "points3D.bin"
        xyzs, rgbs = [], []
        with open(bin_path, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_points):
                # id(Q), xyz(3d), rgb(3B), error(d) -> 8 + 24 + 3 + 8 = 43 字节
                data = struct.unpack("<QdddBBBd", f.read(43))
                xyzs.append(data[1:4])
                rgbs.append(data[4:7])
                f.read(struct.unpack("<Q", f.read(8))[0] * 8)  # skip track

        xyzs = np.array(xyzs)
        # 不翻转，保持 COLMAP 坐标系，但需要和相机进行相同的缩放
        xyzs -= self.avg_center
        xyzs *= self.scene_scale

        xyzs = torch.from_numpy(xyzs).float()
        rgbs = torch.from_numpy(np.array(rgbs)).float() / 255.0

        # 过滤离群点：只保留在相机球体附近的点（半径基于相机分布），避免colmap导致的飞点
        # 相机归一化后 max abs = 1.0，保留 radius 倍范围内的点
        radius = 3.0
        point_norms = torch.norm(xyzs, dim=-1)
        inlier_mask = point_norms < radius
        xyzs = xyzs[inlier_mask]
        rgbs = rgbs[inlier_mask]

        # 限制初始点数，flowers 这种场景通常有 5-10 万个点，渲染太慢
        if len(xyzs) > 15000:
            idx = torch.randperm(len(xyzs))[:15000]
            xyzs, rgbs = xyzs[idx], rgbs[idx]
        return xyzs, rgbs

    def load_images(self):
        img_dir = self.path / f"images_{self.factor}"
        if not img_dir.exists(): img_dir = self.path / "images"
        imgs = []
        for name in self.img_names:
            img = Image.open(img_dir / name).convert("RGB")
            img = img.resize((self.W, self.H), Image.LANCZOS)
            imgs.append(np.array(img) / 255.0)
        return torch.from_numpy(np.array(imgs)).float()

    def get_view_params(self, index):
        image = self.images[index]
        c2w = torch.eye(4)
        c2w[:3, :4] = self.poses[index]
        camera_pos = c2w[:3, 3]  # 相机在世界坐标系的位置
        return image, c2w, torch.inverse(c2w), self.K, camera_pos

    def get_initial_pcd(self):
        return self.initial_points, self.initial_colors

    def get_normalization_params(self):
        # radius 应与点云过滤半径一致
        return {"radius": 3.0}


if __name__ == "__main__":

    loader = GSDataLoader(
        data_path="../data/360_extra_scenes/flowers",
        factor=8
    )