import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset, DataLoader


class TinyNeRFDataset(Dataset):
    def __init__(self,
                 data_path="../data/tiny_nerf_data.npz",
                 mode='train',
                 test_idx=101):

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据集文件: {data_path}")

        # 1. 加载数据
        data = np.load(data_path, allow_pickle=True)
        images = data['images']  # (106, 100, 100, 3)
        poses = data['poses']  # (106, 4, 4)

        # 106表示围绕这个3D模型拍了106张照片
        # poses[index]表示下表为index图片的视角，具体内容是4 x 4的矩阵
        # 矩阵内部结构图解:
        # [
        #   [ R11, R12, R13,  Tx ],  <-- R 为旋转矩阵 (Rotation)，定义相机朝向
        #   [ R21, R22, R23,  Ty ],  <-- T 为平移向量 (Translation)，定义相机位置
        #   [ R31, R32, R33,  Tz ],
        #   [  0,   0,   0,   1  ]   <-- 齐次坐标占位符
        # ]
        #
        # 列向量的具体几何意义：3个轴的方向（3个向量）+ 1个中心坐标
        # 第 0 列 ([:3, 0]): 相机坐标系的 X 轴 (Right 向量) 在世界坐标系中的方向
        # 第 1 列 ([:3, 1]): 相机坐标系的 Y 轴 (Up 向量) 在世界坐标系中的方向
        # 第 2 列 ([:3, 2]): 相机坐标系的 Z 轴 (Back 向量) 在世界坐标系中的方向
        # 第 3 列 ([:3, 3]): 相机中心在世界坐标系中的 (x, y, z) 坐标

        self.focal = float(data['focal'])

        self.H, self.W = images.shape[1:3]

        # 2. 划分数据集 (原数据集共106张图)
        # 通常前100张用于训练，最后几张用于测试
        if mode == 'train':
            self.images = images[:test_idx]
            self.poses = poses[:test_idx]
        else:
            self.images = images[test_idx:]
            self.poses = poses[test_idx:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()
        pose = torch.from_numpy(self.poses[idx]).float()
        return image, pose


def visualize_dataset(dataset):
    """
    优化版可视化：更清晰的布局，防止 3D 图形遮挡内容
    """
    fig = plt.figure(figsize=(18, 7))
    # 创建左侧 2x2 展示图片，右侧展示 3D 空间
    gs = gridspec.GridSpec(2, 4, figure=fig)

    # --- 1. 可视化图片 (左侧 4 张) ---
    for i in range(4):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.imshow(dataset.images[i * 20])  # 每隔20张抽一张，看不同角度
        ax.set_title(f"View Index: {i * 20}", fontsize=10)
        ax.axis('off')

    # --- 2. 可视化 3D 相机位姿 (右侧大图) ---
    # 使用 gs[:, 2:] 让 3D 图占据右半部分
    ax_3d = fig.add_subplot(gs[:, 2:], projection='3d')

    poses = dataset.poses
    centers = poses[:, :3, 3]  # 提取所有相机中心

    # 绘制相机中心 (减小 size, 增加 alpha)
    ax_3d.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                  color='crimson', s=10, alpha=0.6, label='Camera Centers')

    # 绘制相机朝向箭头
    # 我们只绘制一部分箭头，避免视觉干扰
    step = 8
    for i in range(0, len(poses), step):
        # 旋转矩阵的第三列是相机坐标系的 Z 轴 (看向前方)
        # 在 c2w 矩阵中，该列定义了相机在世界空间中的指向
        direction = poses[i, :3, 2]
        ax_3d.quiver(centers[i, 0], centers[i, 1], centers[i, 2],
                     -direction[0], -direction[1], -direction[2],
                     color='royalblue', length=0.3, normalize=True, alpha=0.4)

    # 绘制坐标原点 (Lego 挖掘机所在的位置)
    ax_3d.scatter([0], [0], [0], color='black', marker='x', s=50, label='Object Center (0,0,0)')

    # 设置轴标签
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')

    # 关键：设置 3D 轴等比例，防止图像变形
    # 在某些 matplotlib 版本中可用 ax_3d.set_box_aspect((1,1,1))
    max_range = np.array([centers[:, 0].max() - centers[:, 0].min(),
                          centers[:, 1].max() - centers[:, 1].min(),
                          centers[:, 2].max() - centers[:, 2].min()]).max() / 2.0
    mid_x = (centers[:, 0].max() + centers[:, 0].min()) * 0.5
    mid_y = (centers[:, 1].max() + centers[:, 1].min()) * 0.5
    mid_z = (centers[:, 2].max() + centers[:, 2].min()) * 0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    ax_3d.set_title("World Space: Camera Distribution", fontsize=12)
    ax_3d.legend(loc='lower right', fontsize=8)

    # 调整布局，防止重叠
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    try:
        dataset = TinyNeRFDataset(data_path="../data/tiny_nerf_data.npz")
        print(f"数据加载成功！")
        print(f"训练样本数: {len(dataset)}")
        print(f"图片尺寸: {dataset.H}x{dataset.W}")
        print(f"焦距 (Focal): {dataset.focal}")

        # 调用可视化函数
        visualize_dataset(dataset)

    except Exception as e:
        print(f"运行失败: {e}")