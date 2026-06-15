import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_image_shape(dataset_name="MNIST"):
    if dataset_name == "MNIST":
        return (1, 28, 28)
    elif dataset_name == "CIFAR10":
        return (3, 32, 32)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataloader(cfg):
    """
    根据配置对象 cfg 获取数据加载器
    """
    # 生成模型通常建议将像素归一化到 [-1, 1] 空间
    # 公式：(pixel - 0.5) / 0.5 -> [0, 1] 映射到 [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if cfg.data.name == "MNIST":
        dataset = datasets.MNIST(
            root='../data',
            train=(cfg.common.mode == "train"),
            download=True,
            transform=transform
        )
    else:
        raise NotImplementedError("Currently only MNIST is supported.")

    loader = DataLoader(
        dataset,
        batch_size=cfg.method.batch_size,
        shuffle=True,
        num_workers=2
    )
    return loader


def denormalize(tensor):
    """将 [-1, 1] 的 Tensor 转回 [0, 1] 用于显示"""
    return (tensor * 0.5) + 0.5


# --- 4. 图像显示辅助函数 ---
def show_images(images, labels, num_rows=2, num_cols=5, title="Samples"):
    """
    images: shape [batch, channels, h, w]
    """
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    fig.suptitle(title, fontsize=16)

    # 确保 images 在 CPU 上
    images = images.cpu()

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # 1. 逆归一化
            img = denormalize(images[i])
            # 2. 转换维度 [C, H, W] -> [H, W, C]
            img = img.permute(1, 2, 0).squeeze().numpy()

            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.set_title(f"L: {labels[i].item()}", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# --- 测试代码 ---
if __name__ == '__main__':

    from types import SimpleNamespace

    mock_cfg = SimpleNamespace(
        data=SimpleNamespace(name="MNIST"),
        common=SimpleNamespace(mode="train"),
        method=SimpleNamespace(batch_size=10)
    )

    loader = get_dataloader(mock_cfg)
    images, labels = next(iter(loader))

    print(f"Dataset Shape: {get_image_shape('MNIST')}")
    print(f"Batch Shape: {images.shape}")
    print(f"Value Range: [{images.min():.2f}, {images.max():.2f}]")

    show_images(images, labels, num_rows=2, num_cols=5)