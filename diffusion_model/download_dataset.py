import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_image_shape():

    # MNIST手写数据识别的shape为：[1, 28, 28]
    shape = (1, 28, 28)
    return shape


def show_images(images, labels, num_rows=2, num_cols=5):

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # 将 Tensor 转回 Numpy
            img = images[i].squeeze().numpy()
            ax.set_title(f"Label: {labels[i]}", fontsize=10)
            ax.imshow(img, cmap='gray')
            ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # 1. 定义预处理：将图片转为 Tensor，并进行标准化
    # MNIST 图片是 28x28 的灰度图
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的全局均值和标准差
    ])

    # 2. 下载训练集
    train_dataset = datasets.MNIST(
        root='../data',      # 存放目录
        train=True,         # 下载训练集
        download=True,      # 如果本地没有则下载
        transform=transform
    )

    # 3. 下载测试集
    test_dataset = datasets.MNIST(
        root='../data',
        train=False,
        transform=transform
    )

    # 4. 使用 DataLoader 加载数据（用于训练循环）
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 看看数据长什么样
    images, labels = next(iter(train_loader))

    print(f"图像的 Shape: {images.shape}")  # torch.Size([64, 1, 28, 28])
    print(f"标签示例: {labels[:10]}")

    show_images(images=images[:10], labels=labels[:10])