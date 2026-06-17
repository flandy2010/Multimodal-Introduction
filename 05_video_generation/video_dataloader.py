import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from types import SimpleNamespace # 用于单测


# 假设你已经把 VideoTransformEngine 放在同目录的 video_generator.py 中
try:
    from .video_generator import VideoTransformEngine
except ImportError:
    # 兼容直接运行此文件作为单测
    from video_generator import VideoTransformEngine


class CharTokenizer:
    def __init__(self):
        # 基础词表：数字 + 中文关键字 + 符号
        self.chars = ["<PAD>"]
        for char in "将0123456789放大缩小旋转倍度水平垂直翻":
            self.chars.append(char)
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(self.chars)
        self.pad_id = self.char_to_id["<PAD>"]

    def encode(self, text, max_len=10):
        ids = [self.char_to_id.get(c, self.pad_id) for c in text]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids):
        # 兼容 Tensor 或 List
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join([self.id_to_char.get(i, "") for i in ids if i != self.pad_id])

    def encode_batch(self, texts, max_len=10, return_tensor=True):
        token_ids = [self.encode(text, max_len) for text in texts]
        if return_tensor:
            token_ids = torch.LongTensor(token_ids)
        return token_ids

class MNISTVideoDataset(Dataset):

    def __init__(self, mnist_dataset, num_frames=16, tokenizer=None):
        self.base_ds = mnist_dataset
        self.num_frames = num_frames
        self.tokenizer = tokenizer or CharTokenizer()
        # 内部实例化转换引擎
        self.engine = VideoTransformEngine()
        self.inst_templates = ["放大", "缩小", "旋转", "水平翻转", "垂直翻转"]
        # self.inst_templates = ["垂直翻转"]

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]

        # 1. 随机挑选一个指令并生成完整文本
        inst_type = random.choice(self.inst_templates)
        inst_text = inst_type

        if inst_type == "旋转":
            param = random.randint(30, 180)
            inst_text = f"旋转{param}度"
        elif inst_type == "放大":
            param = random.randint(2, 4)
            inst_text = f"放大{param}倍"
        elif inst_type == "缩小":
            param = random.randint(2, 4)
            inst_text = f"缩小{param}倍"

        # 2. 调用引擎：注意这里的方法名要与 video_generator.py 里的 transform_to_tensor 一致
        # 并且传入的是完整字符串 inst_text，让引擎内部去正则解析
        video_tensor = self.engine.transform_to_tensor(img, inst_text, num_frames=self.num_frames)

        # 3. 指令转 ID
        inst_text = f"将{label}{inst_text}"
        inst_ids = torch.LongTensor(self.tokenizer.encode(inst_text))

        return {
            "video": video_tensor,   # [F, 1, 28, 28]
            "inst_ids": inst_ids,    # [L]
            "inst_text": inst_text,
            "label": label
        }

def get_video_dataloader(cfg):

    transform = transforms.Compose([
        transforms.ToTensor(),
        # 0.1307 是均值，0.3081 是标准差
        transforms.Normalize((0.1307,), (0.3081,))
        # transforms.Normalize((0.5,), (0.5,))
    ])

    # 建议此处路径由 cfg 提供
    mnist_base = datasets.MNIST(
        root=cfg.data.root,
        train=(cfg.common.mode == "train"),
        download=True,
        transform=transform
    )

    video_ds = MNISTVideoDataset(
        mnist_base,
        num_frames=cfg.video.num_frames,
        tokenizer=CharTokenizer()
    )

    loader = DataLoader(
        video_ds,
        batch_size=cfg.method.batch_size,
        shuffle=True,
        num_workers=cfg.common.num_workers
    )
    return loader


# --- 单测代码 ---
if __name__ == "__main__":

    from types import SimpleNamespace

    print("开始运行数据加载单测...")

    cfg = SimpleNamespace(
        common=SimpleNamespace(mode="train", num_workers=0),
        data=SimpleNamespace(root="../data"),
        method=SimpleNamespace(batch_size=10),
        video=SimpleNamespace(num_frames=2)
    )

    try:
        dataloader = get_video_dataloader(cfg)
        batch = next(iter(dataloader))

        videos = batch["video"]  # [Batch, Frame, C=1, H=28, W=28]
        inst_ids = batch["inst_ids"]  # [Batch, L=10]
        labels = batch["label"]  # [Batch]
        texts = batch["inst_text"]  # List of strings

        print(f"Batch={cfg.method.batch_size}, num_frames={cfg.video.num_frames}")
        print(f"videos.shape = {videos.shape}")
        print(f"inst_ids.shape = {inst_ids.shape}")
        print(f"labels.shape = {labels.shape}")

        engine = VideoTransformEngine()
        save_path = engine.save_to_grid_image(videos[0], inst_text=texts[0], save_dir="./examples", cell_size=(28, 28))
        print(f"save result of \'{texts[0]}\' to: {save_path}")

    except Exception as e:
        print(f"❌ 单测失败: {e}")
        import traceback

        traceback.print_exc()