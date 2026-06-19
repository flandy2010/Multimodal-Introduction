import argparse
from types import SimpleNamespace


def get_config():
    # 1. 定义全局默认配置
    cfg = SimpleNamespace()

    # --- 基础/环境配置 ---
    cfg.common = SimpleNamespace(
        device="cuda",
        seed=42,
        ckpt_dir="./ckpt",
        mode="train",  # train / sample
        num_workers=4,
    )

    # --- 推理配置 ---
    cfg.inference = SimpleNamespace(
        infer_mode="zip",
    )

    # --- 数据集配置 ---
    cfg.data = SimpleNamespace(
        root="../data",
        name="MNIST",
        img_size=28,
        img_shape=(1, 28, 28),
        in_channels=1,
        vocab_size=32,
    )

    cfg.video = SimpleNamespace(
        num_frames=16,
        frame_shape=(1, 28, 28),
        gif_size=(256, 256),
        fps=12,
    )

    # --- 训练/采样算法配置 ---
    cfg.method = SimpleNamespace(
        type="flow_matching",  # flow_matching / ddpm
        n_steps=1000,  # 训练步数 (对于 FM 是时间映射尺度)
        n_classes=11,
        sample_steps=20,  # 推理步数
        cfg_scale=4.0,  # CFG 引导强度
        batch_size=16,
        lr=1e-3,
        epochs=100,
        drop_rate=0.2  # 标签丢弃率
    )

    # --- UNet 模型配置 ---
    cfg.unet = SimpleNamespace(
        channels=[16, 32, 64, 128],
        pe_dim=128,  # 建议调大，10太小了
        residual=True
    )

    # --- DiT 模型配置 ---
    cfg.dit = SimpleNamespace(
        input_size=28,
        patch_size=2,
        in_channels=1,
        hidden_size=128,
        depth=6,
        num_heads=4
    )

    # --- 当前选择的模型 ---
    cfg.model_type = "unet"  # unet / dit

    return cfg


def update_config(cfg, args):
    """根据命令行参数更新配置"""
    # 更新通用参数
    if args.mode: cfg.common.mode = args.mode
    if args.method: cfg.method.type = args.method
    if args.model: cfg.model_type = args.model
    if args.lr: cfg.method.lr = args.lr
    if args.batch: cfg.method.batch_size = args.batch
    if args.s: cfg.method.cfg_scale = args.s

    if args.n_steps: cfg.method.n_steps = args.n_steps
    if args.num_frames: cfg.video.num_frames = args.num_frames
    if args.sample_steps: cfg.method.sample_steps = args.sample_steps

    if args.device: cfg.common.device = args.device

    if args.infer_mode: cfg.inference.infer_mode = args.infer_mode
    if args.infer_labels: cfg.inference.infer_labels = args.infer_labels
    if args.infer_scales: cfg.inference.infer_scales = args.infer_scales

    # 特殊处理ddpm下的min_beta和max_beta
    if args.min_beta: cfg.method.min_beta = args.min_beta
    if args.max_beta: cfg.method.max_beta = args.max_beta

    # 特殊处理 UNet 的 channels 列表
    if args.channels:
        cfg.unet.channels = args.channels

    if args.patch_size:
        cfg.dit.patch_size = args.patch_size

    return cfg