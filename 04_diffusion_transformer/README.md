# Diffusion Model

本章主要用于搭建一个简易版本的训练框架：
- 在模型选型上，支持UNet，Diffusion Transformer模型
- 在训练方式上，支持DDPM和Flow Matching

模型上增加对于Diffusion Transformer的支持，但暂时不开启VAE，默认开启CFG

# 环境依赖
```shell
conda create -n py312DDPM python=3.12
conda activate py312DDPM
pip install -r requirements.txt
```

# 项目结构
预期项目结构如下：
```text
04_diffusion_transformer/
├── configs.py           # 核心：使用 Namespace 或字典存储所有超参数
├── main.py              # 入口：负责训练和推理的调度
├── core/
│   ├── __init__.py
│   ├── base_engine.py   # 定义采样和训练的抽象接口
│   ├── ddpm_engine.py   # 扩散模型逻辑
│   └── fm_engine.py     # 流匹配逻辑
├── models/
│   ├── __init__.py
│   ├── unet.py          # 经典的 U-Net
│   └── dit.py           # Diffusion Transformer
├── utils/
│   ├── data.py          # 数据加载
│   └── logger.py        # 进度条和可视化
└── runs/                # 训练记录
```

# 使用介绍

### 模型训练
```shell
# 使用DDPM + UNet
python main.py --mode train --method ddpm --model unet --lr 0.0002 --batch 128 --channels 64 128 256 512 --device mps

# 使用DDPM + DiT

# 使用FlowMatching + DiT
```

### 模型推理
通过指定训练记录文件夹，自动读入配置，并根据输入的测试数据进行推理。
```shell
python main.py --mode sample --exp_dir runs/{model_name} --infer_labels 0 1 2 --infer_scales 1.0 4.0 10.0
```

# 效果展示

# 踩坑记录