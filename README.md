# Multimodal-Introduction

### [2026-06-13] 扩散模型（Nano版）

- 实现记录：[Nano-DDPM](https://github.com/flandy2010/Multimodal-Introduction/blob/main/diffusion_model/README.md)
- 训练数据：MNIST手写数据集
- 训练效果：
![result](01_diffusion_model/example/inference_visualize.png)

### [2026-06-14] 扩散模型 & CFG（Nano版）

- 实现记录：[Nano-DDPM-CFG](https://github.com/flandy2010/Multimodal-Introduction/blob/main/diffusion_model_CFG/README.md)
- 训练数据：MNIST手写数据集
- 训练效果：
![result](02_guided_diffusion_model/example/inference_visualize.png)

### [2026-06-14] Flow Match Model & CFG
- 实现记录：[Nano-Flow-Matching-CFG](https://github.com/flandy2010/Multimodal-Introduction/blob/main/flow_matching_model_CFG/README.md)
- 训练数据：MNIST手写数据集
- 训练效果：
![result](03_guided_flow_matching/example/inference_visualize.png)

### [2026-06-15] DiT & CFG
基于Diffusion Transformer结构进行实验，带CFG但考虑到数据集分辨率较低暂时不使用VAE结构。
同时实现一个可以复用的训练框架，支持DiT，UNet等不同格式，支持DDPM和Flow Matching训练，
- 实现记录：[Nano-Flow-Matching-CFG](https://github.com/flandy2010/Multimodal-Introduction/blob/main/04_diffusion_transformer/README.md)
- 训练数据：MNIST手写数据集
- 训练效果：DiT + FlowMatching + sample_step=50
![result](04_diffusion_transformer/examples/dit_fm_step50.png)

### [TODO] Vedio Generation
基于Diffusion Transformer结构进行实验，带CFG但考虑到数据集分辨率较低暂时不使用VAE结构。使用工程化方案基于现有的图片进行视频生成：
- 原始素材：MNIST手写数据集
- 加工方式：图片缩放，图片翻转，图片旋转
- 控制信息：通过模版方式生成如：“生成一张(数字1)(上下翻转)的视频片段”

尝试时常3-5秒，分辨率28x28的视频生成。