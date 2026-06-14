# Multimodal-Introduction

### [2026-06-13] 扩散模型（Nano版）

- 实现记录：[Nano-DDPM](https://github.com/flandy2010/Multimodal-Introduction/blob/main/diffusion_model/README.md)
- 训练数据：MNIST手写数据集
- 训练效果：
![result](./diffusion_model/example/inference_visualize.png)

### [2026-06-14] 扩散模型 & CFG（Nano版）

- 实现记录：[Nano-DDPM-CFG](https://github.com/flandy2010/Multimodal-Introduction/blob/main/diffusion_model_CFG/README.md)
- 训练数据：MNIST手写数据集
- 训练效果：
![result](./diffusion_model_CFG/example/inference_visualize.png)

### [2026-06-14] Flow Match Model & CFG
- 实现记录：[Nano-Flow-Matching-CFG](https://github.com/flandy2010/Multimodal-Introduction/blob/main/flow_matching_model_CFG/README.md)
- 训练数据：MNIST手写数据集
- 训练效果：
![result](./flow_matching_model_CFG/example/inference_visualize.png)

### [TODO] DiT & CFG
基于Diffusion Transformer结构进行实验，带CFG但考虑到数据集分辨率较低暂时不使用VAE结构。
同时实现一个可以复用的训练框架，支持DiT，UNet等不同格式，支持DDPM和Flow Matching训练，
- 实现记录：[Nano-Flow-Matching-CFG](https://github.com/flandy2010/Multimodal-Introduction/blob/main/diffusion_transformer/README.md)

