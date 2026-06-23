# Multimodal-Introduction

### [2026-06-13] 扩散模型（Nano版）

- 实现记录：[Nano-DDPM](https://github.com/flandy2010/Multimodal-Introduction/blob/main/01_diffusion_model/README.md)
- 训练数据：MNIST手写数据集
- 训练效果：

![result](01_diffusion_model/example/inference_visualize.png)

### [2026-06-14] 扩散模型 & CFG（Nano版）

- 实现记录：[Nano-DDPM-CFG](https://github.com/flandy2010/Multimodal-Introduction/blob/main/02_guided_diffusion_model/README.md)
- 训练数据：MNIST手写数据集
- 训练效果：

![result](02_guided_diffusion_model/example/inference_visualize.png)

### [2026-06-14] Flow Match Model & CFG
- 实现记录：[Nano-Flow-Matching-CFG](https://github.com/flandy2010/Multimodal-Introduction/blob/main/03_guided_flow_matching/README.md)
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

### [2026-06-19] Video Generation
基于Diffusion Transformer结构进行实验，带CFG但考虑到数据集分辨率较低暂时不使用VAE结构。使用工程化方案基于现有的图片进行视频生成：
- 实现记录：[Nano-Flow-Matching-CFG](https://github.com/flandy2010/Multimodal-Introduction/blob/main/05_video_generation/README.md)
  - 原始素材：MNIST手写数据集
  - 加工方式：图片缩放，图片翻转，图片旋转
  - 控制信息：通过模版方式生成如：“生成一张(数字1)(上下翻转)的视频片段”
- 训练效果：DiT + FlowMatching + sample_step=100 + n_frames=16
- 吐槽：本来想尝试时长3-5秒，分辨率28x28的视频生成。实际跑下来发现勉强能训的动16帧的生成

![result](05_video_generation/examples/record_demo_02/ret_将0水平翻转.gif)
![result](05_video_generation/examples/record_demo_02/ret_将1垂直翻转.gif)
![result](05_video_generation/examples/record_demo_02/ret_将7缩小2倍.gif)
![result](05_video_generation/examples/record_demo_02/ret_将4旋转120度.gif)

### [TODO] Neural Radiance Fields
步入3D建模内容，先练习一下经典的神经辐射场（Neural Radiance Fields）。
- 实现记录：[Neural-Radiance-Fields](https://github.com/flandy2010/Multimodal-Introduction/blob/main/06_neural_radiance_fields/README.md)
- 训练数据：tiny_nerf_data
- 训练效果：n_samples=64 + iter=3000

![result](06_neural_radiance_fields/examples/record_demo_01/iter3000_testpsnr24.61.png)

### [2026-06-22] Instant Neural Graphics Primitives
NeRF的改进版本，将`(x, y, z)`对应的特征内容从使用神经网络记忆变成查哈希表。仅使用神经网络对于特征进行处理，生成颜色和密度。
- 实现记录：[Instant-NGP](https://github.com/flandy2010/Multimodal-Introduction/blob/main/07_instant_neural_graphics_primitives/README.md)
- 训练数据：tiny_nerf_data
- 训练效果：n_samples=192 + iter=10000

<p align="center">
  <img src="07_instant_neural_graphics_primitives/examples/demo02_iter9500_rotation_1.gif" alt="result">
</p>


![result](07_instant_neural_graphics_primitives/examples/demo02_iter9500_comparison_4.png)


### [TODO] 3D Gaussian Splatting
主动生成版本，以粒子而非光线作为主体，让作为粒子的椭圆球体主动发光并投影到2D画布上，叠加后得到最终的颜色。
- 实现记录：[3DGS](https://github.com/flandy2010/Multimodal-Introduction/blob/main/08_3d_gaussian_splatting/README.md)
- 训练数据：tiny_nerf_data
- 训练效果：