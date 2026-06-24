# SDF

本章主要用于记录SDF的nano手搓版本，用于对SDF有一个基本的认识。

# 环境依赖
```shell
conda create -n py312threeD python=3.12
conda activate py312threeD
pip install -r requirements.txt
```

# 使用介绍

###  数据下载
使用仅2D图片，预测d->sigma的方式进行训练，复用NeRF的训练集，放在公共路径下的`tiny_nerf_data.npz`。

### 模型训练
```shell
# shell脚本
bash train.sh

# python命令
python train.py \
    --init_radius 1.0 \
    --s_val 400.0 \
    --n_samples 128 \
    --n_iters 20000 \
    --display_int 100 \
    --exp_dir ./runs/demo02 \
    --device mps
```

### 模型推理

# 踩坑记录

### 渲染结果呈烟雾状
- 现象：iter=2000的时候渲染结果呈烟雾状，继续训练也并没有改进。
- 原因：距离到密度的转换函数有问题，同时采样点数不足
- 解决方案：参考volSFT和NeuS的alpha计算方式

![error](examples/error01_iter5000_psnr14.png)

### PSNR较高但边缘不清晰
- 现象：iter=12000的时候，PSNR达到28，但z=0的边界切片非常不规整。图片背景（红色区域）布满了明显的方块状纹理，像是在一张有格子的纸上画画。
- 原因：模型学会了去贴上颜色，但并没有真正学到几何结构。结合训练日志判断发现：
  - iter=12000的时候，Eikonal Loss仍维持在33-35，说明模型完全没有遵守SDF的基本物理原则
  - s_val从5增加到了400，但在SDF场还没学好的情况下加锐化导致模型摆烂
- 解决方案：加大Eikonal Loss的权重，约束s_val的增长速度，增加采样范围

![error](examples/error02_iter12000_psnr28.png)
<p align="center">
  <img src="examples/error02_iter12000_sdf_slice.png" alt="error">
</p>