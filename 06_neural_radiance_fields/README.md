# NeRF

本章主要用于记录NeRF的nano手搓版本，用于对NeRF有一个基本的认识。

# 环境依赖
```shell
conda create -n py312threeD python=3.12
conda activate py312threeD
pip install -r requirements.txt
```

# 使用介绍

###  数据下载
```shell
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
# 备份链接
# https://github.com/houchenst/FastNeRF/blob/master/tiny_nerf_data.npz.1
```
数据下载后放在公共文件夹`data`下面，可以使用dataloader.py进行可视化：
```shell
python dataloader.py
```
- 左边4张图表示了4个视角下的真实结果
- 右边散点图表示了106个摄像点
![examlpe](./examples/lego_digger.png)

### 模型训练
- 使用mps进行训练，n_samples=64，iter=3000的时间消耗大概是30分钟
- 使用h20进行训练，n_samples=128，iter=10000的时间消耗大概是80分钟
```shell
# MPS训练命令
python train.py --data_path ../data/tiny_nerf_data.npz --exp_dir ./runs --device mps

# H20训练命令
python train.py --data_path ../data/tiny_nerf_data.npz --exp_dir ./runs/h20_demo --device cuda --n_iters 10000 --n_samples 128
```
训练过程中使用PSNR作为衡量指标，MPS环境iter=3000的情况下，PSNR=24.61，属于整体轮廓、颜色、大体形状清晰，但距离优秀还有第一定距离：

![iter_0](examples/record_demo_01/iter0_testpsnr7.22.png)
![iter_400](examples/record_demo_01/iter400_testpsnr19.15.png)
![iter_2000](examples/record_demo_01/iter2000_testpsnr23.39.png)
![iter_3000](examples/record_demo_01/iter3000_testpsnr24.61.png)

在H20环境iter=10000的情况下，PSNR可以达到26，但积木的颗粒感没有完全展现：
![iter_10000](examples/record_demo_02.png)

# 踩坑记录

### 输出图像缺失细节
- 现象：模型在iter=10000的情况下，误差热力图显示边缘仍然存在较多的亮点
- 原因：“连续的神经网络”难以拟合“离散的马赛克边缘”
- 解决方案：尝试扩大模型参数规模，缩小采样范围，加入分层采样

![iter_10000](examples/record_demo_02.png)

### 模型难以突破PSNR=26
- 现象：模型在iter=1400的时候PSNR达到了24，直到但iter=4000一直在24-25.9之间横跳
- 原因：
  - 部分原因是LR过高，同时也因为没有加任何的训练Trick（比如各类正则化）
  - 加了正则化又跑了一次还是不对，细看的话主要问题出在没有积木的颗粒感
- 解决方案：尝试各类正则化（对于像素的坐标抖动，对于光线采样点的随机抖动，对于sigma加随机noise）


# 参考资料
1. [NeRF: Neural Radiance Fields](https://github.com/bmild/nerf)