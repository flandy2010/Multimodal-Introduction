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
使用mps进行训练，n_samples=64，iter=3000的时间消耗大概是30分钟：
```shell
python train.py --data_path ../data/tiny_nerf_data.npz --exp_dir ./runs --device mps
```
训练过程中使用PSNR作为衡量指标，iter=3000的情况下，PSNR=24.61，属于整体轮廓、颜色、大体形状清晰，但距离优秀还有第一定距离：

![iter_0](examples/record_demo_01/iter0_testpsnr7.22.png)
![iter_400](examples/record_demo_01/iter400_testpsnr19.15.png)
![iter_2000](examples/record_demo_01/iter2000_testpsnr23.39.png)
![iter_3000](examples/record_demo_01/iter3000_testpsnr24.61.png)

### 推理结果


# 踩坑记录
暂无

# 参考资料
1. [NeRF: Neural Radiance Fields](https://github.com/bmild/nerf)