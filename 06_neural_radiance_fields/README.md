# NeRF

本章主要用于记录NeRF的nano手搓版本，用于对NeRF有一个基本的认识。

# 环境依赖
```shell
conda create -n py312threeD python=3.12
conda activate py312threeD
pip install -r requirements.txt
```

# 数据下载
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


# 参考资料
1. [NeRF: Neural Radiance Fields](https://github.com/bmild/nerf)