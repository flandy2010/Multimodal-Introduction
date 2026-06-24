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
```shell
```

### 模型训练

### 模型推理

# 踩坑记录

### 渲染结果呈烟雾状
- 现象：iter=2000的时候渲染结果呈烟雾状，继续训练也并没有改进。
- 原因：距离到密度的转换函数有问题，同时采样点数不足
- 解决方案：参考volSFT和NeuS的alpha计算方式

![error](examples/error01_iter5000_psnr14.png)