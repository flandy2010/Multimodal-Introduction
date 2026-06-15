# Video Generation based on DiT
该项目主要用于进阶到视频生成领域：
- 构建视频：为对MNIST手写数据集进行缩放，翻转，旋转等操作来构成一个3-5秒的视频。
- 构建控制文本：搭配模板来生成对应的文本描述。考虑到token有限，直接按char切分，然后从零初始化一个embedding。

# 环境依赖
```shell
conda create -n py312DDPM python=3.12
conda activate py312DDPM
pip install -r requirements.txt
```

# 使用介绍

### 数据构建
使用video_generator可以根据给定的图片&指令生成对应的视频，提供了两种可视化方式：
- 输出成.mp4格式的视频
- 输出成.png格式的胶片序列图
```python
python video_generator.py
```
生成的胶片序列图样例如下：
- (上图) 对数字7进行水平翻转
- (下图) 对数字5旋转60度

<p align="center">
  <img src="./examples/数字7_垂直翻转_预览.png" width="300" title="对数字7进行水平翻转">
</p>
<p align="center">
  <img src="./examples/数字5_旋转60度_预览.png" width="300" title="对数字5旋转60度">
</p>

### 模型训练

### 模型推理

# 输出效果