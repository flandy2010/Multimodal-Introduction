# 3D Gaussian Splatting

本章主要用于记录3DGS的nano手搓版本，用于对3DGS有一个基本的认识。

# 环境依赖
```shell
conda create -n py312threeD python=3.12
conda activate py312threeD
pip install -r requirements.txt
```

# 使用介绍

### 数据下载
- 复用NeRF的训练集，放在公共路径下的`tiny_nerf_data.npz`。但发现从零训练特别难。
- 下载`360_extra_scenes`数据集，放在公共路径下，包含`flowers`和`treehill`两个场景。
```shell
cd ../data && mkdir 360_extra_scenes
cd 360_extra_scenes
wget https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip
unzip 360_extra_scenes.zip
```

### 模型训练
使用H20进行进行训练，
```shell
# bash脚本
bash train_h20.sh

# python命令
DATA_PATH="../data/360_extra_scenes/flowers"
EXP_DIR="./runs/h20_v4"

python train.py \
    --data_path $DATA_PATH \
    --exp_dir $EXP_DIR \
    --factor 4 \
    --num_points 50000 \
    --n_iters 30000 \
    --sh_degree 3 \
    --tile_size 64 \
    --grad_threshold 0.0005 \
    --display_int 250 \
    --device cuda
```

### 模型推理

# 踩坑记录

### 推理结果是一团光晕
- 现象：模型输出的结果是好几团非常大的光晕，没有任何形状或轮廓
- 原因：发现是椭球初始化的太大了
- 解决方案：缩小椭球初始化时候的大小

### 推理结果无轮廓
- 现象：有大致的形状但没有清晰的轮廓
- 原因：初步怀疑是Nano版本中没有加椭球数量的自适应，然后point=3000 + 椭球初始化的比较小导致不太够用。
- 解决方案：暂未解决

![error](examples/error01_iter10000_p3000.png)

### 椭球迅速变大
- 现象：从训练开始，椭球就快速变大导致图像轮廓逐渐不清晰了
- 原因：模型发现"把点放大→覆盖更多像素→MSE 快速降低"比"精细调整颜色和位置"更高效，所以所有点都在膨胀。
- 解决方案：尝试降低scales的学习率，同时增强apply_constraints中对于scale上限的约束（0.05 -> 0.03)

![error](examples/error02/step_0000.png)
![error](examples/error02/step_0010.png)
![error](examples/error02/step_0020.png)

# 吐槽
对于这个数据集来说，3DGS真的比Instant-NGP难训太多了：
- 如果高斯点不在正确的位置，颜色就毫无意义；如果颜色不对，产生的梯度就会把位置带偏。
- 在随机初始化时，个参数（Means, Scale, Quat, Opacity, Color）都在同时乱跳，这种高维度震荡极难收敛。

