# NeRF

本章主要用于记录NeRF的nano手搓版本，用于对NeRF有一个基本的认识。

# 环境依赖
```shell
conda create -n py312threeD python=3.12
conda activate py312threeD
pip install -r requirements.txt
```

# 使用介绍

### 数据下载
复用NeRF的训练集，放在公共路径下的`tiny_nerf_data.npz`。

### 模型训练
使用mps进行训练，n_samples=64，iter=2000的时间消耗大概是10分钟：
```shell
python train.py --data_path ../data/tiny_nerf_data.npz --exp_dir ./runs --device mps
```

# 踩坑记录

### 验证集psnr卡在22.5无法继续提升
现象：大概在iter=600的时候就提升到psnr=22了，但iter=2400的时候psnr也还在22.4-22.6徘徊。
渲染出来的结果呈现出严重的“颗粒感”或“噪点感”。虽然挖掘机的形状和颜色都对，但它看起来不像一个实心的物体，而像是一堆发光的“沙子”或者碎屑。
claude老师给出的原因如下：
- 缺乏训练扰动：这是最可能的原因。在训练时，如果每条射线上的采样点位置是固定的（比如总是等间距的 64 个点），模型会学会一种“作弊”方式：它只在那些固定的点上填入颜色，而点与点之间的空间是空的。
- 哈希冲突：哈希表设置得太小，不同的空间点会共用同一个特征。导致空间中莫名其妙地出现一些亮斑或暗点，也就是图里的细碎杂色。

![error_01](examples/error01_ter2450_testpsnr22.23.png)

改进方案为训练时候追加随机扰动：
```python
def render_rays(model, rays_o, rays_d, near, far, n_samples):
    # 1. 生成标准的等间距采样点 t_vals
    t_vals = torch.linspace(0., 1., steps=n_samples).to(rays_o.device)
    z_vals = near * (1.-t_vals) + far * t_vals
    z_vals = z_vals.expand(rays_o.shape[:-1] + (n_samples,))

    # --- 关键修改：增加随机扰动 (只在训练模式开启) ---
    if model.training:
        # 获取采样点之间的间距
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # 在区间内随机抖动
        t_rand = torch.rand(z_vals.shape).to(rays_o.device)
        z_vals = lower + (upper - lower) * t_rand
    # ----------------------------------------------

    # 剩下的逻辑不变...
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    # ... 进行模型查询和体渲染积分 ...
```

### 验证集psnr=9.39并不随iter变化

现象：模型在iter=0的时候会输出全黑的图片，psnr=9.39，但进行1000次迭代后（以及迭代过程中）模型都会输出全黑的图片，psnr维持9.39不变。

