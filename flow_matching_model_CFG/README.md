# Flow Matching Model

本章在扩散模型&CFG的nano手搓版本基础上，通过修改训练目标和推理代码，将其改成flow matching model。


# 环境依赖
```shell
conda create -n py312DDPM python=3.12
conda activate py312DDPM
pip install -r requirements.txt
```

# 项目结构
- ddpm.py：用于给定原始图片，时间t计算线性插值。同时也用于从随机噪声出发进行图片生成
- model.py: 神经网络结构，同于给定原始图片和时间t，预测速度场vector
- train.py: 训练代码
- inference.py: 推理代码

```text
├── README.md
├── data
│   └── MNIST
└── diffusion_model
    ├── README.md
    ├── ckpt
    │   └── checkpoint_epoch_100.pth
    ├── ddpm.py
    ├── download_dataset.py
    ├── example
    │   └── mnist_visualize.png
    ├── inference.py
    ├── model.py
    └── train.py
```

# 运行效果

## 数据下载
使用python脚本可以下载MNIST手写数字识别数据集：
```shell
# cd diffusion_model
python download_dataset.py
```
![example](./example/mnist_visualize.png)

## 模型训练
使用train.py可以使用MNIST手写数字数据集进行扩散模型训练。
```shell
python train.py
```
输出结果如下:
```text
Starting training on mps...
Epoch 000/30: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:11<00:00,  9.84it/s, avg_loss=0.1844, loss=0.0340]
==> Epoch 000 Final Avg Loss: 0.184419
Epoch 001/30: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:11<00:00, 10.34it/s, avg_loss=0.0286, loss=0.0227]
==> Epoch 001 Final Avg Loss: 0.028581
Epoch 002/30: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:11<00:00, 10.21it/s, avg_loss=0.0203, loss=0.0240]
==> Epoch 002 Final Avg Loss: 0.020335
Epoch 003/30: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:11<00:00, 10.30it/s, avg_loss=0.0170, loss=0.0137]
==> Epoch 003 Final Avg Loss: 0.016954
Epoch 004/30: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:11<00:00, 10.22it/s, avg_loss=0.0164, loss=0.0257]
==> Epoch 004 Final Avg Loss: 0.016356
Epoch 005/30: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:11<00:00, 10.26it/s, avg_loss=0.0147, loss=0.0139]
==> Epoch 005 Final Avg Loss: 0.014666
[SAVE] Checkpoint saved: ./ckpt/model_epoch_005.pth
```

## 模型推理
使用inference.py可以从随机噪声开始进行手写数字生成。
```shell
python inference.py
```
下面是训练了30个Epoch的扩散模型的效果：
- pred表示给定的控制条件是预测数字几，s表示引导系数
- 基本上s>=1的时候就可以完全生成需要的内容了
- 相比于扩散模型来说，推理step从1000降低到10，速度提升100倍，效果也没怎么降低。
![result](./example/inference_visualize.png)

# 踩坑记录

### gold_v计算问题

gold_v应该描述的是从随机噪音指向真实图片的速度方向，而不是插值结果x_t指向真实图片的速度方向。

```python
# 正确示范
# [train.py] 
gold_v = x - eps

# 错误示范
# [ddpm.py]
# gold_v = x - x_t
```

