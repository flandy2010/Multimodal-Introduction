import os
import json
from datetime import datetime
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter


def dict_to_namespace(d):
    """递归将字典转回 SimpleNamespace"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def setup_experiment(cfg):

    # 创建运行目录: outputs/20231027_153022_unet_flow_matching/
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{time_str}_{cfg.model_type}_{cfg.method.type}"
    exp_dir = os.path.join("./runs/", exp_name)

    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "samples"), exist_ok=True)

    # 将配置保存为 JSON，方便以后复现
    # 注意：SimpleNamespace 需要转为 dict
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        # 递归转换函数省略，简单处理：
        json.dump(str(cfg), f, indent=4)

    return exp_dir