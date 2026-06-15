import os
import json
from datetime import datetime
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter


def namespace_to_dict(obj):
    """递归将 SimpleNamespace 转为 dict"""
    if isinstance(obj, SimpleNamespace):
        # vars(obj) 可以获取 Namespace 内部的字典
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(item) for item in obj]
    else:
        return obj

def dict_to_namespace(obj):
    """递归将 dict 转为 SimpleNamespace"""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [dict_to_namespace(item) for item in obj]
    else:
        return obj


def load_config_from_dir(exp_dir):
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)  # 此时得到的是 dict

    # 将 dict 重新转回 SimpleNamespace 使得 .common 操作合法
    cfg = dict_to_namespace(cfg_dict)
    return cfg


def setup_experiment(cfg):

    # 创建运行目录: outputs/20231027_153022_unet_flow_matching/
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{time_str}_{cfg.model_type}_{cfg.method.type}"
    exp_dir = os.path.join("./runs/", exp_name)

    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "samples"), exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        # 先转成字典，再存 JSON
        cfg_dict = namespace_to_dict(cfg)
        json.dump(cfg_dict, f, indent=4)

    return exp_dir