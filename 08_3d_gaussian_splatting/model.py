import torch
import torch.nn as nn
import numpy as np


# --- SH 球谐函数工具 ---
# 使用 0-3 阶球谐（共 16 个基函数）
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.4453057213202769,
    -0.5900435899266435,
]


def eval_sh(deg, sh_coeffs, dirs):
    """
    评估球谐函数
    sh_coeffs: [N, C, (deg+1)^2] 或 [N, (deg+1)^2, C]
    dirs: [N, 3] 归一化方向向量
    返回: [N, C] 颜色值
    """
    # sh_coeffs: [N, n_sh, 3]
    result = SH_C0 * sh_coeffs[:, 0]

    if deg > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result + \
            SH_C1 * (-y * sh_coeffs[:, 1] + z * sh_coeffs[:, 2] - x * sh_coeffs[:, 3])

    if deg > 1:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = result + \
            SH_C2[0] * xy * sh_coeffs[:, 4] + \
            SH_C2[1] * yz * sh_coeffs[:, 5] + \
            SH_C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[:, 6] + \
            SH_C2[3] * xz * sh_coeffs[:, 7] + \
            SH_C2[4] * (xx - yy) * sh_coeffs[:, 8]

    if deg > 2:
        result = result + \
            SH_C3[0] * y * (3.0 * xx - yy) * sh_coeffs[:, 9] + \
            SH_C3[1] * xy * z * sh_coeffs[:, 10] + \
            SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coeffs[:, 11] + \
            SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeffs[:, 12] + \
            SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coeffs[:, 13] + \
            SH_C3[5] * z * (xx - yy) * sh_coeffs[:, 14] + \
            SH_C3[6] * x * (xx - 3.0 * yy) * sh_coeffs[:, 15]

    return result


def rgb_to_sh0(rgb):
    """将 [0,1] RGB 转为 0 阶 SH 系数的 DC 分量"""
    return (rgb - 0.5) / SH_C0


class GaussianModel(nn.Module):

    def __init__(self, num_points=2000, radius=1.5, sh_degree=3, pcd=None):
        super().__init__()

        self.sh_degree = sh_degree
        self.n_sh_coeffs = (sh_degree + 1) ** 2  # 0阶=1, 1阶=4, 2阶=9, 3阶=16

        if pcd is not None:
            means, colors_raw = pcd
            # 初始化 SH 系数：DC (第 0 个) 用 RGB 初始化，其余为 0
            sh = torch.zeros(means.shape[0], self.n_sh_coeffs, 3)
            sh[:, 0, :] = rgb_to_sh0(colors_raw)  # DC 分量
        else:
            means = torch.rand(num_points, 3) * 2 - 1
            sh = torch.zeros(num_points, self.n_sh_coeffs, 3)

        self.num_points = means.shape[0]
        self.radius = radius

        self.gauss_params = nn.ParameterDict({
            "means": nn.Parameter(means),
            # 初始化时给三轴不同的随机扰动，打破球体对称性
            "scales": nn.Parameter(torch.log(
                torch.ones(means.shape[0], 3) * 0.003 + torch.rand(means.shape[0], 3) * 0.001
            )),
            "rotations": nn.Parameter(torch.tile(torch.tensor([1.0, 0, 0, 0]), (means.shape[0], 1))),
            "opacities": nn.Parameter(torch.ones(means.shape[0], 1) * 0.0),
            "sh_coeffs": nn.Parameter(sh),  # [N, n_sh, 3]
        })

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def rotations(self):
        return self.gauss_params["rotations"]

    @property
    def opacity(self):
        return self.gauss_params["opacities"]

    @property
    def sh_coeffs(self):
        return self.gauss_params["sh_coeffs"]

    def get_scaling(self):
        return torch.exp(self.scales)

    def get_rotation(self):
        return torch.nn.functional.normalize(self.rotations)

    def get_opacity(self):
        return torch.sigmoid(self.opacity)

    def get_color_from_sh(self, viewdirs):
        """
        根据观察方向计算 SH 颜色
        viewdirs: [N, 3] 从高斯中心指向相机的归一化方向
        返回: [N, 3] RGB in [0, 1]
        """
        color = eval_sh(self.sh_degree, self.sh_coeffs, viewdirs)
        # SH 输出可能超出 [0,1]，用 sigmoid 或 clamp
        return (color + 0.5).clamp(0.0, 1.0)

    def forward(self, camera_pos=None):
        """
        camera_pos: [3] 相机在世界坐标系中的位置（用于计算视角相关颜色）
        如果为 None，使用 DC 分量作为颜色（无视角依赖）
        """
        result = {
            "means": self.means,
            "scales": self.get_scaling(),
            "rotations": self.get_rotation(),
            "opacity": self.get_opacity(),
        }

        if camera_pos is not None:
            # 计算每个高斯指向相机的方向
            viewdirs = camera_pos.unsqueeze(0) - self.means  # [N, 3]
            viewdirs = viewdirs / (viewdirs.norm(dim=-1, keepdim=True) + 1e-8)
            result["colors"] = self.get_color_from_sh(viewdirs)
        else:
            # fallback: 只用 DC 分量
            result["colors"] = (SH_C0 * self.sh_coeffs[:, 0] + 0.5).clamp(0.0, 1.0)

        return result

    @torch.no_grad()
    def apply_constraints(self):
        # 限制缩放上限：factor=4 时 0.008 约为 2-3 像素半径，不会形成光团
        self.gauss_params["scales"].clamp_(max=np.log(0.008))
        # 限制位置
        limit = self.radius * 2.0
        self.gauss_params["means"].clamp_(-limit, limit)

    @torch.no_grad()
    def reset_opacity(self):
        """透明度重置：杀掉膨胀的幽灵球，让有用的点重新浮现（论文 trick）"""
        self.gauss_params["opacities"].fill_(-4.0)  # sigmoid(-4) ≈ 0.018

    def get_optimizer_groups(self, lr):
        """
        返回优化器参数组配置
        SH 系数中 DC 分量和高阶分量分开处理
        """
        return [
            {'params': [self.gauss_params["means"]], 'lr': lr * 0.1, 'name': 'means'},
            {'params': [self.gauss_params["sh_coeffs"]], 'lr': lr * 2.0, 'name': 'sh_coeffs'},
            {'params': [self.gauss_params["opacities"]], 'lr': lr * 2.0, 'name': 'opacities'},
            {'params': [self.gauss_params["scales"]], 'lr': lr * 1.0, 'name': 'scales'},
            {'params': [self.gauss_params["rotations"]], 'lr': lr * 0.5, 'name': 'rotations'},
        ]

    @torch.no_grad()
    def densify_and_prune(self, optimizer, grad_threshold=0.0004, min_opacity=0.01):
        """密度控制：克隆、分裂与剪枝"""
        if self.means.grad is None: return optimizer

        grads = torch.norm(self.means.grad, dim=-1)
        densify_mask = grads > grad_threshold
        opacities = self.get_opacity().squeeze()
        prune_mask = opacities < min_opacity

        new_params = {}
        for name, param in self.gauss_params.items():
            remain_param = param[~prune_mask]
            added_param = param[densify_mask]
            new_params[name] = nn.Parameter(torch.cat([remain_param, added_param], dim=0))

        self.gauss_params = nn.ParameterDict(new_params)
        self.num_points = self.gauss_params["means"].shape[0]

        current_lr = optimizer.param_groups[0]['lr']
        new_optimizer = torch.optim.Adam(self.get_optimizer_groups(current_lr), eps=1e-15)
        return new_optimizer

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
