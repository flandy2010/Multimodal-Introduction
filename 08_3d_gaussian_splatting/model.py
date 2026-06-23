import torch
import torch.nn as nn
import numpy as np


# --- SH 球谐函数工具 ---
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
    sh_coeffs: [N, n_sh, 3]
    dirs: [N, 3] 归一化方向向量
    返回: [N, 3] 颜色值
    """
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
        self.n_sh_coeffs = (sh_degree + 1) ** 2
        self.radius = radius

        if pcd is not None:
            means, colors_raw = pcd
            sh = torch.zeros(means.shape[0], self.n_sh_coeffs, 3)
            sh[:, 0, :] = rgb_to_sh0(colors_raw)
        else:
            means = torch.rand(num_points, 3) * 2 - 1
            sh = torch.zeros(num_points, self.n_sh_coeffs, 3)

        self.num_points = means.shape[0]

        self.gauss_params = nn.ParameterDict({
            "means": nn.Parameter(means),
            # 三轴随机扰动打破球体对称
            "scales": nn.Parameter(torch.log(
                torch.ones(means.shape[0], 3) * 0.003 + torch.rand(means.shape[0], 3) * 0.002
            )),
            "rotations": nn.Parameter(torch.tile(torch.tensor([1.0, 0, 0, 0]), (means.shape[0], 1))),
            "opacities": nn.Parameter(torch.ones(means.shape[0], 1) * (-2.0)),  # sigmoid(-2) ≈ 0.12
            "sh_coeffs": nn.Parameter(sh),
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
        color = eval_sh(self.sh_degree, self.sh_coeffs, viewdirs)
        return (color + 0.5).clamp(0.0, 1.0)

    def forward(self, camera_pos=None):
        result = {
            "means": self.means,
            "scales": self.get_scaling(),
            "rotations": self.get_rotation(),
            "opacity": self.get_opacity(),
        }

        if camera_pos is not None:
            viewdirs = camera_pos.unsqueeze(0) - self.means
            viewdirs = viewdirs / (viewdirs.norm(dim=-1, keepdim=True) + 1e-8)
            result["colors"] = self.get_color_from_sh(viewdirs)
        else:
            result["colors"] = (SH_C0 * self.sh_coeffs[:, 0] + 0.5).clamp(0.0, 1.0)

        return result

    @torch.no_grad()
    def apply_constraints(self):
        # 论文推荐：剔除比场景还大的噪音球
        max_scale = self.radius * 0.1  # scene_radius * 0.1
        self.gauss_params["scales"].clamp_(max=np.log(max_scale))
        # 限制位置
        limit = self.radius * 2.0
        self.gauss_params["means"].clamp_(-limit, limit)

    @torch.no_grad()
    def reset_opacity(self):
        """透明度重置（论文 trick）：强制所有点重新证明自己的存在价值"""
        # sigmoid(-4.6) ≈ 0.01
        self.gauss_params["opacities"].fill_(-4.6)

    def get_optimizer_groups(self):
        """
        返回论文推荐的绝对学习率配置
        注意：这里使用绝对学习率，不依赖外部 lr 参数
        """
        return [
            {'params': [self.gauss_params["means"]], 'lr': 0.00016, 'name': 'means'},
            {'params': [self.gauss_params["sh_coeffs"]], 'lr': 0.0025, 'name': 'sh_coeffs'},
            {'params': [self.gauss_params["opacities"]], 'lr': 0.05, 'name': 'opacities'},
            {'params': [self.gauss_params["scales"]], 'lr': 0.005, 'name': 'scales'},
            {'params': [self.gauss_params["rotations"]], 'lr': 0.001, 'name': 'rotations'},
        ]

    @torch.no_grad()
    def densify_and_prune(self, optimizer, grad_threshold=0.0002, min_opacity=0.005, max_scale=None):
        """
        密度控制：分裂大点、克隆小点、剪枝透明/巨大点
        完全按照论文逻辑实现
        """
        if self.means.grad is None:
            return optimizer

        grads = torch.norm(self.means.grad, dim=-1)
        scales = self.get_scaling()
        opacities = self.get_opacity().squeeze()
        scale_mean = scales.mean(dim=-1)

        # 需要 densify 的点：梯度超过阈值
        need_densify = grads > grad_threshold

        # Split：梯度大 + 尺度大 → 劈成两个更小的点
        split_mask = need_densify & (scale_mean > 0.01)

        # Clone：梯度大 + 尺度小 → 复制一个点
        clone_mask = need_densify & (scale_mean <= 0.01)

        # Prune：透明度太低 或 尺度超过场景半径 * 0.1
        prune_mask = opacities < min_opacity
        if max_scale is not None:
            prune_mask = prune_mask | (scale_mean > max_scale)

        # 构造新参数
        new_params = {}
        for name, param in self.gauss_params.items():
            # 1. 保留未被 prune 的点
            remain = param[~prune_mask]

            # 2. Clone 的点直接复制
            cloned = param[clone_mask]

            # 3. Split 的点：复制一份，scale 缩小
            split_src = param[split_mask]
            if name == "scales":
                # 分裂后 scale 除以 1.6
                split_src = split_src - np.log(1.6)

            new_params[name] = nn.Parameter(torch.cat([remain, cloned, split_src], dim=0))

        self.gauss_params = nn.ParameterDict(new_params)
        self.num_points = self.gauss_params["means"].shape[0]

        # 重建优化器
        new_optimizer = torch.optim.Adam(self.get_optimizer_groups(), eps=1e-15)
        return new_optimizer

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
