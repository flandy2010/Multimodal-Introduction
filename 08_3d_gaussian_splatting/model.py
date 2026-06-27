import torch
import torch.nn as nn
import torch.nn.functional as F
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
    评估球谐函数 (Evaluate Spherical Harmonics)
    --------------------------------------------------------------
    功能: 给定观察方向，利用存储的球谐系数计算出该方向下的RGB颜色。
    这是3DGS实现“视角相关外观”（如金属反光、光泽感）的核心数学引擎。

    Args:
        deg (int): 球谐函数的阶数 (通常为3)。决定了表达复杂度的上限。
        sh_coeffs (torch.Tensor): 形状 [N, n_sh, 3]。
            存储的系数，n_sh = (deg+1)^2 (deg=3时为16)。
            其中 [:, 0, :] 是0阶(DC)， [:, 1:, :] 是高阶系数，全部可训练。
        dirs (torch.Tensor): 形状 [N, 3]。归一化的观察方向向量。
            注意：这里的 (x, y, z) 是方向向量的分量，不是高斯球的空间坐标！

    Returns:
        torch.Tensor: 形状 [N, 3]，即该高斯球在此观察方向下应呈现的RGB颜色。

    --------------------------------------------------------------
    物理意义与运算详解 (含索引映射表):

    球谐基函数将“球面上的颜色分布”分解为不同频率的波。
    color(dir) = Σ_{l} Σ_{m} c_{l,m} * Y_{l,m}(dir)

    代码中 SH_C0, SH_C1, SH_C2, SH_C3 是固定常数（归一化系数），
    它们确保了基函数在球面上的正交性和能量守恒，不参与训练。
    系数 c_{l,m} (即 sh_coeffs) 由梯度下降优化得出。
    """

    # ==================== 第 0 阶 (l=0, m=0) ====================
    # 运算: SH_C0 * sh_coeffs[:, 0]
    # SH_C0 ≈ 0.28209479 (即 1/(2*sqrt(pi)))
    # 物理意义: 球谐函数的“直流分量”。它不随观察方向变化，
    #           代表物体的基础固有色/漫反射颜色（类似哑光材质的底色）。
    #           这也是初始化时加载点云RGB填充的位置。
    result = SH_C0 * sh_coeffs[:, 0]

    # 拆分方向向量，便于后续运算
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]

    # ==================== 第 1 阶 (l=1, m=-1, 0, 1) ====================
    if deg > 0:
        # 运算: SH_C1 * ( -y*c1 + z*c2 - x*c3 )
        # 索引对应: idx=1(-y), idx=2(z), idx=3(-x)
        # SH_C1 ≈ 0.48860251 (sqrt(3/(4*pi)))
        # 物理意义: 线性渐变。捕捉平滑的明暗过渡。
        #   - idx=1 (-y) : 沿Y轴方向（上下）的明暗变化（如顶光照明）。
        #   - idx=2 (z)  : 沿Z轴（前后/视线方向）的变化。
        #   - idx=3 (-x) : 沿X轴方向（左右）的明暗变化。
        # 直观感受: 让球体看起来有立体感，亮面与暗面平滑过渡。
        result = result + SH_C1 * (-y * sh_coeffs[:, 1] + z * sh_coeffs[:, 2] - x * sh_coeffs[:, 3])

    # ==================== 第 2 阶 (l=2, m=-2, -1, 0, 1, 2) ====================
    if deg > 1:
        # 预计算二次项，提升运算效率
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z

        # 运算: 5个二次基函数的加权和，对应索引 idx=4 ~ 8
        # 物理意义: 开始表现“各向异性”和“光泽感”。
        #   - idx=4 (xy)   : 正相关于 x*y。关注对角方向。
        #         如果这个系数为正，斜着看物体时颜色会变亮/变暗。
        #   - idx=5 (yz)   : 正相关于 y*z。关注垂直与深度方向的对角。
        #   - idx=6 (2zz-xx-yy) : 沿Z轴的拉伸/压缩。
        #         若为正，正面看向物体（Z方向）颜色突出；侧面看则弱化。
        #   - idx=7 (xz)   : 正相关于 x*z。
        #   - idx=8 (xx-yy) : 正相关于 (x^2 - y^2)。分辨水平 vs 垂直方向。
        #         如果为正，水平方向（左右看）比垂直方向（上下看）更亮。
        result = result + \
                 SH_C2[0] * xy * sh_coeffs[:, 4] + \
                 SH_C2[1] * yz * sh_coeffs[:, 5] + \
                 SH_C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[:, 6] + \
                 SH_C2[3] * xz * sh_coeffs[:, 7] + \
                 SH_C2[4] * (xx - yy) * sh_coeffs[:, 8]

    # ==================== 第 3 阶 (l=3, m=-3 ... 3) ====================
    if deg > 2:
        # 运算: 7个三次基函数的加权和，对应索引 idx=9 ~ 15
        # 物理意义: 捕捉极其精细的高光闪烁、复杂金属拉丝纹路。
        #   - idx=9  : y*(3xx-yy)     -> 沿Y轴的三次扭曲光泽
        #   - idx=10 : xy*z           -> 空间对角的三次耦合
        #   - idx=11 : y*(4zz-xx-yy)  -> 垂直方向的高级光泽衰减
        #   - idx=12 : z*(2zz-3xx-3yy)-> 深度方向的非线性高光
        #   - idx=13 : x*(4zz-xx-yy)  -> 水平方向的高级光泽衰减
        #   - idx=14 : z*(xx-yy)      -> 水平/垂直与深度的混合干涉
        #   - idx=15 : x*(xx-3yy)     -> 沿X轴的三次扭曲光泽
        # 直观感受: 让物体出现“随角度闪烁”的锐利高光（如抛光金属）。
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

    def __init__(self, fx, fy, num_points=2000, radius=1.5, sh_degree=3, pcd=None):
        super().__init__()

        self.fx = fx
        self.fy = fy

        self.sh_degree = sh_degree          # 最终目标阶数
        self.active_sh_degree = 0           # 渐进激活：训练初期只用 0 阶
        self.n_sh_coeffs = (sh_degree + 1) ** 2
        self.radius = radius

        if pcd is not None:
            means, colors_raw = pcd
            sh_dc   = torch.zeros(means.shape[0], 1, 3)
            sh_dc[:, 0, :] = rgb_to_sh0(colors_raw)
            sh_rest = torch.zeros(means.shape[0], self.n_sh_coeffs - 1, 3)
        else:
            means   = torch.rand(num_points, 3) * 2 - 1
            sh_dc   = torch.zeros(num_points, 1, 3)
            sh_rest = torch.zeros(num_points, self.n_sh_coeffs - 1, 3)

        self.num_points = means.shape[0]

        self.gauss_params = nn.ParameterDict({
            "means":     nn.Parameter(means),
            "scales":    nn.Parameter(torch.log(torch.ones(means.shape[0], 3) * 0.003 + torch.rand(means.shape[0], 3) * 0.002)),
            "rotations": nn.Parameter(torch.tile(torch.tensor([1.0, 0, 0, 0]), (means.shape[0], 1))),
            "opacities": nn.Parameter(torch.ones(means.shape[0], 1) * (-2.0)),
            "sh_dc":     nn.Parameter(sh_dc),    # [N, 1, 3]  DC 分量，学习率 0.0025
            "sh_rest":   nn.Parameter(sh_rest),  # [N, 15, 3] 高阶分量，学习率 0.0025/20
        })

        # 屏幕空间最大半径（像素），每步渲染后从外部写入，用于 densify 时的屏幕空间剪枝
        # 不作为可学习参数，不放入 gauss_params
        self.max_radii2D = torch.zeros(self.num_points)

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
        """将 sh_dc 和 sh_rest 拼接返回，保持 [N, n_sh, 3] 接口兼容"""
        return torch.cat([self.gauss_params["sh_dc"], self.gauss_params["sh_rest"]], dim=1)

    def get_scaling(self):
        return torch.exp(self.scales)

    def get_rotation(self):
        return torch.nn.functional.normalize(self.rotations)

    def get_opacity(self):
        return torch.sigmoid(self.opacity)

    def get_color_from_sh(self, viewdirs):
        # 用 active_sh_degree 而非 sh_degree，实现渐进激活
        color = eval_sh(self.active_sh_degree, self.sh_coeffs, viewdirs)
        return (color + 0.5).clamp(0.0, 1.0)

    def oneupSHdegree(self):
        """每隔 1000 步调用一次，逐步开放更高阶 SH，论文标准做法"""
        if self.active_sh_degree < self.sh_degree:
            self.active_sh_degree += 1

    def forward(self, camera_pos=None):
        result = {
            "means": self.means,
            "scales": self.get_scaling(),
            "rotations": self.get_rotation(),
            "opacity": self.get_opacity(),
            # 原始叶节点参数引用，供 simple_rasterizer 手动写梯度用
            "_raw_opacity":   self.gauss_params["opacities"],   # logit, leaf
            "_raw_sh_coeffs": self.gauss_params["sh_dc"],   # 兼容旧接口（实际只有dc部分）
            "_raw_scales":    self.gauss_params["scales"],       # log-space, leaf
            "_raw_rotations": self.gauss_params["rotations"],    # unnorm quat, leaf
        }

        if camera_pos is not None:
            with torch.no_grad():
                viewdirs = camera_pos.unsqueeze(0) - self.means.detach()
                viewdirs = viewdirs / (viewdirs.norm(dim=-1, keepdim=True) + 1e-8)
            result["colors"] = self.get_color_from_sh(viewdirs)
        else:
            result["colors"] = (SH_C0 * self.sh_coeffs[:, 0] + 0.5).clamp(0.0, 1.0)

        return result

    def get_optimizer_groups(self):
        """
        返回论文推荐的绝对学习率配置
        注意：这里使用绝对学习率，不依赖外部 lr 参数
        """
        return [
            {'params': [self.gauss_params["means"]], 'lr': 0.00016, 'name': 'means'},
            # SH 系数分离存储（对齐原论文）：DC 分量 lr=0.0025，高阶分量 lr=0.0025/20
            {'params': [self.gauss_params["sh_dc"]],   'lr': 0.0025,        'name': 'sh_dc'},
            {'params': [self.gauss_params["sh_rest"]], 'lr': 0.0025 / 20.0, 'name': 'sh_rest'},
            {'params': [self.gauss_params["opacities"]], 'lr': 0.05, 'name': 'opacities'},
            {'params': [self.gauss_params["scales"]], 'lr': 0.005, 'name': 'scales'},
            {'params': [self.gauss_params["rotations"]], 'lr': 0.001, 'name': 'rotations'},
        ]

    @torch.no_grad()
    def reset_opacity(self):
        """透明度重置：强制所有点重新证明自己的存在价值"""
        # sigmoid(-4.6) ≈ 0.01
        self.gauss_params["opacities"].fill_(-4.6)

    @staticmethod
    def compute_screen_space_gradient(means, means_grad, c2w, fx, fy, eps=1e-6):
        """
        将 3D 世界空间的位置梯度，映射到 2D 图像空间的像素梯度。
        这是解决"近大远小"增密不公平问题的核心数学变换。

        Args:
            means (torch.Tensor): 高斯球的 3D 中心坐标，形状 [N, 3]。
            means_grad (torch.Tensor): means 对应的梯度，形状 [N, 3]。
            c2w (torch.Tensor): 当前相机的 3x4 或 4x4 外参矩阵（相机到世界）。
            fx, fy (float): 内参焦距（像素单位）。
            eps (float): 防止深度除零的小常数。

        Returns:
            torch.Tensor: 每个高斯球在屏幕上对应的 2D 梯度模长，形状 [N]。
        """

        # ==================== 第 1 步：将 3D 点转到相机坐标系 ====================
        # 提取旋转 R 和平移 T（世界坐标系到相机坐标系）
        # 注意：传入的 c2w 是“相机→世界”，所以世界→相机需要取逆矩阵。
        if c2w.shape == (3, 4):
            R_w2c = c2w[:3, :3].T  # 旋转矩阵的逆 = 转置
            t_w2c = -R_w2c @ c2w[:3, 3:4]  # t_c2w = -R_w2c @ t_w2c
        elif c2w.shape == (4, 4):
            R_w2c = c2w[:3, :3].T
            t_w2c = -R_w2c @ c2w[:3, 3:4]
        else:
            raise ValueError(f"c2w 矩阵形状 {c2w.shape} 不合法，应为 (3,4) 或 (4,4)")

        # 做线性变换：P_cam = R_w2c @ P_world + t_w2c
        cam_xyz = means @ R_w2c.T + t_w2c.squeeze(1)  # 形状 [N, 3]
        X, Y, Z = cam_xyz[:, 0], cam_xyz[:, 1], cam_xyz[:, 2]
        Z = torch.clamp(Z, min=eps)  # 避免除以 0

        # ==================== 第 2 步：计算透视投影的雅可比矩阵 ====================
        # 针孔投影方程：u = fx * X/Z + cx,   v = fy * Y/Z + cy
        # 雅可比 J (2x3) 表示 3D 点移动一点点，像素坐标移动多少：
        #   du/dX = fx/Z,   du/dY = 0,      du/dZ = -fx*X/Z^2
        #   dv/dX = 0,      dv/dY = fy/Z,   dv/dZ = -fy*Y/Z^2
        grad_x, grad_y, grad_z = means_grad[:, 0], means_grad[:, 1], means_grad[:, 2]

        # 将 3D 梯度映射到 2D 像素梯度
        grad_u = fx * (grad_x / Z - grad_z * X / (Z ** 2))
        grad_v = fy * (grad_y / Z - grad_z * Y / (Z ** 2))

        # ==================== 第 3 步：返回 2D 梯度的模长 ====================
        # 取 (du, dv) 的 L2 范数，作为判断“该点是否欠拟合”的最终指标
        grad_2d_norm = torch.norm(torch.stack([grad_u, grad_v], dim=-1), dim=-1)
        return grad_2d_norm

    @torch.no_grad()
    def update_densification_stats(self, viewspace_points, image_hw=None, gids=None, radii=None):
        """
        累积 2D 屏幕空间梯度统计 + 更新 max_radii2D，供 densify_and_prune 使用。
        必须在 loss.backward() 之后调用。

        viewspace_points: gsplat 返回的 info["means2d"]
          - absgrad=True 模式：.absgrad 存放 |∇2D|（归一化坐标）
        image_hw: (H, W) 将归一化梯度还原为像素坐标
        gids: 可见点在全量 N 中的原始下标
        radii: [N] 或 [N_visible] 屏幕空间半径（像素），用于更新 max_radii2D
        """
        if viewspace_points is None:
            return

        if hasattr(viewspace_points, "absgrad") and viewspace_points.absgrad is not None:
            grad = viewspace_points.absgrad.squeeze(0)
            if image_hw is not None:
                H, W = image_hw
                scale = torch.tensor([W, H], dtype=grad.dtype, device=grad.device)
                grad = grad * scale
            grad_norm = grad.norm(dim=-1)
        elif viewspace_points.grad is not None:
            grad = viewspace_points.grad.squeeze(0)
            grad_norm = grad.norm(dim=-1)
        else:
            return

        if not hasattr(self, "_grad_accum") or self._grad_accum.shape[0] != self.num_points:
            self._grad_accum = torch.zeros(self.num_points, device=grad_norm.device)
            self._grad_count = torch.zeros(self.num_points, device=grad_norm.device)

        if gids is not None:
            self._grad_accum.scatter_add_(0, gids, grad_norm)
            self._grad_count.index_add_(0, gids, torch.ones_like(grad_norm))
        else:
            n = min(grad_norm.shape[0], self.num_points)
            self._grad_accum[:n] += grad_norm[:n]
            self._grad_count[:n] += 1

        # 同步更新 max_radii2D（原论文在 train.py 里做，封装到这里更简洁）
        if radii is not None:
            r = radii.view(-1).float()
            if self.max_radii2D.device != r.device:
                self.max_radii2D = self.max_radii2D.to(r.device)
            visibility = r > 0
            self.max_radii2D[visibility] = torch.max(
                self.max_radii2D[visibility], r[visibility]
            )

    @torch.no_grad()
    def densify_and_prune(self, optimizer, grad_threshold=0.0002, min_opacity=0.005,
                          max_scale=None, c2w=None, max_screen_size=None):
        """
        自适应密度控制（对齐原论文）
        max_screen_size: 屏幕空间半径上限（像素），超过则 prune。
                         原论文：step > opacity_reset_interval(3000) 后传 20，之前传 None。
        """
        # ==================== 1. 前置检查 ====================
        # 优先使用 absgrad 累积量（来自 update_densification_stats），
        # 其次回退到单帧 means.grad（兼容旧模式）
        has_absgrad = hasattr(self, "_grad_accum") and self._grad_count.sum() > 0
        if not has_absgrad and self.means.grad is None:
            return optimizer

        # ==================== 2. 获取屏幕空间梯度 ====================
        if has_absgrad:
            # absgrad 模式：直接用累积的平均 2D 梯度范数，已经是屏幕空间，无需再投影
            count = self._grad_count.clamp(min=1)
            grads = self._grad_accum / count  # [N] 平均 2D 梯度范数
            # 用完后重置累积量
            self._grad_accum.zero_()
            self._grad_count.zero_()
        else:
            # 回退：用 means.grad 通过雅可比矩阵投影到屏幕空间
            grads = self.compute_screen_space_gradient(
                self.gauss_params["means"],
                self.gauss_params["means"].grad,
                c2w,
                self.fx,
                self.fy,
            )

        # ==================== 3. 生成增密与剪枝掩码 ====================
        scales    = self.get_scaling()                # [N, 3]
        opacities = self.get_opacity().squeeze()      # [N]
        scale_max = scales.max(dim=-1).values         # [N] 最大轴半径

        # 梯度分布诊断（帮助排查 clone/split 不触发的问题）
        print(f"  [Densify diag] grads: max={grads.max():.5f} mean={grads.mean():.5f} "
              f"threshold={grad_threshold:.5f} "
              f"above_threshold={( grads > grad_threshold).sum().item()}/{self.num_points}")

        # 需要增密的点：屏幕空间梯度 > 阈值
        need_densify = grads > grad_threshold

        # Split（大点分裂）：梯度大 且 最大轴半径 > percent_dense * scene_extent
        # 论文：percent_dense=0.01，scene_extent=radius
        split_threshold = self.radius * 0.01
        split_mask = need_densify & (scale_max > split_threshold)

        # Clone（小点克隆）：梯度大 且 最大轴半径 <= 阈值
        clone_mask = need_densify & (scale_max <= split_threshold)

        # Prune：透明度太低 | world-space 超大球 | screen-space 超大球（原论文三合一）
        prune_mask = opacities < min_opacity
        if max_scale is not None:
            prune_mask = prune_mask | (scale_max > max_scale)
        if max_screen_size is not None:
            device = self.gauss_params["means"].device
            radii = self.max_radii2D.to(device)
            prune_mask = prune_mask | (radii > max_screen_size)

        # ==================== 4. 如果没有变化，清空梯度后提前返回 ====================
        if not (split_mask.any() or clone_mask.any() or prune_mask.any()):
            # 清空梯度防止下一轮迭代重复触发（关键安全步骤）
            for param in self.gauss_params.parameters():
                if param.grad is not None:
                    param.grad = None
            return optimizer

        # ==================== 5. 构造新的参数张量 ====================
        # 先计算 split 的位置偏移（需要 scales 和 rotations，在遍历前计算）
        # 原论文做法：沿最大尺度轴方向采样偏移，而非随机均匀偏移
        # 步骤：① 找每个 split 球的最大尺度轴方向（局部坐标系中的 e_i）
        #       ② 用四元数把该方向旋转到世界坐标系
        #       ③ 沿该世界方向采样偏移量（以物理尺度为标准差的正态分布）
        if split_mask.any():
            split_scales = torch.exp(self.gauss_params["scales"][split_mask])   # [S, 3] 物理尺度
            split_quats  = F.normalize(self.gauss_params["rotations"][split_mask], p=2, dim=-1)  # [S, 4]

            # 最大尺度轴的单位向量（局部坐标系中的 x/y/z 之一）
            max_axis_idx = split_scales.argmax(dim=-1)   # [S]，值为 0/1/2
            # 构建局部轴向量 [S, 3]
            local_axis = torch.zeros_like(split_scales)
            local_axis.scatter_(1, max_axis_idx.unsqueeze(1), 1.0)  # one-hot

            # 用四元数旋转局部轴到世界坐标系
            # 四元数 (w, x, y, z) 旋转向量 v：R(q) * v
            w = split_quats[:, 0:1]; x = split_quats[:, 1:2]
            y = split_quats[:, 2:3]; z = split_quats[:, 3:4]
            vx = local_axis[:, 0:1]; vy = local_axis[:, 1:2]; vz = local_axis[:, 2:3]
            # R(q)*v 展开（标准四元数旋转公式）
            world_axis = torch.cat([
                (1 - 2*(y*y + z*z))*vx + 2*(x*y - w*z)*vy + 2*(x*z + w*y)*vz,
                2*(x*y + w*z)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - w*x)*vz,
                2*(x*z - w*y)*vx + 2*(y*z + w*x)*vy + (1 - 2*(x*x + y*y))*vz,
            ], dim=-1)   # [S, 3]，已归一化

            # 偏移量：以最大物理尺度为标准差采样（原论文做法）
            max_scale = split_scales.gather(1, max_axis_idx.unsqueeze(1)).squeeze(1)  # [S]
            offset = torch.randn(split_mask.sum().item(), device=world_axis.device)    # [S]
            offset = offset * max_scale   # 按最大尺度缩放
            split_means_offset = world_axis * offset.unsqueeze(1)   # [S, 3]
        else:
            split_means_offset = None

        new_params = {}
        for name, param in self.gauss_params.items():
            # 1) 保留点：未被剪枝 且 未被分裂的点
            #    注意：split 的原球要移除（它被两个子球替代），所以排除 split_mask
            keep_mask = ~prune_mask & ~split_mask
            remain = param[keep_mask]

            # 2) 克隆的点直接复制（位置不变）
            cloned = param[clone_mask]

            # 3) 分裂：两份拷贝，scale ÷ 1.6，means 沿最大尺度轴偏移
            split_src = param[split_mask]
            if name == "scales":
                # log(s) - log(1.6)  等价于物理尺度 ÷ 1.6
                split_src = split_src - np.log(1.6)
            elif name == "means" and split_means_offset is not None:
                # 两份 split 球的位置：一个 +offset，一个 -offset
                split_src = torch.cat([
                    split_src + split_means_offset,
                    split_src - split_means_offset,
                ], dim=0)
                new_params[name] = torch.cat([remain, cloned, split_src], dim=0)
                continue   # means 已经处理，跳过下面的 cat

            # 非 means 的参数：split 球两份相同（scale 已缩小，其他保持原值）
            new_params[name] = torch.cat([remain, cloned, split_src, split_src], dim=0)

        # ==================== 6. 更新模型参数 ====================
        self.gauss_params = nn.ParameterDict(new_params)
        self.num_points = self.gauss_params["means"].shape[0]

        # 点数变化后重置梯度累积量 和 max_radii2D（尺寸对齐新 num_points）
        device = self.gauss_params["means"].device
        self._grad_accum = torch.zeros(self.num_points, device=device)
        self._grad_count = torch.zeros(self.num_points, device=device)
        self.max_radii2D  = torch.zeros(self.num_points, device=device)  # 设备对齐

        # ==================== 7. 选项A：彻底销毁旧优化器 + 重建新优化器 ====================
        # 7.1 将旧优化器中所有参数的梯度置 None（释放梯度显存）
        optimizer.zero_grad(set_to_none=True)

        # 7.2 清空旧优化器的内部状态字典（释放 Adam 动量的显存）
        optimizer.state.clear()

        # 7.3 手动删除旧优化器对象（在函数返回前强制释放引用）
        del optimizer
        torch.cuda.empty_cache()  # 强制释放 CUDA 碎片，防止新旧优化器状态同时驻留

        # 7.4 使用新的参数组创建全新的 Adam 优化器
        # 注意：新参数的 grad 默认为 None，所以新优化器的动量从零开始
        new_optimizer = torch.optim.Adam(
            self.get_optimizer_groups(),  # 这个方法需要返回 list of dict 参数组
            eps=1e-15  # 3DGS 官方推荐值
        )

        return new_optimizer

    @torch.no_grad()
    def get_diagnostics(self, step=None, min_opacity=0.005):
        """
        返回训练诊断统计量字典
        核心监控 4 项：op / ab_op / r / ab_r
        """
        diagnostics = {}
        total = self.num_points
        diagnostics['num_gaussians_total'] = total

        # ================== 不透明度 ==================
        opacities = self.get_opacity().squeeze()  # [N]

        # top10% 不透明度均值（最不透明的 10% 椭球）
        k_op = max(1, int(total * 0.1))
        top10_op, _ = torch.topk(opacities, k_op)
        diagnostics['op_top10_mean'] = top10_op.mean().item()
        diagnostics['op_mean']       = opacities.mean().item()

        # 不透明度过低的比例（< min_opacity 视为无效椭球）
        diagnostics['ab_op'] = (opacities < min_opacity).float().mean().item()

        # 保留旧字段兼容（logger 可能用到）
        diagnostics['frac_opacity_below_0.05'] = (opacities < 0.05).float().mean().item()
        diagnostics['effective_fraction'] = (opacities > min_opacity).float().mean().item()

        # ================== 半径 ==================
        scales = self.get_scaling()           # [N, 3] 物理尺度
        avg_radius = scales.mean(dim=-1)      # [N] 每颗高斯的平均半径

        # top10% 半径均值（最大的 10% 椭球）
        k_r = max(1, int(total * 0.1))
        top10_r, _ = torch.topk(avg_radius, k_r)
        diagnostics['r_top10_mean'] = top10_r.mean().item()
        diagnostics['r_mean']       = avg_radius.mean().item()

        # 半径过大的比例（> apply_constraints 阈值 = radius * 0.01）
        max_allowed = self.radius * 0.01
        diagnostics['ab_r'] = (avg_radius > max_allowed).float().mean().item()

        # 保留旧字段兼容
        diagnostics['avg_radius_max'] = avg_radius.max().item()

        # ================== 球谐系数 ==================
        sh_dc   = self.gauss_params["sh_dc"]    # [N, 1, 3]
        sh_rest = self.gauss_params["sh_rest"]  # [N, 15, 3]
        sh = torch.cat([sh_dc, sh_rest], dim=1)
        diagnostics['sh_abs_mean'] = sh.abs().mean().item()
        if self.sh_degree > 0:
            dc   = sh[:, 0, :]
            high = sh[:, 1:, :]
            dc_norm   = dc.norm(dim=-1).mean().item()
            high_norm = high.norm(dim=-1).mean().item()
            diagnostics['sh_dc_norm']        = dc_norm
            diagnostics['sh_high_norm']      = high_norm
            diagnostics['sh_high_dc_ratio']  = high_norm / (dc_norm + 1e-8)
        else:
            diagnostics['sh_high_dc_ratio'] = 0.0

        if step is not None:
            diagnostics['step'] = step

        return diagnostics

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
