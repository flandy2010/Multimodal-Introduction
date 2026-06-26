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
            # sh存放了高斯球的SH系数，第0阶为RGB(3维，取值[0, 1])通过固定因子缩放得到，高阶留到训练时慢慢学。
            sh = torch.zeros(means.shape[0], self.n_sh_coeffs, 3)
            sh[:, 0, :] = rgb_to_sh0(colors_raw)
        else:
            means = torch.rand(num_points, 3) * 2 - 1
            sh = torch.zeros(num_points, self.n_sh_coeffs, 3)

        self.num_points = means.shape[0]

        self.gauss_params = nn.ParameterDict({
            # means: 高斯球在3D世界空间的中心坐标，shape=[N, 3]
            "means": nn.Parameter(means),
            # scales: 高斯椭球在 X、Y、Z三个方向上的半径，shape=[N, 3]，设定为对数尺度方便使用的时候用exp转为正数
            "scales": nn.Parameter(torch.log(torch.ones(means.shape[0], 3) * 0.003 + torch.rand(means.shape[0], 3) * 0.002)),
            # rotations: 每个高斯椭球在空间中的朝向，即单位四元数
            "rotations": nn.Parameter(torch.tile(torch.tensor([1.0, 0, 0, 0]), (means.shape[0], 1))),
            # opacities：高斯球对最终渲染像素的贡献权重，范围在 0~1 之间，shape=[N, ]
            "opacities": nn.Parameter(torch.ones(means.shape[0], 1) * (-2.0)),  # sigmoid(-2) ≈ 0.12
            # sh_coeffs：颜色随观察方向变化的球谐系数，shape=[N, n_sh_coeffs, 3]
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
            "_raw_sh_coeffs": self.gauss_params["sh_coeffs"],   # leaf
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
            {'params': [self.gauss_params["sh_coeffs"]], 'lr': 0.0025, 'name': 'sh_coeffs'},
            {'params': [self.gauss_params["opacities"]], 'lr': 0.05, 'name': 'opacities'},
            {'params': [self.gauss_params["scales"]], 'lr': 0.005, 'name': 'scales'},
            {'params': [self.gauss_params["rotations"]], 'lr': 0.001, 'name': 'rotations'},
        ]

    @torch.no_grad()
    def apply_constraints(self):
        # 论文原版：剔除 log(scale) > log(scene_extent * percent_dense)
        # percent_dense=0.01，scene_extent 即归一化后的 scene_radius
        # 这里 radius 已是归一化后的场景半径（相机球半径，约 1.0）
        # 论文实际用 0.01 * scene_radius 作为上界，约束更紧
        max_log_scale = np.log(self.radius * 0.01)  # 0.01 * radius（论文标准）
        self.gauss_params["scales"].clamp_(max=max_log_scale)
        # 约束不能偏离中心太远
        limit = self.radius * 1.5  # 论文用 scene_extent，比之前的 2.0 更严格
        self.gauss_params["means"].clamp_(-limit, limit)

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
    def update_densification_stats(self, viewspace_points):
        """
        从 gsplat absgrad=True 模式中累积 2D 梯度统计，用于致密化判断。
        viewspace_points: info["means2d"]，形状 [1, N, 2]，其 .absgrad 由 gsplat 在 backward 时填充。
        注意：必须在 loss.backward() 之后调用（此时 absgrad 已被填充）。
        """
        if viewspace_points is None:
            return
        # gsplat absgrad 模式：梯度累积在 .absgrad 属性上，形状 [1, N, 2]
        if hasattr(viewspace_points, "absgrad") and viewspace_points.absgrad is not None:
            grad = viewspace_points.absgrad.squeeze(0)  # [N, 2]
            grad_norm = grad.norm(dim=-1)               # [N]
        elif viewspace_points.grad is not None:
            # 兼容 retain_grad 旧模式
            grad = viewspace_points.grad.squeeze(0)
            grad_norm = grad.norm(dim=-1)
        else:
            return

        if not hasattr(self, "_grad_accum"):
            self._grad_accum = torch.zeros(self.num_points, device=grad_norm.device)
            self._grad_count = torch.zeros(self.num_points, device=grad_norm.device)

        n = min(grad_norm.shape[0], self.num_points)
        self._grad_accum[:n] += grad_norm[:n]
        self._grad_count[:n] += 1

    @torch.no_grad()
    def densify_and_prune(self, optimizer, grad_threshold=0.0002, min_opacity=0.005, max_scale=None, c2w=None):
        """
        自适应密度控制（选项A：简洁安全版）
        - 使用屏幕空间梯度判断增密（精准解决近大远小）
        - 直接重建优化器，丢弃旧动量（避免复杂的索引映射Bug）
        - 彻底清理旧优化器，防止显存泄漏
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
        # 获取物理属性（注意：scales和opacities已经是经过 exp/sigmoid 还原的物理值）
        scales = self.get_scaling()  # [N, 3]
        opacities = self.get_opacity().squeeze()  # [N]
        scale_max = scales.max(dim=-1).values  # [N] 最大轴半径（论文用 max 而非 mean）

        # 需要增密的点：屏幕空间梯度 > 阈值
        need_densify = grads > grad_threshold

        # Split（大点分裂）：梯度大 且 最大轴半径 > percent_dense * scene_extent
        # 论文：percent_dense=0.01，scene_extent=radius
        split_threshold = self.radius * 0.01
        split_mask = need_densify & (scale_max > split_threshold)

        # Clone（小点克隆）：梯度大 且 最大轴半径 <= 阈值
        clone_mask = need_densify & (scale_max <= split_threshold)

        # Prune（剪枝）：透明度太低 或 最大轴尺度超过限制
        prune_mask = opacities < min_opacity
        if max_scale is not None:
            prune_mask = prune_mask | (scale_max > max_scale)

        # ==================== 4. 如果没有变化，清空梯度后提前返回 ====================
        if not (split_mask.any() or clone_mask.any() or prune_mask.any()):
            # 清空梯度防止下一轮迭代重复触发（关键安全步骤）
            for param in self.gauss_params.parameters():
                if param.grad is not None:
                    param.grad = None
            return optimizer

        # ==================== 5. 构造新的参数张量 ====================
        new_params = {}
        for name, param in self.gauss_params.items():
            # 1) 保留未被剪枝的点
            remain = param[~prune_mask]

            # 2) 克隆的点直接复制
            cloned = param[clone_mask]

            # 3) 分裂的点复制一份，但尺度要缩小（物理半径除以 1.6）
            split_src = param[split_mask]
            if name == "scales":
                # 因为 self.gauss_params 存的是 log(scale)，
                # 所以 log(s) - log(1.6) 等价于物理尺度除以 1.6
                split_src = split_src - np.log(1.6)

            # 拼接顺序：[保留点, 克隆点, 分裂点]
            new_params[name] = torch.cat([remain, cloned, split_src], dim=0)

        # ==================== 6. 更新模型参数 ====================
        self.gauss_params = nn.ParameterDict(new_params)
        self.num_points = self.gauss_params["means"].shape[0]

        # 点数变化后重置梯度累积量，尺寸对齐新 num_points
        device = self.gauss_params["means"].device
        self._grad_accum = torch.zeros(self.num_points, device=device)
        self._grad_count = torch.zeros(self.num_points, device=device)

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
        返回训练诊断统计量字典，对应监控清单 ②③④⑤
        调用示例: stats = model.get_diagnostics(step=iteration)
        """
        diagnostics = {}

        # ================== ② 高斯数量与密集化动态 ==================
        total = self.num_points
        diagnostics['num_gaussians_total'] = total

        opacities = self.get_opacity().squeeze()          # [N] 物理透明度
        effective = (opacities > min_opacity).sum().item()
        diagnostics['num_gaussians_effective'] = effective
        diagnostics['effective_fraction'] = effective / total if total > 0 else 0.0

        # ================== ③ 透明度分布 ==================
        diagnostics['opacity_min'] = opacities.min().item()
        diagnostics['opacity_max'] = opacities.max().item()
        diagnostics['opacity_mean'] = opacities.mean().item()
        diagnostics['opacity_median'] = opacities.median().item()

        # 分段占比
        diagnostics['frac_opacity_below_0.05'] = (opacities < 0.05).float().mean().item()
        diagnostics['frac_opacity_0.05_0.5']  = ((opacities >= 0.05) & (opacities < 0.5)).float().mean().item()
        diagnostics['frac_opacity_0.5_0.95']  = ((opacities >= 0.5) & (opacities < 0.95)).float().mean().item()
        diagnostics['frac_opacity_above_0.95']= (opacities >= 0.95).float().mean().item()

        # ================== ④ 协方差与缩放 ==================
        scales = self.get_scaling()                     # [N, 3] 物理尺度
        # 各轴平均尺度与标准差
        for i, axis in enumerate(['x','y','z']):
            diagnostics[f'scale_mean_{axis}'] = scales[:, i].mean().item()
            diagnostics[f'scale_std_{axis}']  = scales[:, i].std().item()

        avg_radius = scales.mean(dim=-1)                # 每颗高斯的平均半径
        diagnostics['avg_radius_mean']   = avg_radius.mean().item()
        diagnostics['avg_radius_median'] = avg_radius.median().item()
        diagnostics['avg_radius_max']    = avg_radius.max().item()
        diagnostics['avg_radius_min']    = avg_radius.min().item()

        # 过大 / 过小尺度警告 (threshold 沿用 apply_constraints 中的 0.1*radius)
        max_allowed = self.radius * 0.1
        diagnostics['num_large_scale'] = (avg_radius > max_allowed).sum().item()
        diagnostics['num_tiny_scale']  = (avg_radius < 1e-5).sum().item()

        # ================== ⑤ 球谐系数统计 ==================
        sh = self.gauss_params["sh_coeffs"]              # [N, n_sh, 3]
        diagnostics['sh_abs_mean'] = sh.abs().mean().item()
        diagnostics['sh_std']      = sh.std().item()

        if self.sh_degree > 0:
            dc = sh[:, 0, :]                             # 0阶直流分量
            high = sh[:, 1:, :]                          # 高阶系数
            dc_norm   = dc.norm(dim=-1).mean().item()
            high_norm = high.norm(dim=-1).mean().item()
            diagnostics['sh_dc_norm']   = dc_norm
            diagnostics['sh_high_norm'] = high_norm
            diagnostics['sh_high_dc_ratio'] = high_norm / (dc_norm + 1e-8)
        else:
            diagnostics['sh_high_dc_ratio'] = 0.0

        if step is not None:
            diagnostics['step'] = step

        return diagnostics

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
