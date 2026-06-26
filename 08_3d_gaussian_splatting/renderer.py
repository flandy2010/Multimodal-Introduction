import torch
import torch.nn.functional as F

# 尝试导入 gsplat，若 CUDA 不可用则标记为 None
try:
    from gsplat import rasterization as _gsplat_rasterization
    # 验证 CUDA 扩展真的可用（Mac/MPS 上 gsplat 会 import 成功但 _C 为 None）
    import gsplat.cuda._wrapper as _gsplat_cuda
    if _gsplat_cuda._C is None:
        raise ImportError("gsplat CUDA extension not available")
    _GSPLAT_AVAILABLE = True
except Exception:
    _GSPLAT_AVAILABLE = False
    print("⚠️  gsplat CUDA 不可用，自动降级到 simple_rasterizer（MPS/CPU 模式）")


def auto_rasterizer(gaussians, w2c, K, H, W, tile_size=16):
    """
    自动三路选择渲染后端：
    ┌─────────────────────────────────────────────────────────────────┐
    │  CUDA 可用（H20/P800） → gsplat_rasterizer  （最快，支持致密化）  │
    │  MPS 可用（Mac M系列） → mps_rasterizer     （向量化，无 for 循环）│
    │  其他（CPU）           → simple_rasterizer  （tile 循环，兜底）   │
    └─────────────────────────────────────────────────────────────────┘
    接口与 gsplat_rasterizer 完全一致，train.py 只需把调用换成 auto_rasterizer。
    注意：MPS/CPU 模式下 gaussians["viewspace_points"] 不会被写入，
          strategy.step 的致密化将自动跳过（already guarded by `is not None`）。
    """
    if _GSPLAT_AVAILABLE:
        return gsplat_rasterizer(gaussians, w2c, K, H, W, tile_size=tile_size)
    elif gaussians["means"].device.type == "mps":
        return mps_rasterizer(gaussians, w2c, K, H, W)
    else:
        return simple_rasterizer(gaussians, w2c, K, H, W, tile_size=128)


def gsplat_rasterizer(gaussians, w2c, K, H, W, tile_size=16):
    """
    作为 simple_rasterizer 的直接替代品。
    gaussians: 一个字典或对象，包含 means, scales, quats, opacities, colors/shs
    w2c: [4, 4] 矩阵 (World-to-Camera)
    K: [3, 3] 矩阵 (Camera Intrinsics)
    """
    # 1. 提取参数并进行必要的激活 (假设模型存储的是原始值)
    means = gaussians["means"]  # [N, 3]
    scales = torch.exp(gaussians["scales"])  # [N, 3] 尺度通常在对数空间
    quats = F.normalize(gaussians["rotations"], p=2, dim=-1)  # [N, 4] 必须单位化
    opacities = torch.sigmoid(gaussians["opacity"])  # [N, 1] 透明度 [0, 1]
    opacities = opacities.squeeze(-1)

    # 获取颜色 (如果是三阶球谐函数，gsplat 会自动处理)
    # 注意：如果你的 model() 已经根据 camera_pos 算好了颜色，直接取
    colors = gaussians["colors"]  # [N, 3] 或 [N, 48, 3]

    # 2. 构造 viewmat (gsplat 需要 [4, 4])
    viewmat = w2c.view(-1, 4, 4)

    if len(K.shape) == 2:
        K = K.unsqueeze(0)

    # 3. 调用 gsplat 加速渲染
    # render_colors: [H, W, 3]
    # absgrad=True：让 gsplat 在 CUDA kernel 内部直接累积 |∇2D| 到 means2d.absgrad，
    # 无需 retain_grad()，不会把渲染中间量 pin 在 PyTorch 计算图里，大幅节省显存。
    render_colors, render_alphas, info = _gsplat_rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        tile_size=tile_size,
        absgrad=True,  # 官方推荐：用 absgrad 替代 retain_grad()
    )

    # 把 means2d 存入 gaussians，strategy.step 通过 means2d.absgrad 读取梯度累积量，
    # 不再需要 retain_grad()，计算图可在 backward 后正常释放。
    if means.requires_grad:
        gaussians["viewspace_points"] = info["means2d"]

    return render_colors


def quaternion_to_rotation_matrix(q):
    """四元数 [N, 4] (w, x, y, z) → 旋转矩阵 [N, 3, 3]"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y,
        2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
        2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y,
    ], dim=-1).reshape(-1, 3, 3)
    return R


def compute_2d_covariance(means3D, scales, rotations, w2c, K):
    """
    3D 椭球 → 2D 椭圆协方差
    Σ_2D = J @ W @ Σ_3D @ W^T @ J^T
    返回：cov2D_inv [N, 2, 2], det [N], means_cam [N, 3]
    """
    N = means3D.shape[0]
    device = means3D.device

    R = quaternion_to_rotation_matrix(rotations)
    S = torch.diag_embed(scales)
    RS = R @ S
    cov3D = RS @ RS.transpose(-1, -2)

    W = w2c[:3, :3]
    means_homo = torch.cat([means3D, torch.ones(N, 1, device=device)], dim=-1)
    means_cam = (w2c @ means_homo.t()).t()[:, :3]

    fx, fy = K[0, 0], K[1, 1]
    tx, ty, tz = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    tz_sq = tz * tz + 1e-8

    zeros = torch.zeros_like(tz)
    J = torch.stack([
        fx / tz, zeros,   -fx * tx / tz_sq,
        zeros,   fy / tz, -fy * ty / tz_sq,
    ], dim=-1).reshape(N, 2, 3)

    JW = J @ W.unsqueeze(0)
    cov2D = JW @ cov3D @ JW.transpose(-1, -2)

    # 低通滤波防奇异
    cov2D[:, 0, 0] += 0.3
    cov2D[:, 1, 1] += 0.3

    a, b, c, d = cov2D[:, 0, 0], cov2D[:, 0, 1], cov2D[:, 1, 0], cov2D[:, 1, 1]
    det = (a * d - b * c).clamp(min=1e-8)

    cov2D_inv = torch.stack([d, -b, -c, a], dim=-1).reshape(N, 2, 2) / det.unsqueeze(-1).unsqueeze(-1)

    return cov2D_inv, det, means_cam


def mps_rasterizer(gaussians, w2c, K, H, W, tile_size=64):
    """
    MPS/CPU 加速版光栅化器（Tile-based 向量化，低显存）
    核心思路：按 tile 分块渲染，每次只处理一个 tile 的像素，
    避免 [B, H*W] 全图矩阵——H*W=26万像素时一次 batch 要 5GB。
    """
    device = gaussians["means"].device
    means3D = gaussians["means"]

    # 1. 2D 协方差投影
    cov2D_inv, _, means_cam = compute_2d_covariance(
        means3D, gaussians["scales"], gaussians["rotations"], w2c, K
    )
    depths = means_cam[:, 2:3]

    # 2. Near clipping
    mask = depths.squeeze(-1) > 0.1
    if not mask.any():
        return torch.zeros((H, W, 3), device=device) + means3D.sum() * 0

    # 3. 投影到像素
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mc = means_cam[mask]
    points_2d = torch.stack([
        fx * mc[:, 0] / mc[:, 2] + cx,
        fy * mc[:, 1] / mc[:, 2] + cy,
    ], dim=-1)

    colors    = gaussians["colors"][mask]    # [N, 3]
    opacities = gaussians["opacity"][mask]   # [N, 1]
    cov_inv   = cov2D_inv[mask]             # [N, 2, 2]

    # 4. 等效半径 + 视锥剔除
    r_a = 1.0 / (cov_inv[:, 0, 0].sqrt() + 1e-4)
    r_d = 1.0 / (cov_inv[:, 1, 1].sqrt() + 1e-4)
    max_radius = 3.0 * torch.maximum(r_a, r_d)

    in_frame = (
        (points_2d[:, 0] + max_radius > 0) & (points_2d[:, 0] - max_radius < W) &
        (points_2d[:, 1] + max_radius > 0) & (points_2d[:, 1] - max_radius < H)
    )
    if not in_frame.any():
        return torch.zeros((H, W, 3), device=device) + means3D.sum() * 0

    points_2d  = points_2d[in_frame]
    colors     = colors[in_frame]
    opacities  = opacities[in_frame]
    cov_inv    = cov_inv[in_frame]
    max_radius = max_radius[in_frame]
    depths_vis = depths[mask][in_frame]

    # 5. 按深度排序
    sort_idx   = torch.argsort(depths_vis.squeeze(-1))
    points_2d  = points_2d[sort_idx]
    colors     = colors[sort_idx]
    opacities  = opacities[sort_idx]
    cov_inv    = cov_inv[sort_idx]
    max_radius = max_radius[sort_idx]
    M = points_2d.shape[0]

    # 6. 预计算 tile 归属（向量化，全在 GPU 完成，无 Python 循环）
    n_tiles_h = (H + tile_size - 1) // tile_size
    n_tiles_w = (W + tile_size - 1) // tile_size

    pts_x = points_2d[:, 0]
    pts_y = points_2d[:, 1]
    tile_x_min = ((pts_x - max_radius) / tile_size).floor().clamp(0, n_tiles_w - 1).int()
    tile_x_max = ((pts_x + max_radius) / tile_size).floor().clamp(0, n_tiles_w - 1).int()
    tile_y_min = ((pts_y - max_radius) / tile_size).floor().clamp(0, n_tiles_h - 1).int()
    tile_y_max = ((pts_y + max_radius) / tile_size).floor().clamp(0, n_tiles_h - 1).int()

    # CPU numpy 仅用于 tile 索引构建（一次性，开销很小）
    tx_min = tile_x_min.cpu().numpy()
    tx_max = tile_x_max.cpu().numpy()
    ty_min = tile_y_min.cpu().numpy()
    ty_max = tile_y_max.cpu().numpy()

    tile_point_lists = [[[] for _ in range(n_tiles_w)] for _ in range(n_tiles_h)]
    for i in range(M):
        for ty in range(ty_min[i], ty_max[i] + 1):
            for tx in range(tx_min[i], tx_max[i] + 1):
                tile_point_lists[ty][tx].append(i)

    # 7. Tile-based 渲染：每次只处理一个 tile 的像素，显存 ∝ tile_size²，可控
    output = torch.zeros((H, W, 3), device=device)
    max_per_tile = 2000

    for ty in range(n_tiles_h):
        for tx in range(n_tiles_w):
            pt_indices = tile_point_lists[ty][tx]
            if not pt_indices:
                continue

            y_start = ty * tile_size
            y_end   = min(y_start + tile_size, H)
            x_start = tx * tile_size
            x_end   = min(x_start + tile_size, W)
            tile_h  = y_end - y_start
            tile_w  = x_end - x_start
            n_pixels = tile_h * tile_w  # tile_size²，最大约 4096（64²）

            idx_t    = torch.tensor(pt_indices[:max_per_tile], device=device, dtype=torch.long)
            t_pts    = points_2d.index_select(0, idx_t)   # [M_tile, 2]
            t_colors = colors.index_select(0, idx_t)       # [M_tile, 3]
            t_opac   = opacities.index_select(0, idx_t)    # [M_tile, 1]
            t_cov    = cov_inv.index_select(0, idx_t)      # [M_tile, 2, 2]
            M_tile   = idx_t.shape[0]

            # 像素坐标 [n_pixels, 2]，只有 tile 内的像素
            gy, gx = torch.meshgrid(
                torch.arange(y_start, y_end, device=device, dtype=torch.float32),
                torch.arange(x_start, x_end, device=device, dtype=torch.float32),
                indexing='ij'
            )
            tile_coords = torch.stack([gx + 0.5, gy + 0.5], dim=-1).reshape(1, -1, 2)

            # 关键：delta 形状是 [M_tile, n_pixels, 2]
            # M_tile ≤ 2000，n_pixels ≤ tile_size² = 4096（tile_size=64）
            # 最大中间张量：2000 × 4096 × 4B = 32 MB，完全可控
            delta = t_pts.unsqueeze(1) - tile_coords
            dx = delta[:, :, 0]
            dy = delta[:, :, 1]
            ci_a = t_cov[:, 0, 0].unsqueeze(1)
            ci_b = t_cov[:, 0, 1].unsqueeze(1)
            ci_d = t_cov[:, 1, 1].unsqueeze(1)
            power = -0.5 * (dx*dx*ci_a + 2*dx*dy*ci_b + dy*dy*ci_d)
            g_weight = torch.exp(power.clamp(max=0)) * (power > -4.5).float()

            alphas = (t_opac * g_weight).clamp(max=0.99)  # [M_tile, n_pixels]
            one_minus = 1.0 - alphas
            transmittance = torch.cumprod(
                torch.cat([torch.ones((1, n_pixels), device=device), one_minus[:-1] + 1e-6], dim=0),
                dim=0
            )  # [M_tile, n_pixels]
            weights = alphas * transmittance  # [M_tile, n_pixels]
            tile_colors = torch.sum(weights.unsqueeze(-1) * t_colors.unsqueeze(1), dim=0)
            output[y_start:y_end, x_start:x_end] = tile_colors.view(tile_h, tile_w, 3)

    return output


def simple_rasterizer(gaussians, w2c, K, H, W, camera_pos=None, tile_size=128):
    """
    Tile-based 各向异性可微栅格化器
    优化：预计算 tile-point 归属表，避免每个 tile 重复做全局 mask
    """
    device = gaussians["means"].device
    means3D = gaussians["means"]

    # 1. 2D 协方差投影（对所有点一次性计算）
    cov2D_inv, cov2D_det, means_cam = compute_2d_covariance(
        means3D, gaussians["scales"], gaussians["rotations"], w2c, K
    )
    depths = means_cam[:, 2:3]

    # 2. Near clipping
    mask = depths.squeeze(-1) > 0.1
    if not mask.any():
        return torch.zeros((H, W, 3), device=device) + means3D.sum() * 0

    # 3. 投影到像素
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mc = means_cam[mask]
    points_2d = torch.stack([
        fx * mc[:, 0] / mc[:, 2] + cx,
        fy * mc[:, 1] / mc[:, 2] + cy,
    ], dim=-1)

    colors = gaussians["colors"][mask]
    opacities = gaussians["opacity"][mask]
    cov_inv = cov2D_inv[mask]

    # 4. 等效半径估算（椭圆最大半轴的 3σ）
    a_inv = cov_inv[:, 0, 0]  # [N_vis]
    d_inv = cov_inv[:, 1, 1]  # [N_vis]
    r_a = 1.0 / (a_inv.sqrt() + 1e-4)
    r_d = 1.0 / (d_inv.sqrt() + 1e-4)
    max_radius = 3.0 * torch.maximum(r_a, r_d)  # element-wise max，不会对标量出错

    # 5. 视锥剔除：只保留覆盖范围与画面有交集的点
    in_frame = (
        (points_2d[:, 0] + max_radius > 0) & (points_2d[:, 0] - max_radius < W) &
        (points_2d[:, 1] + max_radius > 0) & (points_2d[:, 1] - max_radius < H)
    )
    if not in_frame.any():
        return torch.zeros((H, W, 3), device=device) + means3D.sum() * 0

    points_2d = points_2d[in_frame]
    colors = colors[in_frame]
    opacities = opacities[in_frame]
    cov_inv = cov_inv[in_frame]
    max_radius = max_radius[in_frame]
    depths_vis = depths[mask][in_frame]

    # 6. 按深度排序（全局排一次）
    sort_idx = torch.argsort(depths_vis.squeeze(-1), descending=False)
    points_2d = points_2d[sort_idx]
    colors = colors[sort_idx]
    opacities = opacities[sort_idx]
    cov_inv = cov_inv[sort_idx]
    max_radius = max_radius[sort_idx]

    N_pts = len(points_2d)

    # 7. 预计算 tile 归属：为每个 tile 找出影响它的点索引
    n_tiles_h = (H + tile_size - 1) // tile_size
    n_tiles_w = (W + tile_size - 1) // tile_size

    # tile 边界张量 [n_tiles_h, n_tiles_w, 4] → (x_start, y_start, x_end, y_end)
    tile_indices = []
    pts_x = points_2d[:, 0]
    pts_y = points_2d[:, 1]
    pts_x_min = pts_x - max_radius  # [N]
    pts_x_max = pts_x + max_radius
    pts_y_min = pts_y - max_radius
    pts_y_max = pts_y + max_radius

    # 每个点覆盖的 tile 范围
    tile_x_min = ((pts_x_min / tile_size).floor().clamp(min=0, max=n_tiles_w - 1)).int()
    tile_x_max = ((pts_x_max / tile_size).floor().clamp(min=0, max=n_tiles_w - 1)).int()
    tile_y_min = ((pts_y_min / tile_size).floor().clamp(min=0, max=n_tiles_h - 1)).int()
    tile_y_max = ((pts_y_max / tile_size).floor().clamp(min=0, max=n_tiles_h - 1)).int()

    # 构建每个 tile 的点索引列表
    tile_point_lists = [[[] for _ in range(n_tiles_w)] for _ in range(n_tiles_h)]
    # 向量化构建：遍历点（比遍历 tile 更高效，因为 tile 数量远大于每个点的覆盖 tile 数）
    tx_min_cpu = tile_x_min.cpu().numpy()
    tx_max_cpu = tile_x_max.cpu().numpy()
    ty_min_cpu = tile_y_min.cpu().numpy()
    ty_max_cpu = tile_y_max.cpu().numpy()

    for i in range(N_pts):
        for ty in range(ty_min_cpu[i], ty_max_cpu[i] + 1):
            for tx in range(tx_min_cpu[i], tx_max_cpu[i] + 1):
                tile_point_lists[ty][tx].append(i)

    # 8. 分块渲染（使用预计算的索引）
    output = torch.zeros((H, W, 3), device=device)
    max_per_tile = 2000

    for ty in range(n_tiles_h):
        for tx in range(n_tiles_w):
            pt_indices = tile_point_lists[ty][tx]
            if not pt_indices:
                continue

            y_start = ty * tile_size
            y_end = min(y_start + tile_size, H)
            x_start = tx * tile_size
            x_end = min(x_start + tile_size, W)
            tile_h = y_end - y_start
            tile_w = x_end - x_start
            n_pixels = tile_h * tile_w

            # 取出该 tile 的点（已按深度排好序）
            # 用列表索引而非单个值索引，保证结果始终是 2D+
            idx_t = torch.tensor(pt_indices[:max_per_tile], device=device, dtype=torch.long)
            t_pts = points_2d.index_select(0, idx_t)     # [M, 2]
            t_colors = colors.index_select(0, idx_t)     # [M, 3]
            t_opac = opacities.index_select(0, idx_t)    # [M, 1]
            t_cov_inv = cov_inv.index_select(0, idx_t)   # [M, 2, 2]
            M = len(idx_t)

            # 像素坐标
            gy, gx = torch.meshgrid(
                torch.arange(y_start, y_end, device=device, dtype=torch.float32),
                torch.arange(x_start, x_end, device=device, dtype=torch.float32),
                indexing='ij'
            )
            tile_coords = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2) + 0.5

            # 各向异性高斯求值
            delta = t_pts.unsqueeze(1) - tile_coords   # [M, P, 2]
            # power = -0.5 * δ^T Σ^{-1} δ
            # 展开: power = -0.5 * (δx² * a + 2*δx*δy * b + δy² * d) 其中 cov_inv = [[a,b],[c,d]]
            dx = delta[:, :, 0]  # [M, P]
            dy = delta[:, :, 1]
            ci_a = t_cov_inv[:, 0, 0].unsqueeze(1)  # [M, 1]
            ci_b = t_cov_inv[:, 0, 1].unsqueeze(1)
            ci_d = t_cov_inv[:, 1, 1].unsqueeze(1)
            power = -0.5 * (dx * dx * ci_a + 2 * dx * dy * ci_b + dy * dy * ci_d)  # [M, P]

            # 3σ 截断
            valid = power > -4.5
            gaussian_weight = torch.exp(power) * valid.float()

            # Alpha 合成
            alphas = (t_opac * gaussian_weight).unsqueeze(-1).clamp(max=0.99)  # [M, P, 1]

            one_minus_alpha = 1.0 - alphas
            transmittance = torch.cumprod(
                torch.cat([torch.ones((1, n_pixels, 1), device=device), one_minus_alpha + 1e-6], dim=0),
                dim=0
            )[:-1]

            weights = alphas * transmittance
            tile_colors = torch.sum(weights * t_colors.unsqueeze(1), dim=0)

            output[y_start:y_end, x_start:x_end] = tile_colors.view(tile_h, tile_w, 3)

    return output
