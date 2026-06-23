import torch


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
