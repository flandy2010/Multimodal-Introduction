import torch


def quaternion_to_rotation_matrix(q):
    """
    四元数 [N, 4] (w, x, y, z) → 旋转矩阵 [N, 3, 3]
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y,
        2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
        2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y,
    ], dim=-1).reshape(-1, 3, 3)
    return R


def compute_2d_covariance(means3D, scales, rotations, w2c, K):
    """
    将 3D 椭球投影为 2D 椭圆（协方差矩阵）
    
    原理：Σ_2D = J @ W @ Σ_3D @ W^T @ J^T
    其中 Σ_3D = R @ S @ S @ R^T, J 是投影雅可比
    
    返回：cov2D_inv [N, 2, 2], det [N]（用于高斯求值）
    """
    N = means3D.shape[0]
    device = means3D.device

    # 1. 构造 3D 协方差: Σ = R @ diag(s^2) @ R^T
    R = quaternion_to_rotation_matrix(rotations)  # [N, 3, 3]
    S = torch.diag_embed(scales)                   # [N, 3, 3]
    RS = R @ S                                      # [N, 3, 3]
    cov3D = RS @ RS.transpose(-1, -2)               # [N, 3, 3]

    # 2. 变换到相机坐标系
    W = w2c[:3, :3]  # [3, 3] 旋转部分
    means_homo = torch.cat([means3D, torch.ones(N, 1, device=device)], dim=-1)
    means_cam = (w2c @ means_homo.t()).t()[:, :3]  # [N, 3]

    # 3. 投影雅可比 J (针对针孔相机)
    fx, fy = K[0, 0], K[1, 1]
    tx, ty, tz = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    tz_sq = tz * tz + 1e-8

    # J = [[fx/tz, 0, -fx*tx/tz^2],
    #      [0, fy/tz, -fy*ty/tz^2]]
    zeros = torch.zeros_like(tz)
    J = torch.stack([
        fx / tz, zeros,   -fx * tx / tz_sq,
        zeros,   fy / tz, -fy * ty / tz_sq,
    ], dim=-1).reshape(N, 2, 3)

    # 4. Σ_2D = J @ W @ Σ_3D @ W^T @ J^T
    JW = J @ W.unsqueeze(0)                 # [N, 2, 3]
    cov2D = JW @ cov3D @ JW.transpose(-1, -2)  # [N, 2, 2]

    # 5. 加一个小的各向同性项防止奇异（低通滤波）
    cov2D[:, 0, 0] += 0.3
    cov2D[:, 1, 1] += 0.3

    # 6. 计算逆和行列式
    a, b, c, d = cov2D[:, 0, 0], cov2D[:, 0, 1], cov2D[:, 1, 0], cov2D[:, 1, 1]
    det = a * d - b * c + 1e-8
    det = det.clamp(min=1e-8)

    cov2D_inv = torch.stack([d, -b, -c, a], dim=-1).reshape(N, 2, 2) / det.unsqueeze(-1).unsqueeze(-1)

    return cov2D_inv, det, means_cam


def simple_rasterizer(gaussians, w2c, K, H, W, camera_pos=None, tile_size=64):
    """
    分块（Tile-based）可微栅格化器 - 各向异性版本
    使用 2D 协方差矩阵渲染椭圆形高斯，支持 rotation 参数
    """
    device = gaussians["means"].device
    means3D = gaussians["means"]

    # 1. 计算 2D 协方差（椭圆投影）
    cov2D_inv, cov2D_det, means_cam = compute_2d_covariance(
        means3D, gaussians["scales"], gaussians["rotations"], w2c, K
    )

    depths = means_cam[:, 2:3]  # COLMAP: +Z 是前方

    # 2. 过滤相机背后的点
    mask = depths.squeeze() > 0.1
    if not mask.any():
        return torch.zeros((H, W, 3), device=device) + means3D.sum() * 0

    # 3. 投影到像素平面
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    points_2d_x = fx * means_cam[mask, 0] / means_cam[mask, 2] + cx
    points_2d_y = fy * means_cam[mask, 1] / means_cam[mask, 2] + cy
    points_2d = torch.stack([points_2d_x, points_2d_y], dim=-1)  # [N_vis, 2]

    # 提取可见点属性
    colors = gaussians["colors"][mask]
    opacities = gaussians["opacity"][mask]
    cov_inv = cov2D_inv[mask]     # [N_vis, 2, 2]
    det = cov2D_det[mask]         # [N_vis]

    # 计算等效半径（椭圆的最大半轴，用于视锥剔除和 3σ 截断）
    # 从协方差矩阵的特征值估算
    a = cov_inv[:, 0, 0]
    d = cov_inv[:, 1, 1]
    # 最大半径 ≈ 3 / sqrt(min eigenvalue of inv) ≈ 3 * sqrt(max eigenvalue of cov)
    # 近似用 max(1/sqrt(a), 1/sqrt(d))
    max_radius = 3.0 * torch.max(1.0 / (a.sqrt() + 1e-4), 1.0 / (d.sqrt() + 1e-4))  # [N_vis]

    # 4. 视锥剔除
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

    # 5. 按深度排序
    sort_idx = torch.argsort(depths_vis.squeeze(), descending=False)
    points_2d = points_2d[sort_idx]
    colors = colors[sort_idx]
    opacities = opacities[sort_idx]
    cov_inv = cov_inv[sort_idx]
    max_radius = max_radius[sort_idx]

    # 6. 分块渲染
    output = torch.zeros((H, W, 3), device=device)
    n_tiles_h = (H + tile_size - 1) // tile_size
    n_tiles_w = (W + tile_size - 1) // tile_size

    for ty in range(n_tiles_h):
        for tx in range(n_tiles_w):
            y_start = ty * tile_size
            y_end = min(y_start + tile_size, H)
            x_start = tx * tile_size
            x_end = min(x_start + tile_size, W)
            tile_h = y_end - y_start
            tile_w = x_end - x_start

            # 找出影响当前 tile 的点
            tile_mask = (
                (points_2d[:, 0] + max_radius > x_start) & (points_2d[:, 0] - max_radius < x_end) &
                (points_2d[:, 1] + max_radius > y_start) & (points_2d[:, 1] - max_radius < y_end)
            )

            if not tile_mask.any():
                continue

            t_pts = points_2d[tile_mask]
            t_colors = colors[tile_mask]
            t_opac = opacities[tile_mask]
            t_cov_inv = cov_inv[tile_mask]      # [M, 2, 2]
            t_max_r = max_radius[tile_mask]

            # 限制单 tile 内点数
            max_per_tile = 4000
            if len(t_pts) > max_per_tile:
                t_pts = t_pts[:max_per_tile]
                t_colors = t_colors[:max_per_tile]
                t_opac = t_opac[:max_per_tile]
                t_cov_inv = t_cov_inv[:max_per_tile]
                t_max_r = t_max_r[:max_per_tile]

            # 像素坐标网格
            gy, gx = torch.meshgrid(
                torch.arange(y_start, y_end, device=device, dtype=torch.float32),
                torch.arange(x_start, x_end, device=device, dtype=torch.float32),
                indexing='ij'
            )
            tile_coords = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2) + 0.5  # [1, P, 2]
            n_pixels = tile_h * tile_w

            # 计算偏移 [M, P, 2]
            delta = t_pts.unsqueeze(1) - tile_coords  # [M, P, 2]

            # 各向异性高斯：power = -0.5 * delta^T @ Sigma_inv @ delta
            # delta: [M, P, 2] → [M, P, 2, 1]
            delta_col = delta.unsqueeze(-1)
            # t_cov_inv: [M, 2, 2] → [M, 1, 2, 2]
            cov_inv_exp = t_cov_inv.unsqueeze(1)
            # [M, P, 2, 1] = [M, 1, 2, 2] @ [M, P, 2, 1]
            tmp = (cov_inv_exp @ delta_col).squeeze(-1)  # [M, P, 2]
            power = -0.5 * (delta.unsqueeze(-2) @ tmp.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # [M, P]

            # 3σ 截断：用 power < -4.5 近似（对应 3σ 处 exp(-4.5) ≈ 0.011）
            valid = power > -4.5
            gaussian_weight = torch.exp(power) * valid.float()

            # Alpha
            alphas = (t_opac * gaussian_weight).unsqueeze(-1)  # [M, P, 1]
            alphas = alphas.clamp(max=0.99)

            # Alpha 合成 (Front-to-Back)
            one_minus_alpha = 1.0 - alphas
            transmittance = torch.cumprod(
                torch.cat([torch.ones((1, n_pixels, 1), device=device), one_minus_alpha + 1e-6], dim=0),
                dim=0
            )[:-1]

            weights = alphas * transmittance
            tile_colors = torch.sum(weights * t_colors.unsqueeze(1), dim=0)

            output[y_start:y_end, x_start:x_end] = tile_colors.view(tile_h, tile_w, 3)

    return output
