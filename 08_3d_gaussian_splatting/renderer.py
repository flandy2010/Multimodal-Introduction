import torch


def simple_rasterizer(gaussians, w2c, K, H, W, camera_pos=None, tile_size=64):
    """
    分块（Tile-based）可微栅格化器
    核心优化：将图像分为 tile_size x tile_size 的块，每块只处理影响该块的高斯点
    显存复杂度从 O(N * H * W) 降为 O(N_tile * tile_size^2)
    """
    device = gaussians["means"].device

    # 1. 投影 3D -> 2D (Camera Space)
    means3D = gaussians["means"]
    means3D_homo = torch.cat([means3D, torch.ones_like(means3D[..., :1])], dim=-1)
    points_cam = (w2c @ means3D_homo.t()).t()
    depths = points_cam[:, 2:3]  # COLMAP 坐标系看向 +Z

    # 过滤相机背后的点 (Near Clipping)
    mask = depths.squeeze() > 0.1
    if not mask.any():
        return torch.zeros((H, W, 3), device=device) + gaussians["means"].sum() * 0

    # 投影到像素平面 (u, v)
    points_2d = (K[:2, :2] @ (points_cam[mask, :2] / depths[mask]).t()).t() + K[:2, 2]

    # 提取属性
    colors = gaussians["colors"][mask]
    opacities = gaussians["opacity"][mask]
    scales = gaussians["scales"][mask]

    # 计算 2D 投影半径
    focal_avg = (K[0, 0] + K[1, 1]) / 2.0
    radii2D = (scales.mean(dim=-1, keepdim=True) * focal_avg) / depths[mask]  # [N_vis, 1]

    # 视锥剔除：过滤完全不在画面内的点
    margin = radii2D.squeeze() * 3.0
    in_frame = (
        (points_2d[:, 0] > -margin) & (points_2d[:, 0] < W + margin) &
        (points_2d[:, 1] > -margin) & (points_2d[:, 1] < H + margin)
    )

    if not in_frame.any():
        return torch.zeros((H, W, 3), device=device) + gaussians["means"].sum() * 0

    points_2d = points_2d[in_frame]
    colors = colors[in_frame]
    opacities = opacities[in_frame]
    radii2D = radii2D[in_frame]
    depths_visible = depths[mask][in_frame]

    # 按深度排序
    sort_indices = torch.argsort(depths_visible.squeeze(), descending=False)
    points_2d = points_2d[sort_indices]
    colors = colors[sort_indices]
    opacities = opacities[sort_indices]
    radii2D = radii2D[sort_indices]

    # --- 分块渲染 ---
    output = torch.zeros((H, W, 3), device=device)
    n_tiles_h = (H + tile_size - 1) // tile_size
    n_tiles_w = (W + tile_size - 1) // tile_size

    for ty in range(n_tiles_h):
        for tx in range(n_tiles_w):
            # 当前 tile 的像素范围
            y_start = ty * tile_size
            y_end = min(y_start + tile_size, H)
            x_start = tx * tile_size
            x_end = min(x_start + tile_size, W)

            tile_h = y_end - y_start
            tile_w = x_end - x_start

            # 找出影响当前 tile 的高斯点（中心 ± 3σ 覆盖到 tile 范围）
            r3 = radii2D.squeeze() * 3.0  # 3σ 范围
            tile_mask = (
                (points_2d[:, 0] + r3 > x_start) & (points_2d[:, 0] - r3 < x_end) &
                (points_2d[:, 1] + r3 > y_start) & (points_2d[:, 1] - r3 < y_end)
            )

            if not tile_mask.any():
                continue

            # 取出当前 tile 相关的点（已按深度排好序）
            t_pts = points_2d[tile_mask]      # [M, 2]
            t_colors = colors[tile_mask]      # [M, 3]
            t_opac = opacities[tile_mask]     # [M, 1]
            t_radii = radii2D[tile_mask]      # [M, 1]

            # 限制单 tile 内点数，防止极端情况
            max_per_tile = 4000
            if len(t_pts) > max_per_tile:
                t_pts = t_pts[:max_per_tile]
                t_colors = t_colors[:max_per_tile]
                t_opac = t_opac[:max_per_tile]
                t_radii = t_radii[:max_per_tile]

            # 构造 tile 内的像素坐标网格
            gy, gx = torch.meshgrid(
                torch.arange(y_start, y_end, device=device, dtype=torch.float32),
                torch.arange(x_start, x_end, device=device, dtype=torch.float32),
                indexing='ij'
            )
            tile_coords = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2) + 0.5  # [1, tile_h*tile_w, 2]
            n_pixels = tile_h * tile_w

            # 计算距离平方 [M, n_pixels]
            d2 = torch.sum((t_pts.unsqueeze(1) - tile_coords) ** 2, dim=-1)

            # 3σ 截断
            cutoff = (3.0 * t_radii) ** 2  # [M, 1]
            valid = d2 <= cutoff

            # 高斯权重
            gaussian_weight = torch.exp(-d2 / (2 * t_radii ** 2 + 1e-5))
            gaussian_weight = gaussian_weight * valid.float()

            # Alpha
            alphas = (t_opac * gaussian_weight).unsqueeze(-1)  # [M, n_pixels, 1]
            alphas = alphas.clamp(max=0.99)

            # Alpha 合成 (Front-to-Back)
            one_minus_alpha = 1.0 - alphas
            transmittance = torch.cumprod(
                torch.cat([torch.ones((1, n_pixels, 1), device=device), one_minus_alpha + 1e-6], dim=0),
                dim=0
            )[:-1]

            weights = alphas * transmittance  # [M, n_pixels, 1]
            tile_colors = torch.sum(weights * t_colors.unsqueeze(1), dim=0)  # [n_pixels, 3]

            # 写回输出
            output[y_start:y_end, x_start:x_end] = tile_colors.view(tile_h, tile_w, 3)

    return output
