import torch


def simple_rasterizer(gaussians, w2c, K, H, W):
    """
    向量化可微栅格化器（含视锥剔除 + 3σ截断优化）
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

    # 计算 2D 投影半径（用 3σ 作为有效覆盖范围）
    focal_avg = (K[0, 0] + K[1, 1]) / 2.0
    radii2D = (scales.mean(dim=-1, keepdim=True) * focal_avg) / depths[mask]

    # 视锥剔除：只保留投影点在画面扩展范围内的点
    margin = radii2D.squeeze() * 3.0  # 3σ
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

    # --- 排序与混合 (Front-to-Back) ---
    indices = torch.argsort(depths_visible.squeeze(), descending=False)

    # 性能平衡点：渲染点数上限
    max_points = min(10000, len(indices))
    render_idx = indices[:max_points]

    means2D_sel = points_2d[render_idx]
    colors_sel = colors[render_idx]
    opacities_sel = opacities[render_idx]
    radii2D_sel = radii2D[render_idx]

    # 构造坐标网格
    gy, gx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixel_coords = torch.stack([gx, gy], dim=-1).float() + 0.5
    pixel_coords = pixel_coords.view(1, H * W, 2)

    # 计算所有像素到高斯中心的平方距离
    d2 = torch.sum((means2D_sel.unsqueeze(1) - pixel_coords) ** 2, dim=-1)

    # 3σ 截断：超过 3 倍半径的贡献直接置零，减少远距离点的影响（也是光晕的根因）
    cutoff = (3.0 * radii2D_sel) ** 2
    truncation_mask = d2 > cutoff  # [N, H*W]

    # 计算高斯权重: exp(-d^2 / 2r^2)，用 where 避免 inplace 操作
    raw_weight = torch.exp(-d2 / (2 * radii2D_sel ** 2 + 1e-5))
    gaussian_weight = torch.where(truncation_mask, torch.zeros_like(raw_weight), raw_weight)

    alphas = (opacities_sel * gaussian_weight).unsqueeze(-1)  # [N, Pixels, 1]
    alphas = alphas.clamp(max=0.99)  # 防止单点完全遮挡

    # Alpha 合成 (Over Operator)
    one_minus_alpha = 1.0 - alphas
    transmittance = torch.cumprod(
        torch.cat([torch.ones((1, H * W, 1), device=device), one_minus_alpha + 1e-6], dim=0),
        dim=0
    )[:-1]

    # 最终颜色 = sum (T_i * alpha_i * color_i)
    weights = alphas * transmittance
    pixel_colors = torch.sum(weights * colors_sel.unsqueeze(1), dim=0)

    return pixel_colors.view(H, W, 3)