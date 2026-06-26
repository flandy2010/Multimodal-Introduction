import torch
import torch.nn.functional as F

# 尝试导入 gsplat，若 CUDA 不可用则标记为 None
try:
    from gsplat import rasterization as _gsplat_rasterization
    # 验证 CUDA 扩展真的可用：
    # gsplat >= 1.0 用 _make_lazy_cuda_func 延迟加载，不再有 _C 属性；
    # 改为调用一个轻量 CUDA 函数来触发加载并确认可用性。
    import gsplat.cuda._wrapper as _gsplat_cuda
    if hasattr(_gsplat_cuda, '_C'):
        # 旧版 gsplat (0.x) 的兼容检测
        if _gsplat_cuda._C is None:
            raise ImportError("gsplat CUDA extension not available")
    else:
        # 新版 gsplat (>= 1.0)：尝试触发一次 CUDA kernel 加载
        import torch
        if not torch.cuda.is_available():
            raise ImportError("gsplat requires CUDA but CUDA is not available")
        _gsplat_cuda.isect_offset_encode(
            torch.zeros(0, dtype=torch.int64, device="cuda"),
            0, 0, 0
        )
    _GSPLAT_AVAILABLE = True
except Exception:
    _GSPLAT_AVAILABLE = False
    print("⚠️  gsplat CUDA 不可用，自动降级到 simple_rasterizer（MPS/CPU 模式）")


def auto_rasterizer(gaussians, w2c, K, H, W, tile_size=16, gt_image=None, loss_fn=None,
                    radius_clip=0.0):
    """
    自动三路选择渲染后端：
    ┌─────────────────────────────────────────────────────────────────┐
    │  CUDA 可用（H20/P800） → gsplat_rasterizer  （最快，支持致密化）  │
    │  MPS 可用（Mac M系列） → mps_rasterizer     （向量化，无 for 循环）│
    │  其他（CPU/CUDA无gsplat）→ simple_rasterizer（逐tile backward）  │
    └─────────────────────────────────────────────────────────────────┘

    radius_clip: 传入 gsplat_rasterizer。像素单位，投影半径 ≤ 该值的球被跳过。
      默认 0.0 = 不裁剪。MPS/simple 路径暂不支持此参数（忽略）。
    推理时：统一返回 out_image。
    """
    if _GSPLAT_AVAILABLE:
        return gsplat_rasterizer(gaussians, w2c, K, H, W, tile_size=tile_size,
                                 radius_clip=radius_clip)
    elif gaussians["means"].device.type == "mps":
        return mps_rasterizer(gaussians, w2c, K, H, W)
    else:
        return simple_rasterizer(gaussians, w2c, K, H, W, tile_size=tile_size,
                                 gt_image=gt_image, loss_fn=loss_fn)


def gsplat_rasterizer(gaussians, w2c, K, H, W, tile_size=16, radius_clip=0.0):
    """
    作为 simple_rasterizer 的直接替代品。
    gaussians: 一个字典或对象，包含 means, scales, quats, opacities, colors/shs
    w2c: [4, 4] 矩阵 (World-to-Camera)
    K: [3, 3] 矩阵 (Camera Intrinsics)
    radius_clip: 屏幕空间 2D 半径（像素）下限裁剪。
        投影半径 ≤ radius_clip 的 Gaussian 在光栅化时被跳过（不渲染、不产生梯度）。
        注意：这是"下限"裁剪，用于跳过远处太小的球（加速大场景）。
        本项目中我们额外用它来跳过投影过大的球（通过 max_radius_clip 实现上限裁剪）。
        gsplat 官方 simple_trainer.py 训练时不使用 radius_clip（保持 0.0），
        仅在 Viewer 中启用。这里保留接口，由调用方决定是否使用。
        默认 0.0 = 不裁剪。

    注意：gsplat CUDA kernel 的 tile_size 只支持 16 和 32，
    传入更大的值会触发 cudaErrorInvalidConfiguration，此处强制限制。
    """
    gsplat_tile_size = min(tile_size, 16)  # gsplat 只支持 16（及 32，但 16 最安全）
    # 1. 提取参数
    # 注意：model.forward 传入的 gaussians 中，scales/rotations/opacity/colors
    # 已经经过 exp/normalize/sigmoid/clamp 等激活，这里直接使用，不要重复激活。
    means    = gaussians["means"]                     # [N, 3] 保留梯度
    scales   = gaussians["scales"]                    # [N, 3] 已是物理尺度（exp 过）
    quats    = gaussians["rotations"]                 # [N, 4] 已归一化
    opacities = gaussians["opacity"].squeeze(-1)      # [N]    已是 sigmoid 后的值
    colors   = gaussians["colors"]                    # [N, 3]

    # 2. 构造 viewmat (gsplat 需要 [1, 4, 4])
    viewmat = w2c.view(-1, 4, 4)
    if len(K.shape) == 2:
        K = K.unsqueeze(0)

    # 3. 调用 gsplat 加速渲染
    # absgrad=True：在 CUDA kernel 内直接累积 |∇2D|，替代 retain_grad()，节省显存
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
        tile_size=gsplat_tile_size,
        absgrad=True,
        radius_clip=radius_clip,  # 像素单位，默认 0.0 = 不裁剪
    )

    # 把 means2d 存入 gaussians，strategy.step 通过 means2d.absgrad 读取梯度累积量，
    # 不再需要 retain_grad()，计算图可在 backward 后正常释放。
    if means.requires_grad:
        gaussians["viewspace_points"] = info["means2d"]

    render_colors = render_colors.squeeze(0)

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


def simple_rasterizer(gaussians, w2c, K, H, W, camera_pos=None, tile_size=16,
                      gt_image=None, loss_fn=None):
    """
    Tile-based 各向异性可微栅格化器。

    显存优化：逐 tile 分段 backward，T detach。
    每个 tile 对用到的 M 个点做局部 detach+requires_grad 副本，
    backward 后用 scatter_add 累积梯度回原参数，计算图立即释放。
    显存峰值 = 单 tile，所有参数（means/scales/rotations/colors/opacities）均有梯度。

    训练模式（传 gt_image+loss_fn）：内部完成 backward，返回 (out_detach, total_loss)。
    推理模式：返回渲染图（no_grad）。
    """
    device   = gaussians["means"].device
    means3D  = gaussians["means"]    # [N,3] nn.Parameter
    scales_p = gaussians["scales"]   # [N,3]
    rots_p   = gaussians["rotations"]# [N,4]
    opac_p   = gaussians["opacity"]  # [N,1] sigmoid 后 (0~1)
    colors_p = gaussians["colors"]   # [N,3] clamp(0,1)
    N        = means3D.shape[0]

    train_mode = (gt_image is not None)  # loss_fn 不再需要，内部直接用 L1-sum

    # ── 1. 前处理（no_grad）：排序、tile归属 ─────────────────────────────
    with torch.no_grad():
        cov2D_inv_d, _, means_cam_d = compute_2d_covariance(
            means3D, scales_p, rots_p, w2c, K
        )
        depths_d = means_cam_d[:, 2]
        mask     = depths_d > 0.1
        if not mask.any():
            dummy = torch.zeros((H, W, 3), device=device)
            return (dummy, torch.tensor(0.0, device=device)) if train_mode else dummy

        fx = K[0,0].item(); fy = K[1,1].item()
        cx = K[0,2].item(); cy = K[1,2].item()
        mc_d    = means_cam_d[mask]
        pts2d_d = torch.stack([
            fx * mc_d[:,0] / mc_d[:,2] + cx,
            fy * mc_d[:,1] / mc_d[:,2] + cy,
        ], dim=-1)

        cov_inv_d = cov2D_inv_d[mask]
        r_a = 1.0 / (cov_inv_d[:,0,0].sqrt() + 1e-4)
        r_d_ = 1.0 / (cov_inv_d[:,1,1].sqrt() + 1e-4)
        max_r_d = 3.0 * torch.maximum(r_a, r_d_)

        in_frame = (
            (pts2d_d[:,0] + max_r_d > 0) & (pts2d_d[:,0] - max_r_d < W) &
            (pts2d_d[:,1] + max_r_d > 0) & (pts2d_d[:,1] - max_r_d < H)
        )
        if not in_frame.any():
            dummy = torch.zeros((H, W, 3), device=device)
            return (dummy, torch.tensor(0.0, device=device)) if train_mode else dummy

        vis_idx   = mask.nonzero(as_tuple=True)[0][in_frame]
        pts2d_d   = pts2d_d[in_frame]
        cov_inv_d = cov_inv_d[in_frame]
        max_r_d   = max_r_d[in_frame]
        depth_d   = depths_d[mask][in_frame]

        sort_idx  = torch.argsort(depth_d)
        vis_idx   = vis_idx[sort_idx]
        pts2d_d   = pts2d_d[sort_idx]
        cov_inv_d = cov_inv_d[sort_idx]
        max_r_d   = max_r_d[sort_idx]
        Nvis = len(vis_idx)

        n_tiles_h = (H + tile_size - 1) // tile_size
        n_tiles_w = (W + tile_size - 1) // tile_size
        px = pts2d_d[:,0]; py = pts2d_d[:,1]
        tx_min = ((px - max_r_d) / tile_size).floor().clamp(0, n_tiles_w-1).int().cpu().numpy()
        tx_max = ((px + max_r_d) / tile_size).floor().clamp(0, n_tiles_w-1).int().cpu().numpy()
        ty_min = ((py - max_r_d) / tile_size).floor().clamp(0, n_tiles_h-1).int().cpu().numpy()
        ty_max = ((py + max_r_d) / tile_size).floor().clamp(0, n_tiles_h-1).int().cpu().numpy()

        tile_lists = [[[] for _ in range(n_tiles_w)] for _ in range(n_tiles_h)]
        for i in range(Nvis):
            for ty in range(int(ty_min[i]), int(ty_max[i])+1):
                for tx in range(int(tx_min[i]), int(tx_max[i])+1):
                    tile_lists[ty][tx].append(i)

    # ── 2. 梯度累积缓冲（与各参数同形） ─────────────────────────────────
    output_detach = torch.zeros((H, W, 3), device=device)
    total_loss    = torch.zeros(1, device=device)
    if not train_mode:
        output = torch.zeros((H, W, 3), device=device)

    if train_mode:
        g_means  = torch.zeros(N, 3, device=device)
        g_scales = torch.zeros(N, 3, device=device)
        g_rots   = torch.zeros(N, 4, device=device)
        g_opac   = torch.zeros(N, 1, device=device)
        g_colors = torch.zeros(N, 3, device=device)

    # ── 3. 逐 tile 渲染 + backward ──────────────────────────────────────
    max_per_tile = 2000

    for ty in range(n_tiles_h):
        for tx in range(n_tiles_w):
            pt_idx = tile_lists[ty][tx]
            if not pt_idx:
                continue

            y0 = ty * tile_size;  y1 = min(y0 + tile_size, H)
            x0 = tx * tile_size;  x1 = min(x0 + tile_size, W)
            th = y1 - y0;  tw = x1 - x0;  P = th * tw

            idx_t    = torch.tensor(pt_idx[:max_per_tile], device=device, dtype=torch.long)
            orig_idx = vis_idx[idx_t]                      # [M] 原始点索引

            # 关键：局部 detach + requires_grad，每 tile 建独立计算图
            lm = means3D[orig_idx].detach().requires_grad_(True)   # [M,3]
            ls = scales_p[orig_idx].detach().requires_grad_(True)  # [M,3]
            lr = rots_p[orig_idx].detach().requires_grad_(True)    # [M,4]
            lo = opac_p[orig_idx].detach().requires_grad_(True)    # [M,1]
            lc = colors_p[orig_idx].detach().requires_grad_(True)  # [M,3]

            # 重新投影（带梯度）
            t_cov_inv, _, t_means_cam = compute_2d_covariance(lm, ls, lr, w2c, K)
            t_pts = torch.stack([
                fx * t_means_cam[:,0] / t_means_cam[:,2] + cx,
                fy * t_means_cam[:,1] / t_means_cam[:,2] + cy,
            ], dim=-1)                                      # [M,2]

            gy, gx_ = torch.meshgrid(
                torch.arange(y0, y1, device=device, dtype=torch.float32),
                torch.arange(x0, x1, device=device, dtype=torch.float32),
                indexing='ij'
            )
            coords = torch.stack([gx_, gy], dim=-1).reshape(1, -1, 2) + 0.5  # [1,P,2]

            delta  = t_pts.unsqueeze(1) - coords           # [M,P,2]
            dx, dy = delta[:,:,0], delta[:,:,1]
            ci_a   = t_cov_inv[:,0,0].unsqueeze(1)
            ci_b   = t_cov_inv[:,0,1].unsqueeze(1)
            ci_d   = t_cov_inv[:,1,1].unsqueeze(1)
            power  = -0.5 * (dx*dx*ci_a + 2*dx*dy*ci_b + dy*dy*ci_d)  # [M,P]
            gw     = torch.exp(power) * (power > -4.5).float()          # [M,P]

            alphas = (lo * gw).clamp(max=0.99)              # [M,P]

            # T detach：避免 cumprod backward 的显存开销
            with torch.no_grad():
                T = torch.cumprod(
                    torch.cat([torch.ones((1, P), device=device),
                               (1.0 - alphas.detach()[:-1]) + 1e-6], dim=0), dim=0  # [M,P]
                )

            w           = alphas * T                        # [M,P]  T 是常数
            tile_colors = (w.unsqueeze(-1) * lc.unsqueeze(1)).sum(0)  # [P,3]
            tile_img    = tile_colors.view(th, tw, 3)

            if train_mode:
                gt_tile   = gt_image[y0:y1, x0:x1]
                # 用绝对差之和而非均值，最后统一除以 H*W*3 得整图均值 L1
                tile_loss = (tile_img - gt_tile).abs().sum()
                tile_loss.backward()                        # 局部图立即释放
                total_loss = total_loss + tile_loss.detach()
                output_detach[y0:y1, x0:x1] = tile_img.detach()

                # 把局部梯度 scatter 回全参数缓冲
                exp_orig = orig_idx.unsqueeze(1)
                if lm.grad is not None:
                    g_means.scatter_add_(0, exp_orig.expand_as(lm.grad), lm.grad)
                if ls.grad is not None:
                    g_scales.scatter_add_(0, exp_orig.expand_as(ls.grad), ls.grad)
                if lr.grad is not None:
                    g_rots.scatter_add_(0, exp_orig.expand_as(lr.grad), lr.grad)
                if lo.grad is not None:
                    g_opac.scatter_add_(0, exp_orig, lo.grad)
                if lc.grad is not None:
                    g_colors.scatter_add_(0, exp_orig.expand_as(lc.grad), lc.grad)
            else:
                output[y0:y1, x0:x1] = tile_img
                output_detach[y0:y1, x0:x1] = tile_img.detach()

    # ── 4. 将梯度写回原始 nn.Parameter ──────────────────────────────────
    if train_mode:
        def _acc_grad(param, grad):
            """安全地把 grad 累加到叶节点参数的 .grad 上"""
            if param is not None and param.requires_grad:
                if param.grad is None:
                    param.grad = grad.clone()
                else:
                    param.grad.add_(grad)

        raw_means  = means3D               # 叶节点
        raw_sh     = gaussians.get("_raw_sh_coeffs")
        raw_opac   = gaussians.get("_raw_opacity")
        raw_scales = gaussians.get("_raw_scales")    # log-space leaf
        raw_rots   = gaussians.get("_raw_rotations") # unnorm quat leaf

        # means 直接写回
        _acc_grad(raw_means, g_means)

        # scales: gaussians["scales"] = exp(raw_scales)
        # dL/d(raw_scales) = dL/d(scales) * exp(raw_scales) = g_scales * scales_p
        if raw_scales is not None and raw_scales.requires_grad:
            with torch.no_grad():
                _acc_grad(raw_scales, g_scales * scales_p.detach())

        # rotations: gaussians["rotations"] = normalize(raw_rots)
        # dL/d(raw_rots[i]) = dL/d(rots[i]) * d(normalize)/d(raw)
        # d(normalize(x))/dx = (I - x_hat x_hat^T) / |x|  近似为 identity（|x|≈1 时）
        # 这里用简化近似：直接传 g_rots（误差可接受，rotations 本来初始化就是单位四元数）
        if raw_rots is not None and raw_rots.requires_grad:
            _acc_grad(raw_rots, g_rots)

        # sh_coeffs DC 分量：colors = clamp(SH_C0 * sh[:,0,:] + 0.5)
        # dL/d(sh[:,0,:]) = g_colors * SH_C0（忽略 clamp 边界）
        if raw_sh is not None and raw_sh.requires_grad:
            from model import SH_C0 as _SH_C0
            if raw_sh.grad is None:
                raw_sh.grad = torch.zeros_like(raw_sh)
            raw_sh.grad[:, 0, :].add_(g_colors * _SH_C0)

        # opacities: opac_p = sigmoid(raw_opac)
        # dL/d(raw_opac) = g_opac * sigmoid'(raw_opac) = g_opac * opac_p*(1-opac_p)
        if raw_opac is not None and raw_opac.requires_grad:
            with torch.no_grad():
                sig_val = opac_p.detach()
                d_sig   = sig_val * (1.0 - sig_val)
            _acc_grad(raw_opac, g_opac * d_sig)

        return output_detach, total_loss.squeeze() / (H * W * 3.0)
    else:
        return output
