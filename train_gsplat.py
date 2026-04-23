"""
3D Gaussian Splatting Scene Optimization from Panorama LDI

Initializes a 3DGS scene from panorama Layered Depth Images and optimizes it
to match perspective views extracted from the panorama.

Based on paper description:
- 240 evenly rotated cameras from center of projection
- Per-layer optimization + composite optimization (simultaneous)
- L1 + SSIM + L2 depth loss
- Standard 3DGS densification and pruning

Usage:
    python train_gsplat.py --ldi_dir output_ldi --output scene.ply
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import cv2
import kornia
from kornia.utils import create_meshgrid

from gsplat import rasterization, DefaultStrategy
from utils.depth_utilsv2 import estimate_depth, estimate_depth_moge


def fov2focal(fov_radians, pixels):
    """Convert field of view to focal length."""
    return pixels / (2 * math.tan(fov_radians / 2))


def cyl_proj(img, f):
    """
    Apply cylindrical projection to convert panorama crop to perspective view.
    
    Args:
        img: [B, C, H, W] tensor in panorama space
        f: Focal length
    
    Returns:
        Perspective image [B, C, H, W]
    """
    temp = create_meshgrid(img.shape[2], img.shape[3], normalized_coordinates=False, device=img.device)
    y, x = temp[..., 0], temp[..., 1]
    h, w = img.shape[2:]
    center_x = w // 2
    center_y = h // 2
    
    x_shifted = (x - center_x)
    y_shifted = (y - center_y)
    
    theta = torch.arctan(x_shifted / f)
    height = y_shifted / torch.sqrt(x_shifted ** 2 + f ** 2)
    
    x_cyl = (f * theta + center_x)
    y_cyl = (height * f + center_y)
    
    img_cyl = kornia.geometry.transform.remap(img, torch.flip(x_cyl, dims=(1, 2)), y_cyl, mode='nearest', align_corners=True)
    img_cyl = torch.rot90(img_cyl, k=3, dims=(2, 3))
    return img_cyl


def ssim(img1, img2, window_size=11):
    """Compute SSIM between two images."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.tensor([math.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                          for x in range(window_size)], device=img1.device, dtype=img1.dtype)
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(img1.shape[-1], 1, window_size, window_size).contiguous()
    
    # Permute to [B, C, H, W]
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0).permute(0, 3, 1, 2)
        img2 = img2.unsqueeze(0).permute(0, 3, 1, 2)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def load_panorama_ldi(ldi_dir):
    """
    Load panorama LDI layers from directory.
    
    Returns:
        rgba_pano: [num_layers, H, W, 4] numpy array (panorama, duplicated for wrapping)
        depth_pano: [num_layers, H, W] numpy array (panorama, duplicated for wrapping)
        original_width: Original panorama width before duplication
    """
    rgba_path = os.path.join(ldi_dir, 'rgba_ldi.npy')
    depth_path = os.path.join(ldi_dir, 'depth_ldi.npy')
    
    print(f'[INFO] Loading panorama LDI from {ldi_dir}')
    rgba_pano = np.load(rgba_path)
    depth_pano = np.load(depth_path)
    
    original_width = rgba_pano.shape[2]
    num_layers = rgba_pano.shape[0]
    
    print(f'[INFO] Panorama LDI shape: {rgba_pano.shape}, depth: {depth_pano.shape}, layers: {num_layers}')
    
    # Triple horizontally so any 512-wide crop centered in the middle copy is valid
    rgba_pano = np.concatenate([rgba_pano, rgba_pano, rgba_pano], axis=2)[::-1].copy()
    depth_pano = np.concatenate([depth_pano, depth_pano, depth_pano], axis=2)[::-1].copy()
    
    return rgba_pano, depth_pano, original_width, num_layers


def extract_perspective_views(rgba_pano, depth_pano, original_width, num_views=240, fov_deg=44.702, view_size=512, device='cuda'):
    """
    Extract perspective views from panorama LDI by cropping and applying cylindrical projection.
    
    Returns:
        views: List of dicts with 'rgba_layers', 'depth_layers', 'theta'
               rgba_layers: [num_layers, H, W, 4] - per-layer RGBA
               depth_layers: [num_layers, H, W] - per-layer depth
    """
    num_layers, h, w = rgba_pano.shape[:3]
    focal = fov2focal(fov_deg * math.pi / 180, view_size)

    views = []

    print(f'[INFO] Extracting {num_views} perspective views from panorama...')
    for view_idx in tqdm(range(num_views), desc="Extracting views"):
        # Rotation angle for this view
        theta = view_idx * (2 * np.pi) / num_views

        # Crop center in the middle copy of the tripled panorama
        center = int(round(original_width + theta * original_width / (2 * math.pi)))
        start = center - view_size // 2
        end = center + view_size // 2
        
        crop_rgba = rgba_pano[:, :, start:end]  # [num_layers, H, W, 4]
        crop_depth = depth_pano[:, :, start:end]  # [num_layers, H, W]
        
        # Apply cylindrical projection to each layer
        rgba_tensor = torch.tensor(crop_rgba, device=device).permute(0, 3, 1, 2).float()
        depth_tensor = torch.tensor(crop_depth, device=device).unsqueeze(1).float()
        
        rgba_persp = cyl_proj(rgba_tensor, focal)
        depth_persp = cyl_proj(depth_tensor, focal)
        
        rgba_persp = rgba_persp.permute(0, 2, 3, 1).cpu().numpy()
        depth_persp = depth_persp.squeeze(1).cpu().numpy()
        
        views.append({
            'rgba_layers': rgba_persp,  # [num_layers, H, W, 4]
            'depth_layers': depth_persp,  # [num_layers, H, W]
            'theta': theta,
            'focal': focal
        })
    
    return views


def prepare_training_targets(views, device='cuda', depth_max=200):
    """
    Prepare per-layer and composite training targets for each view.
    
    Following reference: layer j is composite of layers 0..j where alpha > 0
    
    Returns:
        targets: List of dicts with:
            - 'layer_images': [num_layers, H, W, 3] - composited images up to each layer
            - 'layer_depths': [num_layers, H, W] - composited depths up to each layer  
            - 'layer_masks': [num_layers, H, W] - valid pixel masks
            - 'composite_image': [H, W, 3] - final composite
            - 'composite_depth': [H, W] - final composite depth
            - 'theta': rotation angle
            - 'focal': focal length
    """
    print('[INFO] Preparing per-layer and composite training targets...')
    
    targets = []
    for view in tqdm(views, desc="Preparing targets"):
        rgba_layers = view['rgba_layers']  # [num_layers, H, W, 4]
        depth_layers = np.clip(view['depth_layers'], 0, depth_max)
        num_layers, H, W, _ = rgba_layers.shape
        
        # Prepare per-layer composited targets (following reference logic)
        layer_images = np.zeros((num_layers, H, W, 4), dtype=np.float32)
        layer_depths = np.zeros((num_layers, H, W), dtype=np.float32)
        
        for j in range(num_layers):
            if j == 0:
                layer_images[j] = rgba_layers[j]
                layer_depths[j] = depth_layers[j]
            else:
                # Where current layer has alpha > 0, use it, else use previous
                mask = rgba_layers[j, ..., 3:4] > 0
                layer_images[j] = np.where(mask, rgba_layers[j], layer_images[j-1])
                layer_depths[j] = np.where(mask[..., 0], depth_layers[j], layer_depths[j-1])
        
        # Create masks for each layer
        layer_masks = layer_images[..., 3] > 0  # [num_layers, H, W]
        
        # Final composite is the last layer
        composite_image = layer_images[-1, ..., :3]
        composite_depth = layer_depths[-1]
        
        targets.append({
            'layer_images': torch.tensor(layer_images[..., :3], dtype=torch.float32, device=device),
            'layer_depths': torch.tensor(layer_depths, dtype=torch.float32, device=device),
            'layer_masks': torch.tensor(layer_masks, dtype=torch.float32, device=device),
            'composite_image': torch.tensor(composite_image, dtype=torch.float32, device=device),
            'composite_depth': torch.tensor(composite_depth, dtype=torch.float32, device=device),
            'theta': view['theta'],
            'focal': view['focal']
        })
    
    return targets


def unproject_panorama_to_points(rgba_ldi, depth_ldi, init_opacity=0.05, depth_max=200, device='cuda'):
    """
    Directly unproject panorama LDI to 3D point cloud using cylindrical geometry.

    Each panorama pixel maps to a unique 3D direction, giving smooth azimuthal
    distribution instead of the polygon artifacts from per-view unprojection.

    Args:
        rgba_ldi: [num_layers, H, W, 4] raw panorama LDI (before duplication/reversal)
        depth_ldi: [num_layers, H, W] raw panorama depth
        device: torch device

    Returns:
        points: [N, 3] tensor of 3D positions
        colors: [N, 3] tensor of RGB colors
        alphas: [N] tensor of opacity values
    """
    num_layers, H, W = depth_ldi.shape
    depth_ldi = np.clip(depth_ldi, 0, depth_max)
    f_pano = W / (2 * math.pi)

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    theta = 2 * math.pi * u / W

    all_points = []
    all_colors = []
    all_alphas = []
    all_depths = []
    all_layer_ids = []

    for layer_idx in range(num_layers):
        rgba = rgba_ldi[layer_idx]
        depth = depth_ldi[layer_idx]

        alpha = rgba[..., 3]
        valid = alpha > 0.1

        if valid.sum() == 0:
            continue

        n_valid = valid.sum()
        d = depth[valid]
        rgb = rgba[..., :3][valid]
        a = alpha[valid] * init_opacity
        t = theta[valid]
        v_valid = v[valid]

        X = d * np.sin(t)
        Z = d * np.cos(t)
        Y = (v_valid - H / 2) * d / f_pano

        all_points.append(np.stack([X, Y, Z], axis=-1))
        all_colors.append(rgb)
        all_alphas.append(a)
        all_depths.append(d)
        # Reverse layer order to match load_panorama_ldi's [::-1] reversal
        reversed_id = (num_layers - 1) - layer_idx
        all_layer_ids.append(np.full(n_valid, reversed_id, dtype=np.int64))

    if len(all_points) == 0:
        return None, None, None, None, None

    return (
        torch.tensor(np.concatenate(all_points), dtype=torch.float32, device=device),
        torch.tensor(np.concatenate(all_colors), dtype=torch.float32, device=device),
        torch.tensor(np.concatenate(all_alphas), dtype=torch.float32, device=device),
        torch.tensor(np.concatenate(all_depths), dtype=torch.float32, device=device),
        torch.tensor(np.concatenate(all_layer_ids), dtype=torch.int64, device=device),
    )


class GSParams:
    """Gaussian Splatting optimization parameters (from reference arguments.py)."""
    def __init__(self):
        self.iterations = 3000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 3000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 500
        self.densify_from_iter = 500
        self.densify_until_iter = 2500
        self.densify_grad_threshold = 0.0002


def initialize_gaussians(points, colors, alphas, depths=None, pano_width=None,
                         scale_mult=2.0, layer_ids=None, device='cuda'):
    """Initialize Gaussian parameters from point cloud."""
    N = points.shape[0]

    means = nn.Parameter(points.clone())
    colors_param = nn.Parameter(colors.clone())

    if depths is not None and pano_width is not None:
        f_pano = pano_width / (2 * math.pi)
        per_point_scale = scale_mult * depths / f_pano
        scales = nn.Parameter(torch.log(per_point_scale).unsqueeze(-1).expand(-1, 3).clone())
    else:
        init_scale = 0.02
        scales = nn.Parameter(torch.full((N, 3), math.log(init_scale), device=device))
    
    quats = nn.Parameter(torch.zeros((N, 4), device=device))
    quats.data[:, 0] = 1.0
    
    opacities = nn.Parameter(torch.logit(alphas.clamp(0.01, 0.99)))
    
    print(f'[INFO] Initialized {N:,} Gaussians')

    gaussians = {
        'means': means,
        'scales': scales,
        'quats': quats,
        'opacities': opacities,
        'colors': colors_param,
    }
    if layer_ids is not None:
        gaussians['layer_ids'] = layer_ids
    return gaussians


def create_camera(theta, focal=582.69, H=512, W=512, device='cuda'):
    """Create a camera at origin, rotated by theta around Y-axis."""
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = np.array([0, 0, 0])
    
    w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)
    
    K = torch.tensor([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    return {'w2c': w2c, 'K': K, 'H': H, 'W': W, 'theta': theta}


def create_orbit_camera(theta, radius=3.0, focal=582.69, H=512, W=512, device='cuda'):
    """Create orbit camera: positioned at radius from origin, looking inward."""
    eye = np.array([radius * np.cos(theta), 0, radius * np.sin(theta)])
    up = np.array([0, 1, 0])
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye

    w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)
    K = torch.tensor([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)

    return {'w2c': w2c, 'K': K, 'H': H, 'W': W, 'theta': theta}


def pearson_depth_loss(rendered_depth, mono_depth):
    """Pearson correlation loss between rendered and monocular depth (Zhu et al. 2024)."""
    rd = rendered_depth.reshape(-1, 1)
    md = mono_depth.reshape(-1, 1)
    valid = (rd > 0).squeeze() & (md > 0).squeeze()
    if valid.sum() < 10:
        return torch.tensor(0.0, device=rendered_depth.device)
    rd = rd[valid]
    md = md[valid]
    vx = rd - rd.mean()
    vy = md - md.mean()
    corr = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum() * (vy ** 2).sum()) + 1e-8)
    return 1.0 - corr


def render_gaussians(gaussians, camera, background=None, max_layer=None):
    """Render Gaussians from a camera view.

    Args:
        max_layer: If set, physically select only Gaussians from layers 0..max_layer
                   and pass that subset to the rasterizer (no opacity zeroing).
    """
    if background is None:
        background = torch.ones(3, device=gaussians['means'].device)

    means = gaussians['means']
    scales = torch.exp(gaussians['scales'])
    quats = F.normalize(gaussians['quats'], dim=-1)
    opacities = torch.sigmoid(gaussians['opacities'])
    colors = gaussians['colors']

    if max_layer is not None and 'layer_ids' in gaussians:
        mask = gaussians['layer_ids'] <= max_layer
        means = means[mask]
        scales = scales[mask]
        quats = quats[mask]
        opacities = opacities[mask]
        colors = colors[mask]

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=camera['w2c'][None],
        Ks=camera['K'][None],
        width=camera['W'],
        height=camera['H'],
        packed=False,
        render_mode="RGB+ED",
        backgrounds=background[None],
        rasterize_mode="antialiased",
        absgrad=True,
    )

    rgb = render_colors[0, ..., :3]
    depth = render_colors[0, ..., 3]
    alpha = render_alphas[0, ..., 0]

    return rgb, depth, alpha, info


def render_360_video(gaussians, targets, output_path, num_frames=60, fps=30, device='cuda'):
    """
    Render a 360 video from the current Gaussian state.
    
    Args:
        gaussians: Gaussian parameters dict (or None for GT-only)
        targets: Training targets with theta and focal
        output_path: Output video path
        num_frames: Number of frames in video
        fps: Frames per second
        device: Device
    """
    import imageio
    
    background = torch.ones(3, device=device)
    
    # Create evenly spaced cameras for smooth 360 rotation
    frames = []
    
    # Use focal from first target
    focal = targets[0]['focal']
    H = targets[0]['composite_image'].shape[0]
    W = targets[0]['composite_image'].shape[1]
    
    for i in range(num_frames):
        theta = i * (2 * np.pi) / num_frames
        camera = create_camera(theta, focal, H, W, device)
        
        with torch.no_grad():
            rgb, _, _, _ = render_gaussians(gaussians, camera, background)
        
        rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
        frames.append(rgb_np)
    
    # Write video
    imageio.mimwrite(output_path, frames, fps=fps)
    print(f'[INFO] Saved 360 video: {output_path}')


def render_gt_360_video(targets, output_path, num_frames=60, fps=30):
    """
    Render a 360 video from ground truth targets.
    
    Interpolates between available training views to create smooth video.
    """
    import imageio
    
    frames = []
    num_targets = len(targets)
    
    for i in range(num_frames):
        # Map frame index to target index
        target_idx = int(i * num_targets / num_frames) % num_targets
        
        gt = targets[target_idx]['composite_image']
        gt_np = (gt.cpu().numpy() * 255).astype(np.uint8)
        frames.append(gt_np)
    
    imageio.mimwrite(output_path, frames, fps=fps)
    print(f'[INFO] Saved GT 360 video: {output_path}')


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Copied from original 3DGS - exponential learning rate scheduler.
    """
    def helper(step):
        if lr_init == lr_final:
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return lr_init
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * min(step / lr_delay_steps, 1.0) if lr_delay_steps > 0 else 1.0
        t = min(step / max_steps, 1.0)
        log_lerp = math.exp(math.log(lr_init) * (1 - t) + math.log(max(lr_final, 1e-10)) * t)
        return delay_rate * log_lerp
    return helper


def post_training_prune(params, layer_ids=None, threshold=0.01):
    """Remove near-transparent Gaussians after training."""
    keep = torch.sigmoid(params['opacities']) > threshold
    n_before = params['means'].shape[0]
    pruned = {}
    for k, v in params.items():
        pruned[k] = nn.Parameter(v.data[keep].clone())
    result = {'params': pruned, 'n_removed': n_before - keep.sum().item()}
    if layer_ids is not None:
        result['layer_ids'] = layer_ids[keep].clone()
    return result


def compute_loss(rendered_rgb, target_rgb, rendered_depth, target_depth, mask=None, lambda_dssim=0.2, depth_weight=0.005):
    """
    Compute 3DGS loss: (1-lambda)*L1 + lambda*SSIM + depth_weight*L2_depth
    """
    if mask is not None:
        mask_rgb = mask.unsqueeze(-1).expand_as(rendered_rgb)
        rendered_rgb = rendered_rgb * mask_rgb
        target_rgb = target_rgb * mask_rgb
        rendered_depth = rendered_depth * mask
        target_depth = target_depth * mask
    
    # L1 loss
    l1_loss = F.l1_loss(rendered_rgb, target_rgb)
    
    # SSIM loss
    ssim_val = ssim(rendered_rgb, target_rgb)
    ssim_loss = 1.0 - ssim_val
    
    # Combined RGB loss
    rgb_loss = (1.0 - lambda_dssim) * l1_loss + lambda_dssim * ssim_loss
    
    # Depth L2 loss
    if mask is not None:
        valid = mask > 0
        if valid.sum() > 0:
            depth_loss = F.mse_loss(rendered_depth[valid], target_depth[valid])
        else:
            depth_loss = torch.tensor(0.0, device=rendered_rgb.device)
    else:
        depth_loss = F.mse_loss(rendered_depth, target_depth)
    
    total_loss = rgb_loss + depth_weight * depth_loss
    
    return total_loss, l1_loss, ssim_loss, depth_loss


def train(gaussians, targets, num_iterations=3000, debug_dir=None, video_interval=500,
          freeze_positions=False, depth_weight=0.005,
          novel_view_weight=0.5, novel_view_radius=3.0, novel_view_start=500,
          novel_view_every=10, novel_view_model='dav2', depth_max=200):
    """
    Optimize Gaussians following paper approach.

    Args:
        video_interval: Render 360 video every N iterations (0 to disable)
        freeze_positions: If True, don't optimize Gaussian positions
        novel_view_weight: Weight for novel view depth loss (Zhu et al. 2024)
        novel_view_radius: Orbit radius for novel view cameras
        novel_view_start: Iteration to start novel view depth loss
        novel_view_every: Run novel view loss every N iterations (reduces cost)
        novel_view_model: 'dav2' or 'moge' for monocular depth estimation
    """
    device = gaussians['means'].device
    num_layers = targets[0]['layer_images'].shape[0]

    opt = GSParams()
    opt.iterations = num_iterations

    position_lr_fn = get_expon_lr_func(
        lr_init=opt.position_lr_init,
        lr_final=opt.position_lr_final,
        lr_delay_mult=opt.position_lr_delay_mult,
        max_steps=opt.position_lr_max_steps
    )

    # Separate layer_ids from trainable params (strategy manages params only)
    layer_ids = gaussians.pop('layer_ids', None)
    params = gaussians  # now only trainable nn.Parameters

    # Per-parameter optimizers (required by DefaultStrategy)
    optimizers = {
        'scales': torch.optim.Adam([params['scales']], lr=opt.scaling_lr, eps=1e-15),
        'quats': torch.optim.Adam([params['quats']], lr=opt.rotation_lr, eps=1e-15),
        'opacities': torch.optim.Adam([params['opacities']], lr=opt.opacity_lr, eps=1e-15),
        'colors': torch.optim.Adam([params['colors']], lr=opt.feature_lr, eps=1e-15),
    }
    if not freeze_positions:
        optimizers['means'] = torch.optim.Adam([params['means']], lr=opt.position_lr_init, eps=1e-15)

    # gsplat DefaultStrategy handles split/clone/prune/opacity-reset
    scene_extent = params['means'].detach().abs().max().item() * 1.1
    strategy = DefaultStrategy(
        absgrad=True,
        grow_grad2d=0.0008,
        refine_start_iter=opt.densify_from_iter,
        refine_stop_iter=opt.densify_until_iter,
        refine_every=opt.densification_interval,
        reset_every=1000,
        verbose=True,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_extent)
    if layer_ids is not None:
        strategy_state['layer_ids'] = layer_ids

    # Wrap params/layer_ids access for convenience
    def get_layer_ids():
        return strategy_state.get('layer_ids', None)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for i in range(min(4, len(targets))):
            gt = targets[i]['composite_image']
            gt_np = (gt.cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(debug_dir, f'gt_view_{i:02d}.png'), gt_np[:, :, ::-1])
        print('[INFO] Rendering GT 360 video...')
        render_gt_360_video(targets, os.path.join(debug_dir, 'gt_360.mp4'),
                           num_frames=len(targets), fps=30)

    print(f'[INFO] Starting optimization for {num_iterations} iterations')
    print(f'[INFO] Num views: {len(targets)}, Num layers: {num_layers}')
    print(f'[INFO] Initial Gaussians: {params["means"].shape[0]:,}')
    print(f'[INFO] Densification (DefaultStrategy): iter {opt.densify_from_iter}–{opt.densify_until_iter}')
    if freeze_positions:
        print(f'[INFO] Positions FROZEN — optimizing color/opacity/scale only')
    if novel_view_weight > 0:
        print(f'[INFO] Novel view depth loss: weight={novel_view_weight}, model={novel_view_model}, '
              f'radius={novel_view_radius}, start={novel_view_start}, every={novel_view_every}')

    # Build a gaussians-like dict for render_gaussians (which expects 'layer_ids' inside)
    def make_render_dict():
        d = dict(params)
        lid = get_layer_ids()
        if lid is not None:
            d['layer_ids'] = lid
        return d

    pbar = tqdm(range(1, num_iterations + 1), desc="Training")
    for iteration in pbar:
        # Update position LR
        if not freeze_positions and 'means' in optimizers:
            optimizers['means'].param_groups[0]['lr'] = position_lr_fn(iteration)

        view_idx = np.random.randint(len(targets))
        target = targets[view_idx]
        camera = create_camera(target['theta'], target['focal'],
                               target['layer_images'].shape[1],
                               target['layer_images'].shape[2], device)
        background = torch.rand(3, device=device)

        render_dict = make_render_dict()

        # === Per-layer loss ===
        random_layer = np.random.randint(num_layers)
        layer_rgb, layer_depth, _, _ = render_gaussians(
            render_dict, camera, background, max_layer=random_layer)
        layer_mask = target['layer_masks'][random_layer]
        loss_layer, _, _, _ = compute_loss(
            layer_rgb, target['layer_images'][random_layer],
            layer_depth, target['layer_depths'][random_layer],
            mask=layer_mask, lambda_dssim=opt.lambda_dssim,
            depth_weight=depth_weight
        )

        # === Composite loss (info used by strategy) ===
        rendered_rgb, rendered_depth, alpha, comp_info = render_gaussians(
            render_dict, camera, background)
        loss_comp, _, _, _ = compute_loss(
            rendered_rgb, target['composite_image'],
            rendered_depth, target['composite_depth'],
            mask=None, lambda_dssim=opt.lambda_dssim,
            depth_weight=depth_weight
        )

        loss = loss_layer + loss_comp

        # === Novel view depth loss (Zhu et al. 2024) ===
        if novel_view_weight > 0 and iteration >= novel_view_start and iteration % novel_view_every == 0:
            nv_theta = np.random.uniform(0, 2 * np.pi)
            nv_camera = create_orbit_camera(
                nv_theta, radius=novel_view_radius,
                focal=target['focal'], H=target['layer_images'].shape[1],
                W=target['layer_images'].shape[2], device=device)
            nv_rgb, nv_depth, _, _ = render_gaussians(render_dict, nv_camera, background)
            nv_rgb_np = (nv_rgb.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            if novel_view_model == 'moge':
                mono = estimate_depth_moge(nv_rgb_np)
            else:
                mono = estimate_depth(nv_rgb_np)
            mono_t = torch.tensor(mono, dtype=torch.float32, device=device)
            nv_loss = pearson_depth_loss(nv_depth, mono_t)
            loss = loss + novel_view_weight * nv_loss

        # Strategy pre-backward (retains grad on means2d for densification)
        strategy.step_pre_backward(params, optimizers, strategy_state, iteration, comp_info)

        loss.backward()

        # Strategy post-backward (split/clone/prune/opacity-reset)
        strategy.step_post_backward(params, optimizers, strategy_state, iteration, comp_info)

        # Guard against Gaussian explosion
        if params['means'].shape[0] > 5_000_000:
            strategy.refine_stop_iter = iteration
            tqdm.write(f'[WARN] Gaussian cap reached ({params["means"].shape[0]:,}), stopping refinement')

        # Optimizer step
        for opt_obj in optimizers.values():
            opt_obj.step()
            opt_obj.zero_grad(set_to_none=True)

        if iteration % 10 == 0:
            n_gauss = params['means'].shape[0]
            pbar.set_description(f"Training (loss: {loss.item():.4f}, N: {n_gauss//1000}k)")

        if debug_dir and iteration % 500 == 0:
            with torch.no_grad():
                for i in [0, len(targets)//4, len(targets)//2, 3*len(targets)//4]:
                    i = i % len(targets)
                    t = targets[i]
                    cam = create_camera(t['theta'], t['focal'],
                                        t['layer_images'].shape[1],
                                        t['layer_images'].shape[2], device)
                    rgb, depth, _, _ = render_gaussians(render_dict, cam, background)
                    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
                    gt_np = (t['composite_image'].cpu().numpy() * 255).astype(np.uint8)
                    comparison = np.hstack([gt_np, rgb_np])
                    cv2.imwrite(os.path.join(debug_dir, f'compare_iter_{iteration:04d}_view_{i:02d}.png'), comparison[:, :, ::-1])

        if debug_dir and video_interval > 0 and iteration % video_interval == 0:
            tqdm.write(f'[INFO] Rendering 360 video at iteration {iteration}...')
            render_360_video(render_dict, targets,
                           os.path.join(debug_dir, f'iter_{iteration:04d}_360.mp4'),
                           num_frames=60, fps=30, device=device)

    if debug_dir:
        print('[INFO] Rendering final 360 video...')
        render_360_video(make_render_dict(), targets,
                        os.path.join(debug_dir, 'final_360.mp4'),
                        num_frames=60, fps=30, device=device)

    # Post-training pruning
    with torch.no_grad():
        result = post_training_prune(params, get_layer_ids())
        params = result['params']
        print(f'[INFO] Post-training prune: {result["n_removed"]:,} removed (opacity < 0.01)')

    # Reconstruct gaussians dict for return
    gaussians = dict(params)
    if 'layer_ids' in result:
        gaussians['layer_ids'] = result['layer_ids']

    print(f'[INFO] Optimization complete')
    print(f'[INFO] Final Gaussians: {gaussians["means"].shape[0]:,}')

    return gaussians


def save_gaussians_ply(gaussians, output_path):
    """Save Gaussians to PLY file."""
    from plyfile import PlyData, PlyElement
    
    means = gaussians['means'].detach().cpu().numpy()
    scales = gaussians['scales'].detach().cpu().numpy()
    quats = gaussians['quats'].detach().cpu().numpy()
    opacities = gaussians['opacities'].detach().cpu().numpy()
    colors = gaussians['colors'].detach().cpu().numpy()
    
    # Convert colors to SH DC component
    SH_C0 = 0.28209479177387814
    f_dc = (colors - 0.5) / SH_C0
    
    N = means.shape[0]
    normals = np.zeros_like(means)
    
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]
    
    elements = np.empty(N, dtype=dtype)
    elements['x'] = means[:, 0]
    elements['y'] = means[:, 1]
    elements['z'] = means[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    elements['opacity'] = opacities
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = quats[:, 0]
    elements['rot_1'] = quats[:, 1]
    elements['rot_2'] = quats[:, 2]
    elements['rot_3'] = quats[:, 3]
    
    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(output_path)
    
    print(f'[INFO] Saved {N:,} Gaussians to {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Train 3DGS scene from panorama LDI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ldi_dir', type=str, required=True,
                        help='Path to panorama LDI directory')
    parser.add_argument('--output', type=str, default='scene_optimized.ply',
                        help='Output PLY file path')
    parser.add_argument('--num_iterations', type=int, default=3000,
                        help='Number of optimization iterations (paper uses ~3000)')
    parser.add_argument('--num_views', type=int, default=240,
                        help='Number of training views (paper uses 240)')
    parser.add_argument('--fov', type=float, default=44.702,
                        help='Field of view in degrees')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug images and videos')
    parser.add_argument('--video_interval', type=int, default=1000,
                        help='Render 360 video every N iterations (0 to disable)')
    parser.add_argument('--init_only', action='store_true',
                        help='Save initialized Gaussians without training')
    parser.add_argument('--scale_mult', type=float, default=2.0,
                        help='Gaussian scale as multiple of pixel footprint')
    parser.add_argument('--init_opacity', type=float, default=0.05,
                        help='Initial opacity for all Gaussians')
    parser.add_argument('--freeze_positions', action='store_true',
                        help='Freeze Gaussian positions during training (appearance-only optimization)')
    parser.add_argument('--depth_weight', type=float, default=0.005,
                        help='Depth L2 loss weight')
    parser.add_argument('--novel_view_weight', type=float, default=0.5,
                        help='Novel view depth loss weight (0 to disable)')
    parser.add_argument('--novel_view_radius', type=float, default=3.0,
                        help='Orbit radius for novel view cameras')
    parser.add_argument('--novel_view_start', type=int, default=500,
                        help='Iteration to start novel view depth loss')
    parser.add_argument('--novel_view_every', type=int, default=10,
                        help='Run novel view depth loss every N iterations')
    parser.add_argument('--novel_view_model', type=str, default='dav2',
                        choices=['dav2', 'moge'],
                        help='Monocular depth model for novel view loss')
    parser.add_argument('--depth_max', type=float, default=200,
                        help='Maximum depth clamp value (meters)')

    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load panorama LDI (doubled + reversed for view extraction)
    rgba_pano, depth_pano, original_width, num_layers = load_panorama_ldi(args.ldi_dir)

    # Extract perspective views for training targets
    views = extract_perspective_views(
        rgba_pano, depth_pano, original_width,
        num_views=args.num_views,
        fov_deg=args.fov,
        view_size=512,
        device=device
    )

    # Prepare training targets (per-layer and composite)
    targets = prepare_training_targets(views, device, depth_max=args.depth_max)

    # Load raw LDI (before duplication/reversal) for direct panorama unprojection
    rgba_raw = np.load(os.path.join(args.ldi_dir, 'rgba_ldi.npy'))
    depth_raw = np.load(os.path.join(args.ldi_dir, 'depth_ldi.npy'))

    print('[INFO] Unprojecting panorama to point cloud...')
    points, colors, alphas, depths, layer_ids = unproject_panorama_to_points(
        rgba_raw, depth_raw, init_opacity=args.init_opacity, depth_max=args.depth_max, device=device)

    print(f'[INFO] Total points: {len(points):,}')

    gaussians = initialize_gaussians(
        points, colors, alphas, depths, original_width,
        scale_mult=args.scale_mult, layer_ids=layer_ids, device=device)
    
    if not args.init_only:
        debug_dir = os.path.join(os.path.dirname(args.output) or '.', 'debug') if args.debug else None
        gaussians = train(gaussians, targets, args.num_iterations, debug_dir, args.video_interval,
                          freeze_positions=args.freeze_positions, depth_weight=args.depth_weight,
                          novel_view_weight=args.novel_view_weight,
                          novel_view_radius=args.novel_view_radius,
                          novel_view_start=args.novel_view_start,
                          novel_view_every=args.novel_view_every,
                          novel_view_model=args.novel_view_model,
                          depth_max=args.depth_max)
    
    # Save
    save_gaussians_ply(gaussians, args.output)
    
    print('[INFO] Done!')


if __name__ == '__main__':
    main()
