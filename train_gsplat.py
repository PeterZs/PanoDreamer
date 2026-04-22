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

from gsplat import rasterization


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
    
    # Duplicate horizontally for seamless 360 wrapping (and flip as in reference)
    rgba_pano = np.concatenate([rgba_pano, rgba_pano], axis=2)[::-1].copy()
    depth_pano = np.concatenate([depth_pano, depth_pano], axis=2)[::-1].copy()
    
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
    
    # Step size for panorama sampling
    step = original_width // num_views
    
    views = []
    
    print(f'[INFO] Extracting {num_views} perspective views from panorama...')
    for view_idx in tqdm(range(num_views), desc="Extracting views"):
        # Rotation angle for this view
        theta = view_idx * (2 * np.pi) / num_views
        
        # Crop window from panorama
        center = w // 2 + step * view_idx
        start = center - view_size // 2
        end = center + view_size // 2
        
        # Handle wrap-around
        if end > w:
            start = w - view_size
            end = w
        
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


def prepare_training_targets(views, device='cuda'):
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
        depth_layers = np.clip(view['depth_layers'], 0, 200)  # clamp extreme depths
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


def unproject_panorama_to_points(rgba_ldi, depth_ldi, init_opacity=0.05, device='cuda'):
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
    depth_ldi = np.clip(depth_ldi, 0, 200)
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
        # Densification stats
        'xyz_gradient_accum': torch.zeros((N, 1), device=device),
        'denom': torch.zeros((N, 1), device=device),
        'max_radii2D': torch.zeros((N,), device=device),
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


def render_gaussians(gaussians, camera, background=None, max_layer=None):
    """Render Gaussians from a camera view.

    Args:
        max_layer: If set, only render Gaussians from layers 0..max_layer.
                   Requires gaussians['layer_ids'] to be present.
    """
    if background is None:
        background = torch.ones(3, device=gaussians['means'].device)

    means = gaussians['means']
    scales = torch.exp(gaussians['scales'])
    quats = F.normalize(gaussians['quats'], dim=-1)
    opacities = torch.sigmoid(gaussians['opacities'])
    colors = gaussians['colors']

    if max_layer is not None and 'layer_ids' in gaussians:
        layer_mask = gaussians['layer_ids'] <= max_layer
        opacities = opacities * layer_mask.float()

    render_colors, render_alphas, _ = rasterization(
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
    )

    rgb = render_colors[0, ..., :3]
    depth = render_colors[0, ..., 3]
    alpha = render_alphas[0, ..., 0]

    return rgb, depth, alpha


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
            rgb, _, _ = render_gaussians(gaussians, camera, background)
        
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


def prune_gaussians(gaussians, mask_keep):
    """
    Prune Gaussians by keeping only those where mask_keep is True.
    Returns new pruned gaussians dict.
    """
    device = gaussians['means'].device
    
    # Prune trainable parameters
    new_gaussians = {
        'means': nn.Parameter(gaussians['means'].data[mask_keep].clone()),
        'scales': nn.Parameter(gaussians['scales'].data[mask_keep].clone()),
        'quats': nn.Parameter(gaussians['quats'].data[mask_keep].clone()),
        'opacities': nn.Parameter(gaussians['opacities'].data[mask_keep].clone()),
        'colors': nn.Parameter(gaussians['colors'].data[mask_keep].clone()),
        # Prune stats
        'xyz_gradient_accum': gaussians['xyz_gradient_accum'][mask_keep].clone(),
        'denom': gaussians['denom'][mask_keep].clone(),
        'max_radii2D': gaussians['max_radii2D'][mask_keep].clone(),
    }
    if 'layer_ids' in gaussians:
        new_gaussians['layer_ids'] = gaussians['layer_ids'][mask_keep].clone()

    return new_gaussians


def densify_and_prune(gaussians, opt, scene_extent=1.0, iteration=0):
    """
    Densify and prune Gaussians based on accumulated gradients.
    Following 3DGS paper approach.
    
    Returns:
        new_gaussians: Updated gaussians dict (or same if no changes)
        n_pruned: Number of Gaussians pruned
    """
    device = gaussians['means'].device
    N = gaussians['means'].shape[0]
    
    # Get gradient statistics
    grads = gaussians['xyz_gradient_accum'] / (gaussians['denom'] + 1e-7)
    grads[grads.isnan()] = 0.0
    grads = grads.squeeze()
    
    # Thresholds
    extent = scene_extent
    
    # Get current scales and opacities
    scales = torch.exp(gaussians['scales'].detach())
    opacities = torch.sigmoid(gaussians['opacities'].detach())
    max_scale = scales.max(dim=1).values
    
    # === Pruning (low opacity or too large) ===
    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
    mask_prune = (opacities < 0.005).squeeze()
    if size_threshold is not None:
        mask_prune = mask_prune | (max_scale > size_threshold * extent)
    
    # Keep mask
    keep_mask = ~mask_prune
    n_pruned = (~keep_mask).sum().item()
    
    if n_pruned > 0:
        new_gaussians = prune_gaussians(gaussians, keep_mask)
        return new_gaussians, n_pruned
    
    return gaussians, 0


def reset_densification_stats(gaussians):
    """Reset accumulated gradients for densification."""
    N = gaussians['means'].shape[0]
    device = gaussians['means'].device
    gaussians['xyz_gradient_accum'] = torch.zeros((N, 1), device=device)
    gaussians['denom'] = torch.zeros((N, 1), device=device)


def add_densification_stats(gaussians, viewspace_grads, visibility_mask):
    """
    Accumulate gradients for densification.
    """
    if viewspace_grads is None:
        return
    
    # Get visible gradients
    if visibility_mask is not None:
        viewspace_grads = viewspace_grads[visibility_mask]
        indices = torch.where(visibility_mask)[0]
    else:
        indices = torch.arange(len(viewspace_grads), device=viewspace_grads.device)
    
    # Accumulate gradient norms
    grad_norms = viewspace_grads.norm(dim=-1, keepdim=True)
    
    gaussians['xyz_gradient_accum'][indices] += grad_norms
    gaussians['denom'][indices] += 1


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
          freeze_positions=False, depth_weight=0.005):
    """
    Optimize Gaussians following paper approach.

    Args:
        video_interval: Render 360 video every N iterations (0 to disable)
        freeze_positions: If True, don't optimize Gaussian positions (preserves 3D structure)
    """
    device = gaussians['means'].device
    num_layers = targets[0]['layer_images'].shape[0]
    
    # Get optimization parameters
    opt = GSParams()
    opt.iterations = num_iterations
    
    # Learning rate scheduler for position
    position_lr_fn = get_expon_lr_func(
        lr_init=opt.position_lr_init,
        lr_final=opt.position_lr_final,
        lr_delay_mult=opt.position_lr_delay_mult,
        max_steps=opt.position_lr_max_steps
    )
    
    def build_optimizer(gaussians):
        """Build optimizer with per-parameter learning rates."""
        params = [
            {'params': [gaussians['scales']], 'lr': opt.scaling_lr, 'name': 'scales'},
            {'params': [gaussians['quats']], 'lr': opt.rotation_lr, 'name': 'quats'},
            {'params': [gaussians['opacities']], 'lr': opt.opacity_lr, 'name': 'opacities'},
            {'params': [gaussians['colors']], 'lr': opt.feature_lr, 'name': 'colors'},
        ]
        if not freeze_positions:
            params.insert(0, {'params': [gaussians['means']], 'lr': opt.position_lr_init, 'name': 'means'})
        return torch.optim.Adam(params, lr=0.0, eps=1e-15)
    
    optimizer = build_optimizer(gaussians)
    
    # Estimate scene extent for densification thresholds
    with torch.no_grad():
        scene_extent = gaussians['means'].abs().max().item() * 1.1
    
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save GT images for first few views
        for i in range(min(4, len(targets))):
            gt = targets[i]['composite_image']
            gt_np = (gt.cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(debug_dir, f'gt_view_{i:02d}.png'), gt_np[:, :, ::-1])
        
        # Render GT 360 video
        print('[INFO] Rendering GT 360 video...')
        render_gt_360_video(targets, os.path.join(debug_dir, 'gt_360.mp4'), 
                           num_frames=len(targets), fps=30)
        
        # Render initial state 360 video (before training)
        print('[INFO] Rendering initial state 360 video...')
        render_360_video(gaussians, targets, os.path.join(debug_dir, 'iter_0000_360.mp4'),
                        num_frames=60, fps=30, device=device)
    
    print(f'[INFO] Starting optimization for {num_iterations} iterations')
    print(f'[INFO] Num views: {len(targets)}, Num layers: {num_layers}')
    print(f'[INFO] Initial Gaussians: {gaussians["means"].shape[0]:,}')
    print(f'[INFO] Densification: iter {opt.densify_from_iter} to {opt.densify_until_iter}, interval {opt.densification_interval}')
    
    if freeze_positions:
        print(f'[INFO] Positions FROZEN — optimizing color/opacity/scale only')

    pbar = tqdm(range(1, num_iterations + 1), desc="Training")
    for iteration in pbar:
        if not freeze_positions:
            for param_group in optimizer.param_groups:
                if param_group['name'] == 'means':
                    param_group['lr'] = position_lr_fn(iteration)
        
        # Random view selection
        view_idx = np.random.randint(len(targets))
        target = targets[view_idx]

        # Create camera
        camera = create_camera(target['theta'], target['focal'],
                               target['layer_images'].shape[1],
                               target['layer_images'].shape[2], device)

        # Random background color prevents Gaussians from learning to be transparent
        background = torch.rand(3, device=device)

        # === Per-layer loss: random layer, render only Gaussians up to that layer ===
        random_layer = np.random.randint(num_layers)
        layer_rgb, layer_depth, _ = render_gaussians(
            gaussians, camera, background, max_layer=random_layer)
        layer_mask = target['layer_masks'][random_layer]
        loss_layer, _, _, _ = compute_loss(
            layer_rgb, target['layer_images'][random_layer],
            layer_depth, target['layer_depths'][random_layer],
            mask=layer_mask, lambda_dssim=opt.lambda_dssim,
            depth_weight=depth_weight
        )

        # === Composite loss: all Gaussians ===
        rendered_rgb, rendered_depth, alpha = render_gaussians(gaussians, camera, background)
        loss_comp, _, _, _ = compute_loss(
            rendered_rgb, target['composite_image'],
            rendered_depth, target['composite_depth'],
            mask=None, lambda_dssim=opt.lambda_dssim,
            depth_weight=depth_weight
        )

        loss = loss_layer + loss_comp
        
        # Backward
        loss.backward()
        
        with torch.no_grad():
            # === Densification ===
            if iteration < opt.densify_until_iter:
                # Track gradients for densification
                if gaussians['means'].grad is not None:
                    grad_norm = gaussians['means'].grad.norm(dim=-1, keepdim=True)
                    gaussians['xyz_gradient_accum'] += grad_norm
                    gaussians['denom'] += 1
                
                # Densify and prune at intervals
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians, n_pruned = densify_and_prune(gaussians, opt, scene_extent, iteration)
                    
                    if n_pruned > 0:
                        # Rebuild optimizer with new parameters
                        optimizer = build_optimizer(gaussians)
                        # Reset stats
                        reset_densification_stats(gaussians)
                        tqdm.write(f'[PRUNE] iter {iteration}: removed {n_pruned:,} Gaussians, now {gaussians["means"].shape[0]:,}')
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Update progress
        if iteration % 10 == 0:
            n_gauss = gaussians['means'].shape[0]
            pbar.set_description(f"Training (loss: {loss.item():.4f}, N: {n_gauss//1000}k)")
        
        # Debug output - comparison images
        if debug_dir and iteration % 500 == 0:
            with torch.no_grad():
                for i in [0, len(targets)//4, len(targets)//2, 3*len(targets)//4]:
                    i = i % len(targets)
                    t = targets[i]
                    cam = create_camera(t['theta'], t['focal'], 
                                        t['layer_images'].shape[1], 
                                        t['layer_images'].shape[2], device)
                    rgb, depth, _ = render_gaussians(gaussians, cam, background)
                    
                    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
                    gt_np = (t['composite_image'].cpu().numpy() * 255).astype(np.uint8)
                    comparison = np.hstack([gt_np, rgb_np])
                    cv2.imwrite(os.path.join(debug_dir, f'compare_iter_{iteration:04d}_view_{i:02d}.png'), comparison[:, :, ::-1])
        
        # Debug output - 360 videos at intervals
        if debug_dir and video_interval > 0 and iteration % video_interval == 0:
            tqdm.write(f'[INFO] Rendering 360 video at iteration {iteration}...')
            render_360_video(gaussians, targets, 
                           os.path.join(debug_dir, f'iter_{iteration:04d}_360.mp4'),
                           num_frames=60, fps=30, device=device)
    
    # Render final 360 video
    if debug_dir:
        print('[INFO] Rendering final 360 video...')
        render_360_video(gaussians, targets, 
                        os.path.join(debug_dir, 'final_360.mp4'),
                        num_frames=60, fps=30, device=device)
    
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
    targets = prepare_training_targets(views, device)

    # Load raw LDI (before duplication/reversal) for direct panorama unprojection
    rgba_raw = np.load(os.path.join(args.ldi_dir, 'rgba_ldi.npy'))
    depth_raw = np.load(os.path.join(args.ldi_dir, 'depth_ldi.npy'))

    print('[INFO] Unprojecting panorama to point cloud...')
    points, colors, alphas, depths, layer_ids = unproject_panorama_to_points(
        rgba_raw, depth_raw, init_opacity=args.init_opacity, device=device)

    print(f'[INFO] Total points: {len(points):,}')

    gaussians = initialize_gaussians(
        points, colors, alphas, depths, original_width,
        scale_mult=args.scale_mult, layer_ids=layer_ids, device=device)
    
    if not args.init_only:
        debug_dir = os.path.join(os.path.dirname(args.output) or '.', 'debug') if args.debug else None
        gaussians = train(gaussians, targets, args.num_iterations, debug_dir, args.video_interval,
                          freeze_positions=args.freeze_positions, depth_weight=args.depth_weight)
    
    # Save
    save_gaussians_ply(gaussians, args.output)
    
    print('[INFO] Done!')


if __name__ == '__main__':
    main()
