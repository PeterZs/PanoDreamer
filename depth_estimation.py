"""
Depth Estimation with View Stitching

Supports two modes:
1. Wide image (perspective): Direct stitching without projection
2. Cylindrical panorama (360°): Uses cylindrical projection for seamless wraparound

Both modes:
- Extract overlapping views
- Run monocular depth estimation on each view  
- Align depths across views using piecewise regression
- Stitch into a coherent depth map

Paper: PanoDreamer - https://people.engr.tamu.edu/nimak/Papers/PanoDreamer/
"""

import os
import math
import argparse
import numpy as np
import cv2
import torch
import kornia
from kornia.utils import create_meshgrid
from PIL import Image
from tqdm import tqdm
import matplotlib.cm as cm

from ropwr import RobustPWRegression
from utils.depth_utilsv2 import (
    estimate_depth, estimate_metric_depth, estimate_depth_moge,
    calibrate_relative_depth
)
from utils.depth_layering import get_depth_bins
from utils.depth import colorize


def fov2focal(fov_radians, pixels):
    """Convert field of view to focal length."""
    return pixels / (2 * math.tan(fov_radians / 2))


def cyl_proj(img, focal_length):
    """
    Project perspective image to cylindrical coordinates.
    
    Args:
        img: Input image tensor [B, C, H, W]
        focal_length: Focal length for projection
        
    Returns:
        Cylindrical projection of image
    """
    device = img.device
    grid = create_meshgrid(img.shape[2], img.shape[3], normalized_coordinates=False, device=device)
    y, x = grid[..., 0], grid[..., 1]
    h, w = img.shape[2:]
    center_x = w // 2
    center_y = h // 2
    
    x_shifted = x - center_x
    y_shifted = y - center_y
    
    theta = torch.arctan(x_shifted / focal_length)
    height = y_shifted / torch.sqrt(x_shifted ** 2 + focal_length ** 2)
    
    x_cyl = focal_length * theta + center_x
    y_cyl = height * focal_length + center_y
    
    img_cyl = kornia.geometry.transform.remap(
        img, torch.flip(x_cyl, dims=(1, 2)), y_cyl, 
        mode='nearest', align_corners=True
    )
    img_cyl = torch.rot90(img_cyl, k=3, dims=(2, 3))
    
    return img_cyl


def cyl_proj_inv(img, focal_length):
    """
    Project cylindrical image back to perspective coordinates.
    
    Args:
        img: Cylindrical image tensor [B, C, H, W]
        focal_length: Focal length for projection
        
    Returns:
        Perspective projection of image
    """
    device = img.device
    grid = create_meshgrid(img.shape[2], img.shape[3], normalized_coordinates=False, device=device)
    y_cyl, x_cyl = grid[..., 0], grid[..., 1]
    h, w = img.shape[2:]
    center_x = w // 2
    center_y = h // 2
    
    theta = (x_cyl - center_x) / focal_length
    height = (y_cyl - center_y) / focal_length
    
    x_shifted = torch.tan(theta) * focal_length
    y_shifted = height * torch.sqrt(x_shifted ** 2 + focal_length ** 2)
    
    x = x_shifted + center_x
    y = y_shifted + center_y
    
    img_persp = kornia.geometry.transform.remap(
        img, torch.flip(x, dims=(1, 2)), y,
        mode='nearest', align_corners=True
    )
    img_persp = torch.rot90(img_persp, k=3, dims=(2, 3))
    
    return img_persp


def get_masks(depth, num_bins=5):
    """
    Get depth masks for piecewise regression.
    
    Args:
        depth: Depth tensor [1, 1, H, W]
        num_bins: Number of depth bins
        
    Returns:
        masks: Array of masks for each depth bin
        bins: Depth bin boundaries
    """
    bins = get_depth_bins(depth=depth, num_bins=num_bins)
    dep = depth[0, 0]
    masks = []

    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            mask = torch.where((dep >= bins[i]) & (dep <= bins[i+1]), 1, 0)
        else:
            mask = torch.where((dep >= bins[i]) & (dep < bins[i+1]), 1, 0)
        masks.append(mask[None])
    
    masks = torch.cat(masks, dim=0).numpy()
    return masks, bins



def estimate_wide_depth(image, save_dir, num_iterations=15, num_bins=10,
                        metric_dataset='vkitti', metric_model='dav2', debug=False):
    """
    Estimate depth for a wide perspective image (no cylindrical projection).

    Args:
        image: Wide image as numpy array [H, W, 3]
        save_dir: Directory to save outputs
        num_iterations: Number of alignment iterations
        num_bins: Number of depth bins for piecewise regression
        metric_dataset: 'vkitti' (outdoor, 80m) or 'hypersim' (indoor, 20m)
        debug: Save debug info
        
    Returns:
        depth: Estimated depth [H, W]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize piecewise regression
    pw = RobustPWRegression(objective="l2", degree=1, monotonic_trend="ascending")
    
    h, w = image.shape[:2]
    view_size = 512
    
    # Calculate view positions with overlap
    step = 32  # Dense overlap for smooth stitching
    num_views = max(1, (w - view_size) // step + 1)
    
    print(f'[INFO] Estimating depth for {num_views} views...')
    
    # Step 1: Estimate depth for each view
    depth_arr = []
    bins_arr = []
    view_starts = []
    
    for view_i in tqdm(range(num_views), desc="Depth estimation"):
        view_start = min(view_i * step, w - view_size)
        view_starts.append(view_start)
        
        # Extract view
        image_curr = image[:, view_start:view_start + view_size]
        image_pil = Image.fromarray(image_curr)
        
        # Estimate depth (DA V2 outputs disparity-like: larger = closer)
        # Align in disparity space (uniform distribution, good for regression)
        monodepth = estimate_depth(image_pil)
        depth_curr = monodepth.astype(np.float32)

        # Percentile bins for piecewise regression in disparity space
        vals = depth_curr.flatten()
        vals_pos = vals[vals > 0]
        bin_edges = np.percentile(vals_pos, np.linspace(0, 100, num_bins + 1))
        bins = sorted(set(bin_edges.tolist()))
        bins[0] -= 1e-6
        bins[-1] += 1e-6

        depth_arr.append(depth_curr)
        bins_arr.append(bins)
    
    if debug:
        np.save(f"{save_dir}/depth_info.npy", {
            'depth_arr': depth_arr, 
            'bins_arr': bins_arr,
            'view_starts': view_starts
        })
    
    print(f'[INFO] Aligning depths over {num_iterations} iterations...')
    
    # Step 2: Iteratively align depths
    for iteration in tqdm(range(num_iterations), desc="Alignment iterations"):
        # Accumulate depth
        depth_full = np.zeros((h, w), dtype=np.float32)
        mask_full = np.zeros((h, w), dtype=np.float32)
        
        for view_i in range(num_views):
            depth_curr = depth_arr[view_i]
            view_start = view_starts[view_i]
            
            # Use center portion to avoid edge artifacts
            margin = 50
            depth_full[:, view_start + margin:view_start + view_size - margin] += depth_curr[:, margin:-margin]
            mask_full[:, view_start + margin:view_start + view_size - margin] += 1
        
        # Average overlapping regions
        mask_full = np.maximum(mask_full, 1e-6)  # Avoid divide by zero
        depth_full = depth_full / mask_full
        
        # Save iteration result (invert disparity for depth visualization)
        depth_viz = 1.0 / np.maximum(depth_full, 0.01)
        if iteration == 0:
            viz_min, viz_max = depth_viz.min(), depth_viz.max()
        depth_normalized = (depth_viz - viz_min) / (viz_max - viz_min + 1e-6)
        depth_colored = colorize(depth_normalized, cmap='turbo')
        cv2.imwrite(f"{save_dir}/depth_iter_{iteration:02d}.png", depth_colored[..., :3][..., ::-1])
        
        if iteration == num_iterations - 1:
            break
        
        # Align individual view depths to current composite
        for view_i in range(num_views):
            depth_curr = depth_arr[view_i]
            view_start = view_starts[view_i]
            
            # Get reference depth from current composite
            depth_ref = depth_full[:, view_start:view_start + view_size]
            
            # Fit piecewise regression
            try:
                pw.fit(depth_curr.flatten(), depth_ref.flatten(), bins_arr[view_i][1:-1])
                depth_curr = pw.predict(depth_curr.flatten()).reshape(depth_curr.shape).astype(np.float32)
                depth_arr[view_i] = depth_curr
            except Exception as e:
                if debug:
                    print(f"[WARNING] Alignment failed for view {view_i}: {e}")
                continue
    
    # Invert disparity to depth, then calibrate to metric
    depth_full = 1.0 / np.maximum(depth_full, 0.01)

    ref_idx = num_views // 2
    ref_start = view_starts[ref_idx]
    ref_image = image[:, ref_start:ref_start + view_size]
    ref_pil = Image.fromarray(ref_image)
    if metric_model == 'moge':
        metric_ref = estimate_depth_moge(ref_pil)
    else:
        metric_ref = estimate_metric_depth(ref_pil, dataset=metric_dataset)
    ref_depth = depth_full[:, ref_start:ref_start + view_size]
    scale, bias = calibrate_relative_depth(ref_depth, metric_ref)
    depth_full = scale * depth_full + bias
    depth_full = np.maximum(depth_full, 0.1)
    print(f'[INFO] Calibrated depth range: [{depth_full.min():.2f}m, {depth_full.max():.2f}m]')

    # Save final outputs
    np.save(f"{save_dir}/depth.npy", depth_full)

    depth_normalized = (depth_full - depth_full.min()) / (depth_full.max() - depth_full.min() + 1e-6)
    depth_colored = colorize(depth_normalized, cmap='turbo')
    cv2.imwrite(f"{save_dir}/depth.png", depth_colored[..., :3][..., ::-1])

    print(f'[INFO] Depth estimation complete!')
    print(f'[INFO] Saved: {save_dir}/depth.npy')
    print(f'[INFO] Saved: {save_dir}/depth.png')

    return depth_full


def estimate_panorama_depth(image_pano, save_dir, num_iterations=15, num_bins=10,
                            input_fov=44.701948991275390, mul_factor=12,
                            metric_dataset='vkitti', metric_model='dav2', debug=False):
    """
    Estimate depth for a cylindrical panorama (360°).
    
    Args:
        image_pano: Panorama image as numpy array [H, W, 3]
        save_dir: Directory to save outputs
        num_iterations: Number of alignment iterations
        num_bins: Number of depth bins for piecewise regression
        input_fov: Field of view in degrees
        mul_factor: View sampling factor
        metric_dataset: 'vkitti' (outdoor, 80m) or 'hypersim' (indoor, 20m)
        debug: Save debug info
        
    Returns:
        depth_pano: Estimated panorama depth [H, W]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize piecewise regression
    pw = RobustPWRegression(objective="l2", degree=1, monotonic_trend="ascending")
    
    # Calculate focal length
    input_focal = fov2focal(input_fov * math.pi / 180, 512)
    
    h, w = image_pano.shape[:2]
    
    # Tile panorama for wraparound
    image_pano_tiled = np.concatenate([image_pano, image_pano], axis=1)
    
    # Calculate view positions
    step = 384 // mul_factor
    num_views = (w // step) + 1
    
    print(f'[INFO] Estimating depth for {num_views} views...')
    
    # Step 1: Estimate depth for each view
    depth_arr = []
    mask_arr = []
    bins_arr = []
    
    for view_i in tqdm(range(num_views), desc="Depth estimation"):
        # Extract view from panorama
        view_start = w // 2 - 256 + step * view_i
        view_end = view_start + 512
        image_curr = image_pano_tiled[:, view_start:view_end]
        
        # Project to perspective
        image_tensor = torch.tensor(image_curr).permute(2, 0, 1)[None].to(device).float() / 255.
        image_proj = cyl_proj(image_tensor, input_focal).cpu().numpy()[0].transpose(1, 2, 0)
        image_proj = Image.fromarray((image_proj * 255).astype(np.uint8))
        
        # Estimate depth (DA V2 outputs disparity-like: larger = closer)
        # Align in disparity space (uniform distribution, good for regression)
        monodepth = estimate_depth(image_proj)
        depth_curr = monodepth.astype(np.float32)

        # Percentile bins for piecewise regression in disparity space
        vals = depth_curr.flatten()
        vals_pos = vals[vals > 0]
        bin_edges = np.percentile(vals_pos, np.linspace(0, 100, num_bins + 1))
        bins = sorted(set(bin_edges.tolist()))
        bins[0] -= 1e-6
        bins[-1] += 1e-6

        depth_arr.append(depth_curr)
        bins_arr.append(bins)
    
    if debug:
        np.save(f"{save_dir}/depth_info.npy", {
            'depth_arr': depth_arr,
            'bins_arr': bins_arr
        })
    
    print(f'[INFO] Aligning depths over {num_iterations} iterations...')
    
    # Step 2: Iteratively align depths
    for iteration in tqdm(range(num_iterations), desc="Alignment iterations"):
        # Accumulate depth panorama
        depth_pano = np.zeros((h, w * 2), dtype=np.float32)
        mask_pano = np.zeros((h, w * 2), dtype=np.float32)
        
        for view_i in range(num_views):
            depth_curr = depth_arr[view_i]
            view_start = w // 2 - 256 + step * view_i
            
            # Project depth back to cylindrical
            depth_tensor = torch.tensor(depth_curr)[None, None].to(device)
            depth_proj = cyl_proj_inv(depth_tensor, input_focal).cpu().numpy()[0, 0]
            
            mask_tensor = torch.tensor(np.ones_like(depth_curr))[None, None].to(device).float()
            mask_proj = cyl_proj_inv(mask_tensor, input_focal).cpu().numpy()[0, 0]
            
            # Accumulate (excluding edges to avoid artifacts)
            depth_pano[:, view_start + 100:view_start + 412] += depth_proj[:, 100:-100]
            mask_pano[:, view_start + 100:view_start + 412] += mask_proj[:, 100:-100]
        
        # Handle 360° wraparound
        depth_pano[:, :w] = depth_pano[:, :w] + depth_pano[:, w:]
        depth_pano[:, w:] = depth_pano[:, :w]
        mask_pano[:, :w] = mask_pano[:, :w] + mask_pano[:, w:]
        mask_pano[:, w:] = mask_pano[:, :w]
        
        # Average overlapping regions
        depth_pano = np.where(mask_pano > 0, depth_pano / mask_pano, depth_pano)
        
        # Store min/max from first iteration for consistent scaling
        if iteration == 0:
            depth_max = depth_pano.max()
            depth_min = depth_pano.min()
        
        # Save iteration result (invert disparity for depth visualization)
        disp_save = depth_pano[:, :w]
        depth_viz = 1.0 / np.maximum(disp_save, 0.01)
        if iteration == 0:
            viz_min, viz_max = depth_viz.min(), depth_viz.max()
        depth_normalized = (depth_viz - viz_min) / (viz_max - viz_min + 1e-6)
        depth_colored = colorize(depth_normalized, cmap='turbo')
        cv2.imwrite(f"{save_dir}/depth_iter_{iteration:02d}.png", depth_colored[..., :3][..., ::-1])
        
        if iteration == num_iterations - 1:
            break
        
        # Align individual view depths to current panorama
        for view_i in range(num_views):
            depth_curr = depth_arr[view_i]
            view_start = w // 2 - 256 + step * view_i
            
            # Get reference depth from current panorama
            depth_ref = depth_pano[:, view_start:view_start + 512]
            depth_ref_tensor = torch.tensor(depth_ref)[None, None].to(device)
            depth_ref_proj = cyl_proj(depth_ref_tensor, input_focal).cpu().numpy()[0, 0]
            
            # Fit piecewise regression
            try:
                pw.fit(depth_curr.flatten(), depth_ref_proj.flatten(), bins_arr[view_i][1:-1])
                depth_curr = pw.predict(depth_curr.flatten()).reshape(depth_curr.shape).astype(np.float32)
                depth_arr[view_i] = depth_curr
            except Exception as e:
                if debug:
                    print(f"[WARNING] Alignment failed for view {view_i}: {e}")
                continue
    
    # Extract final half panorama (disparity space) and invert to depth
    disp_pano_half = depth_pano[:, :w]
    depth_pano_half = 1.0 / np.maximum(disp_pano_half, 0.01)

    # Calibrate to metric depth using multiple reference views
    num_calib_views = 8
    calib_step = num_views // num_calib_views
    log_rel_all, log_met_all = [], []

    # Tile disp_pano_half for wraparound access during calibration
    disp_pano_tiled = np.concatenate([disp_pano_half, disp_pano_half], axis=1)

    print(f'[INFO] Calibrating with {num_calib_views} reference views (metric_model={metric_model})...')
    for ci in range(num_calib_views):
        vi = ci * calib_step
        ref_start = w // 2 - 256 + step * vi
        ref_image = image_pano_tiled[:, ref_start:ref_start + 512]
        ref_tensor = torch.tensor(ref_image).permute(2, 0, 1)[None].to(device).float() / 255.
        ref_persp_np = cyl_proj(ref_tensor, input_focal).cpu().numpy()[0].transpose(1, 2, 0)
        ref_persp_pil = Image.fromarray((ref_persp_np * 255).astype(np.uint8))

        if metric_model == 'moge':
            metric_ref = estimate_depth_moge(ref_persp_pil)
        else:
            metric_ref = estimate_metric_depth(ref_persp_pil, dataset=metric_dataset)

        ref_disp_cyl = disp_pano_tiled[:, ref_start:ref_start + 512]
        ref_disp_tensor = torch.tensor(ref_disp_cyl).float()[None, None].to(device)
        ref_disp_persp = cyl_proj(ref_disp_tensor, input_focal).cpu().numpy()[0, 0]
        ref_depth_persp = 1.0 / np.maximum(ref_disp_persp, 0.01)

        if ref_depth_persp.shape != metric_ref.shape:
            metric_ref = cv2.resize(metric_ref, (ref_depth_persp.shape[1], ref_depth_persp.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

        valid = (ref_depth_persp > 0.1) & (metric_ref > 0.1) & np.isfinite(metric_ref)
        if valid.sum() > 100:
            log_rel_all.append(np.log(ref_depth_persp[valid]))
            log_met_all.append(np.log(metric_ref[valid]))

    # Robust scale+shift calibration in log space
    log_rel = np.concatenate(log_rel_all)
    log_met = np.concatenate(log_met_all)

    # Check correlation direction
    corr = np.corrcoef(log_rel, log_met)[0, 1]
    print(f'[INFO] Log correlation: {corr:.3f}')

    if corr > 0.1:
        # Positive correlation: fit log(metric) = a * log(relative) + b
        if len(log_rel) > 100000:
            idx = np.random.choice(len(log_rel), 100000, replace=False)
            log_rel_s, log_met_s = log_rel[idx], log_met[idx]
        else:
            log_rel_s, log_met_s = log_rel, log_met
        A = np.stack([log_rel_s, np.ones_like(log_rel_s)], axis=1)
        a, b = np.linalg.lstsq(A, log_met_s, rcond=None)[0]
        a = max(a, 0.1)  # enforce positive slope
        print(f'[INFO] Log-space fit: a={a:.4f}, b={b:.4f}')
        log_depth = np.log(np.maximum(depth_pano_half, 0.01))
        depth_pano_half = np.exp(a * log_depth + b)
    else:
        # Negative or weak correlation: use median ratio scaling
        # This happens when DA V2 disparity-derived depth inverts the ordering
        ratios = np.exp(log_met - log_rel)
        scale = np.median(ratios)
        print(f'[INFO] Median ratio calibration: scale={scale:.4f}')
        depth_pano_half = depth_pano_half * scale

    depth_pano_half = np.maximum(depth_pano_half, 0.1)
    print(f'[INFO] Calibrated depth range: [{depth_pano_half.min():.2f}m, {depth_pano_half.max():.2f}m]')

    # Save final outputs
    np.save(f"{save_dir}/depth_pano.npy", depth_pano_half)

    depth_normalized = (depth_pano_half - depth_pano_half.min()) / (depth_pano_half.max() - depth_pano_half.min() + 1e-6)
    depth_colored = colorize(depth_normalized, cmap='turbo')
    cv2.imwrite(f"{save_dir}/depth_pano.png", depth_colored[..., :3][..., ::-1])

    print(f'[INFO] Depth estimation complete!')
    print(f'[INFO] Saved: {save_dir}/depth_pano.npy')
    print(f'[INFO] Saved: {save_dir}/depth_pano.png')

    return depth_pano_half


def estimate_panorama_depth_moge(image_pano, save_dir, num_iterations=15, num_bins=10,
                                  input_fov=44.701948991275390, mul_factor=12, debug=False):
    """
    Estimate panorama depth using MoGe V2 directly (metric depth, no calibration needed).
    Same view extraction + stitching as DA V2 but with MoGe's metric output.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(save_dir, exist_ok=True)
    h, w = image_pano.shape[:2]
    input_focal = fov2focal(input_fov * math.pi / 180, 512)
    step = 384 // mul_factor
    num_views = (w // step) + 1

    image_pano_tiled = np.concatenate([image_pano, image_pano], axis=1)

    print(f'[INFO] MoGe V2 panorama depth: {num_views} views, {num_iterations} alignment iterations')

    depth_arr = np.zeros((num_views, h, 512), dtype=np.float32)

    for view_i in tqdm(range(num_views), desc="MoGe depth"):
        start = w // 2 - 256 + step * view_i
        image_crop = image_pano_tiled[:, start:start + 512]
        image_tensor = torch.tensor(image_crop).permute(2, 0, 1)[None].to(device).float() / 255.
        image_proj = cyl_proj(image_tensor, input_focal).cpu().numpy()[0].transpose(1, 2, 0)
        image_proj = (np.clip(image_proj, 0, 1) * 255).astype(np.uint8)
        depth_arr[view_i] = estimate_depth_moge(image_proj)

    # Build validity masks and convert to log-depth for alignment
    # Log-depth gives equal weight to each depth octave, preserving dynamic range
    valid_arr = []
    log_arr = np.zeros_like(depth_arr)
    for view_i in range(num_views):
        valid_mask = depth_arr[view_i] > 0.5
        valid_arr.append(valid_mask)
        log_arr[view_i] = np.where(valid_mask, np.log(depth_arr[view_i]), 0)

    bins_arr = []
    for view_i in range(num_views):
        v = valid_arr[view_i]
        if v.sum() < 100:
            bins_arr.append(None)
            continue
        log_valid = log_arr[view_i][v]
        num_actual = min(num_bins, max(2, len(np.unique(log_valid)) // 10))
        bins = np.percentile(log_valid, np.linspace(0, 100, num_actual + 1))
        bins_arr.append(bins)

    log_pano = np.zeros((h, 2 * w), dtype=np.float32)
    weight_pano = np.zeros((h, 2 * w), dtype=np.float32)

    pw = RobustPWRegression(objective="l2", degree=1, monotonic_trend="ascending")
    for iteration in range(num_iterations):
        log_pano[:] = 0
        weight_pano[:] = 0

        for view_i in range(num_views):
            start = w // 2 - 256 + step * view_i
            end = start + 512
            v = valid_arr[view_i]
            log_pano[:, start:end] += log_arr[view_i] * v
            weight_pano[:, start:end] += v

        # Fold wraparound: views past column w wrap back to [0, w]
        log_pano[:, :w] += log_pano[:, w:]
        log_pano[:, w:] = log_pano[:, :w]
        weight_pano[:, :w] += weight_pano[:, w:]
        weight_pano[:, w:] = weight_pano[:, :w]

        has_data = weight_pano > 0
        log_pano[has_data] /= weight_pano[has_data]

        if iteration == num_iterations - 1:
            break

        for view_i in range(num_views):
            if bins_arr[view_i] is None:
                continue
            start = w // 2 - 256 + step * view_i
            end = start + 512
            v = valid_arr[view_i].flatten()
            log_ref = log_pano[:, start:end].flatten()[v]
            log_curr = log_arr[view_i].flatten()[v]
            if len(log_ref) < 100:
                continue
            try:
                pw.fit(log_curr, log_ref, bins_arr[view_i][1:-1])
                full_pred = pw.predict(log_arr[view_i].flatten())
                log_arr[view_i] = full_pred.reshape(log_arr[view_i].shape).astype(np.float32)
                log_arr[view_i] *= valid_arr[view_i]
            except Exception as e:
                if debug:
                    print(f"[WARNING] Alignment failed for view {view_i}: {e}")
                continue

    # Convert back to depth
    depth_pano_half = np.where(log_pano[:, :w] != 0, np.exp(log_pano[:, :w]), 0)
    depth_pano_half = np.maximum(depth_pano_half, 0.1)
    print(f'[INFO] MoGe depth range: [{depth_pano_half.min():.2f}m, {depth_pano_half.max():.2f}m]')

    np.save(f"{save_dir}/depth_pano.npy", depth_pano_half)
    depth_normalized = (depth_pano_half - depth_pano_half.min()) / (depth_pano_half.max() - depth_pano_half.min() + 1e-6)
    depth_colored = colorize(depth_normalized, cmap='turbo')
    cv2.imwrite(f"{save_dir}/depth_pano.png", depth_colored[..., :3][..., ::-1])

    print(f'[INFO] Saved: {save_dir}/depth_pano.npy')
    print(f'[INFO] Saved: {save_dir}/depth_pano.png')
    return depth_pano_half


def _poisson_merge_cylindrical(depth_views, view_starts, pano_w, pano_h, view_w=512):
    """
    Merge overlapping depth views using Poisson integration on log-depth gradients.
    Adapted from MoGe's merge_panorama_depth for cylindrical panoramas.
    Wraps horizontally (360°), no wrap vertically.
    """
    from scipy.sparse import lil_matrix, vstack as sp_vstack
    from scipy.sparse.linalg import lsmr
    from scipy.ndimage import convolve

    N = pano_h * pano_w
    num_views = len(depth_views)

    # Accumulate log-depth gradients from all views
    grad_x_sum = np.zeros((pano_h, pano_w), dtype=np.float64)
    grad_y_sum = np.zeros((pano_h, pano_w), dtype=np.float64)
    grad_x_count = np.zeros((pano_h, pano_w), dtype=np.float64)
    grad_y_count = np.zeros((pano_h, pano_w), dtype=np.float64)

    for vi in range(num_views):
        log_d = np.log(np.maximum(depth_views[vi], 0.01))
        start = view_starts[vi]
        mask = depth_views[vi] > 0.01

        gx = log_d[:, :-1] - log_d[:, 1:]
        mx = mask[:, :-1] & mask[:, 1:]
        gy = log_d[:-1, :] - log_d[1:, :]
        my = mask[:-1, :] & mask[1:, :]

        for c in range(view_w - 1):
            pc = (start + c) % pano_w
            grad_x_sum[:, pc] += gx[:, c] * mx[:, c]
            grad_x_count[:, pc] += mx[:, c]
        for c in range(view_w):
            pc = (start + c) % pano_w
            grad_y_sum[:-1, pc] += gy[:, c] * my[:, c]
            grad_y_count[:-1, pc] += my[:, c]

    valid_gx = grad_x_count > 0
    valid_gy = grad_y_count > 0
    grad_x_avg = np.where(valid_gx, grad_x_sum / np.maximum(grad_x_count, 1), 0)
    grad_y_avg = np.where(valid_gy, grad_y_sum / np.maximum(grad_y_count, 1), 0)

    # Build sparse gradient system: x_i - x_j = grad for each valid pair
    rows = []
    cols_i = []
    cols_j = []
    vals = []
    eq_idx = 0

    # Horizontal gradients (wrap in x)
    for r in range(pano_h):
        for c in range(pano_w):
            if valid_gx[r, c]:
                c_next = (c + 1) % pano_w
                rows.append(eq_idx)
                cols_i.append(r * pano_w + c)
                cols_j.append(r * pano_w + c_next)
                vals.append(grad_x_avg[r, c])
                eq_idx += 1

    # Vertical gradients (no wrap)
    for r in range(pano_h - 1):
        for c in range(pano_w):
            if valid_gy[r, c]:
                rows.append(eq_idx)
                cols_i.append(r * pano_w + c)
                cols_j.append((r + 1) * pano_w + c)
                vals.append(grad_y_avg[r, c])
                eq_idx += 1

    num_eqs = eq_idx
    from scipy.sparse import coo_matrix
    row_idx = np.array(rows + rows, dtype=np.int32)
    col_idx = np.array(cols_i + cols_j, dtype=np.int32)
    data = np.array([1.0] * num_eqs + [-1.0] * num_eqs, dtype=np.float64)
    A = coo_matrix((data, (row_idx, col_idx)), shape=(num_eqs, N)).tocsr()
    b = np.array(vals, dtype=np.float64)

    # Initial guess from weighted average
    depth_pano = np.zeros((pano_h, pano_w), dtype=np.float64)
    weight = np.zeros((pano_h, pano_w), dtype=np.float64)
    for vi in range(num_views):
        start = view_starts[vi]
        for c in range(view_w):
            pc = (start + c) % pano_w
            depth_pano[:, pc] += depth_views[vi][:, c]
            weight[:, pc] += 1
    valid = weight > 0
    depth_pano[valid] /= weight[valid]
    depth_pano[~valid] = 1.0
    x0 = np.log(np.maximum(depth_pano, 0.01)).reshape(-1)

    result, *_ = lsmr(A, b, x0=x0, atol=1e-5, btol=1e-5, show=False)
    depth_result = np.exp(result.reshape(pano_h, pano_w))

    return depth_result.astype(np.float32)


def estimate_panorama_depth_moge_poisson(image_pano, save_dir, num_bins=10,
                                          input_fov=44.701948991275390, mul_factor=12,
                                          debug=False):
    """
    Estimate panorama depth using MoGe V2 + Poisson gradient merge.

    Adapted from MoGe's panorama pipeline for cylindrical panoramas:
    - Our cylindrical view extraction (cyl_proj)
    - MoGe V2 metric depth per view
    - Poisson integration on log-depth gradients for globally consistent merge
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(save_dir, exist_ok=True)
    h, w = image_pano.shape[:2]
    input_focal = fov2focal(input_fov * math.pi / 180, 512)
    step = 384 // mul_factor
    num_views = (w // step) + 1

    image_pano_tiled = np.concatenate([image_pano, image_pano], axis=1)

    print(f'[INFO] MoGe V2 + Poisson merge: {num_views} views')

    depth_views = []
    view_starts = []

    for view_i in tqdm(range(num_views), desc="MoGe depth"):
        start = w // 2 - 256 + step * view_i
        image_crop = image_pano_tiled[:, start:start + 512]
        image_tensor = torch.tensor(image_crop).permute(2, 0, 1)[None].to(device).float() / 255.
        image_proj = cyl_proj(image_tensor, input_focal).cpu().numpy()[0].transpose(1, 2, 0)
        image_proj = (np.clip(image_proj, 0, 1) * 255).astype(np.uint8)
        depth_view = estimate_depth_moge(image_proj)
        depth_views.append(depth_view)
        view_starts.append(start % w)

    print('[INFO] Running Poisson merge on log-depth gradients...')
    depth_pano_full = _poisson_merge_cylindrical(depth_views, view_starts, w, h, view_w=512)
    depth_pano_full = np.maximum(depth_pano_full, 0.1)

    print(f'[INFO] MoGe+Poisson depth range: [{depth_pano_full.min():.2f}m, {depth_pano_full.max():.2f}m]')

    np.save(f"{save_dir}/depth_pano.npy", depth_pano_full)
    depth_normalized = (depth_pano_full - depth_pano_full.min()) / (depth_pano_full.max() - depth_pano_full.min() + 1e-6)
    depth_colored = colorize(depth_normalized, cmap='turbo')
    cv2.imwrite(f"{save_dir}/depth_pano.png", depth_colored[..., :3][..., ::-1])

    print(f'[INFO] Saved: {save_dir}/depth_pano.npy')
    print(f'[INFO] Saved: {save_dir}/depth_pano.png')
    return depth_pano_full


def estimate_wide_depth_moge(image, save_dir, num_iterations=15, num_bins=10, debug=False):
    """
    Estimate depth for a wide perspective image using MoGe V2 directly.
    """
    os.makedirs(save_dir, exist_ok=True)
    h, w = image.shape[:2]
    view_size = min(h, 512)
    overlap = view_size // 2
    stride = view_size - overlap
    num_views = max(1, (w - view_size) // stride + 1)
    view_starts = [min(i * stride, w - view_size) for i in range(num_views)]

    print(f'[INFO] MoGe V2 wide depth: {num_views} views, {num_iterations} alignment iterations')

    depth_arr = np.zeros((num_views, h, view_size), dtype=np.float32)
    for view_i in tqdm(range(num_views), desc="MoGe depth"):
        start = view_starts[view_i]
        crop = image[:, start:start + view_size]
        depth_arr[view_i] = estimate_depth_moge(crop)

    bins_arr = []
    for view_i in range(num_views):
        d_tensor = torch.tensor(depth_arr[view_i])[None, None]
        bins = get_depth_bins(depth=d_tensor, num_bins=num_bins)
        bins_arr.append(bins)

    depth_full = np.zeros((h, w), dtype=np.float32)
    weight_full = np.zeros((h, w), dtype=np.float32)

    for iteration in range(num_iterations):
        depth_full[:] = 0
        weight_full[:] = 0
        for view_i in range(num_views):
            start = view_starts[view_i]
            depth_full[:, start:start + view_size] += depth_arr[view_i]
            weight_full[:, start:start + view_size] += 1
        valid = weight_full > 0
        depth_full[valid] /= weight_full[valid]
        if iteration == num_iterations - 1:
            break
        for view_i in range(num_views):
            start = view_starts[view_i]
            depth_ref = depth_full[:, start:start + view_size]
            depth_curr = depth_arr[view_i]
            try:
                pw = RobustPWRegression()
                pw.fit(depth_curr.flatten(), depth_ref.flatten(), bins_arr[view_i][1:-1])
                depth_arr[view_i] = pw.predict(depth_curr.flatten()).reshape(depth_curr.shape).astype(np.float32)
            except Exception as e:
                if debug:
                    print(f"[WARNING] Alignment failed for view {view_i}: {e}")

    depth_full = np.maximum(depth_full, 0.1)
    print(f'[INFO] MoGe depth range: [{depth_full.min():.2f}m, {depth_full.max():.2f}m]')

    np.save(f"{save_dir}/depth.npy", depth_full)
    depth_normalized = (depth_full - depth_full.min()) / (depth_full.max() - depth_full.min() + 1e-6)
    depth_colored = colorize(depth_normalized, cmap='turbo')
    cv2.imwrite(f"{save_dir}/depth.png", depth_colored[..., :3][..., ::-1])

    print(f'[INFO] Saved: {save_dir}/depth.npy')
    print(f'[INFO] Saved: {save_dir}/depth.png')
    return depth_full


def main():
    parser = argparse.ArgumentParser(
        description='Estimate depth for wide images or cylindrical panoramas'
    )
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='output_depth',
                        help='Output directory')
    parser.add_argument('--mode', type=str, default='wide', choices=['wide', 'panorama'],
                        help='Mode: "wide" for perspective images, "panorama" for 360° cylindrical')
    parser.add_argument('--iterations', type=int, default=15,
                        help='Number of alignment iterations')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='Number of depth bins for alignment')
    parser.add_argument('--fov', type=float, default=44.701948991275390,
                        help='Field of view in degrees (panorama mode only)')
    parser.add_argument('--metric_dataset', type=str, default='vkitti',
                        choices=['vkitti', 'hypersim'],
                        help='DA V2 metric depth model: vkitti (outdoor, 80m) or hypersim (indoor, 20m)')
    parser.add_argument('--method', type=str, default='dav2',
                        choices=['dav2', 'dav2+moge', 'moge', 'moge+poisson'],
                        help='Depth method: '
                             'dav2 = DA V2 relative + DA V2 metric calibration (original), '
                             'dav2+moge = DA V2 relative + MoGe V2 metric calibration, '
                             'moge = MoGe V2 direct metric + ropwr alignment, '
                             'moge+poisson = MoGe V2 metric + Poisson gradient merge (adapted from MoGe panorama)')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug info (large files)')

    args = parser.parse_args()
    
    # Load image
    image = np.array(Image.open(args.input_image).convert('RGB'))
    
    print(f'[INFO] Input image: {args.input_image}')
    print(f'[INFO] Image size: {image.shape[1]}x{image.shape[0]}')
    print(f'[INFO] Mode: {args.mode}')
    print(f'[INFO] Output directory: {args.output_dir}')
    
    print(f'[INFO] Method: {args.method}')

    if args.method == 'moge+poisson':
        assert args.mode == 'panorama', 'moge+poisson only supports panorama mode'
        depth = estimate_panorama_depth_moge_poisson(
            image_pano=image,
            save_dir=args.output_dir,
            num_bins=args.num_bins,
            input_fov=args.fov,
            debug=args.debug
        )
    elif args.method == 'moge':
        if args.mode == 'panorama':
            depth = estimate_panorama_depth_moge(
                image_pano=image,
                save_dir=args.output_dir,
                num_iterations=args.iterations,
                num_bins=args.num_bins,
                input_fov=args.fov,
                debug=args.debug
            )
        else:
            depth = estimate_wide_depth_moge(
                image=image,
                save_dir=args.output_dir,
                num_iterations=args.iterations,
                num_bins=args.num_bins,
                debug=args.debug
            )
    elif args.method == 'dav2+moge':
        if args.mode == 'panorama':
            depth = estimate_panorama_depth(
                image_pano=image,
                save_dir=args.output_dir,
                num_iterations=args.iterations,
                num_bins=args.num_bins,
                input_fov=args.fov,
                metric_model='moge',
                debug=args.debug
            )
        else:
            depth = estimate_wide_depth(
                image=image,
                save_dir=args.output_dir,
                num_iterations=args.iterations,
                num_bins=args.num_bins,
                metric_model='moge',
                debug=args.debug
            )
    else:  # dav2
        if args.mode == 'panorama':
            depth = estimate_panorama_depth(
                image_pano=image,
                save_dir=args.output_dir,
                num_iterations=args.iterations,
                num_bins=args.num_bins,
                input_fov=args.fov,
                metric_dataset=args.metric_dataset,
                debug=args.debug
            )
        else:
            depth = estimate_wide_depth(
                image=image,
                save_dir=args.output_dir,
                num_iterations=args.iterations,
                num_bins=args.num_bins,
                metric_dataset=args.metric_dataset,
                debug=args.debug
            )
    
    print(f'[INFO] Done!')


if __name__ == '__main__':
    main()
