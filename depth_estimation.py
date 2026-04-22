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
from utils.depth_utilsv2 import estimate_depth, estimate_metric_depth, calibrate_relative_depth
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
                        metric_dataset='vkitti', debug=False):
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
                            metric_dataset='vkitti', debug=False):
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

    # Calibrate to metric depth using a reference view
    ref_start = w // 2 - 256
    ref_image = image_pano_tiled[:, ref_start:ref_start + 512]
    ref_tensor = torch.tensor(ref_image).permute(2, 0, 1)[None].to(device).float() / 255.
    ref_persp_np = cyl_proj(ref_tensor, input_focal).cpu().numpy()[0].transpose(1, 2, 0)
    ref_persp_pil = Image.fromarray((ref_persp_np * 255).astype(np.uint8))

    metric_ref = estimate_metric_depth(ref_persp_pil, dataset=metric_dataset)

    ref_disp_cyl = disp_pano_half[:, ref_start:ref_start + 512]
    ref_disp_tensor = torch.tensor(ref_disp_cyl).float()[None, None].to(device)
    ref_disp_persp = cyl_proj(ref_disp_tensor, input_focal).cpu().numpy()[0, 0]
    ref_depth_persp = 1.0 / np.maximum(ref_disp_persp, 0.01)

    scale, bias = calibrate_relative_depth(ref_depth_persp, metric_ref)
    depth_pano_half = scale * depth_pano_half + bias
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
                        help='Metric depth model: vkitti (outdoor, 80m) or hypersim (indoor, 20m)')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug info (large files)')

    args = parser.parse_args()
    
    # Load image
    image = np.array(Image.open(args.input_image).convert('RGB'))
    
    print(f'[INFO] Input image: {args.input_image}')
    print(f'[INFO] Image size: {image.shape[1]}x{image.shape[0]}')
    print(f'[INFO] Mode: {args.mode}')
    print(f'[INFO] Output directory: {args.output_dir}')
    
    if args.mode == 'wide':
        depth = estimate_wide_depth(
            image=image,
            save_dir=args.output_dir,
            num_iterations=args.iterations,
            num_bins=args.num_bins,
            metric_dataset=args.metric_dataset,
            debug=args.debug
        )
    else:  # panorama
        depth = estimate_panorama_depth(
            image_pano=image,
            save_dir=args.output_dir,
            num_iterations=args.iterations,
            num_bins=args.num_bins,
            input_fov=args.fov,
            metric_dataset=args.metric_dataset,
            debug=args.debug
        )
    
    print(f'[INFO] Done!')


if __name__ == '__main__':
    main()
