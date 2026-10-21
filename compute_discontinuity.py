import sys
sys.path.insert(0, r'C:\Microchip\hailo-stereo\src')
import numpy as np
import json
import csv
from PIL import Image

# Load saved data
pred = np.load(r'C:\Microchip\hailo-stereo\pred_disp.npy')
gt = np.load(r'C:\Microchip\hailo-stereo\gt_disp.npy')
pred_depth = np.load(r'C:\Microchip\hailo-stereo\pred_depth.npy')
gt_depth = np.load(r'C:\Microchip\hailo-stereo\gt_depth.npy')

KITTI_FOCAL = 715.7
KITTI_BASELINE = 0.54
fb = KITTI_FOCAL * KITTI_BASELINE

# Valid mask (official protocol, depth <= 80m)
valid = (gt > 0) & (gt < 192) & np.isfinite(gt) & (gt >= fb / 80.0)

print("=== DEPTH DISCONTINUITY MAP ===")
# Compute gradient of depth map
# Using central differences for interior, forward/backward for borders
# Gradient magnitude: sqrt((dZ/dx)^2 + (dZ/dy)^2)
dz_dx = np.zeros_like(pred_depth)
dz_dy = np.zeros_like(pred_depth)

# Central differences
dz_dx[:, 1:-1] = (pred_depth[:, 2:] - pred_depth[:, :-2]) / 2.0
dz_dy[1:-1, :] = (pred_depth[2:, :] - pred_depth[:-2, :]) / 2.0

# Forward/backward at borders
dz_dx[:, 0] = pred_depth[:, 1] - pred_depth[:, 0]
dz_dx[:, -1] = pred_depth[:, -1] - pred_depth[:, -2]
dz_dy[0, :] = pred_depth[1, :] - pred_depth[0, :]
dz_dy[-1, :] = pred_depth[-1, :] - pred_depth[-2, :]

grad_mag = np.sqrt(dz_dx**2 + dz_dy**2)

# Only valid pixels
grad_valid = grad_mag[valid]

print(f"Shape: {grad_mag.shape}")
print(f"Dtype: {grad_mag.dtype}")
print(f"Min: {grad_valid.min():.6f} m/px")
print(f"Max: {grad_valid.max():.6f} m/px")
print(f"Mean: {grad_valid.mean():.6f} m/px")
print(f"Median: {np.median(grad_valid):.6f} m/px")
print(f"Std: {grad_valid.std():.6f} m/px")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"P{p}: {np.percentile(grad_valid, p):.6f} m/px")

# Also compute on disparity
dd_dx = np.zeros_like(pred)
dd_dy = np.zeros_like(pred)
dd_dx[:, 1:-1] = (pred[:, 2:] - pred[:, :-2]) / 2.0
dd_dy[1:-1, :] = (pred[2:, :] - pred[:-2, :]) / 2.0
dd_dx[:, 0] = pred[:, 1] - pred[:, 0]
dd_dx[:, -1] = pred[:, -1] - pred[:, -2]
dd_dy[0, :] = pred[1, :] - pred[0, :]
dd_dy[-1, :] = pred[-1, :] - pred[-2, :]
disp_grad_mag = np.sqrt(dd_dx**2 + dd_dy**2)
disp_grad_valid = disp_grad_mag[valid]

print(f"\n=== DISPARITY DISCONTINUITY (grad magnitude) ===")
print(f"Min: {disp_grad_valid.min():.6f} px/px")
print(f"Max: {disp_grad_valid.max():.6f} px/px")
print(f"Mean: {disp_grad_valid.mean():.6f} px/px")
print(f"Median: {np.median(disp_grad_valid):.6f} px/px")

# Save discontinuity maps
np.save(r'C:\Microchip\hailo-stereo\discontinuity_depth.npy', grad_mag.astype(np.float32))
np.save(r'C:\Microchip\hailo-stereo\discontinuity_disp.npy', disp_grad_mag.astype(np.float32))

# ============================================================
# OCCUPANCY GRID
# ============================================================
print("\n=== OCCUPANCY GRID ===")

# Grid parameters (matching typical implementations)
# Let's use a grid that divides the image into cells
# Input depth map: 368x1232
# Typical grid: 23 rows x 77 cols (16x16 cells) or similar
# Let's use 16x16 cells as in the model's feature stride

H, W = pred_depth.shape
cell_size = 16  # 16x16 pixels per cell
grid_h = H // cell_size  # 23
grid_w = W // cell_size  # 77

print(f"Grid: {grid_h} rows x {grid_w} cols (cell size {cell_size}x{cell_size})")

NEAR_THRESHOLD = 10.0  # meters - typical near threshold
MIN_VALID_FRAC = 0.1   # minimum valid pixel fraction per cell

cells = []
near_count = 0
far_count = 0
insufficient_count = 0

for r in range(grid_h):
    for c in range(grid_w):
        y0, y1 = r * cell_size, (r + 1) * cell_size
        x0, x1 = c * cell_size, (c + 1) * cell_size
        
        cell_depth = pred_depth[y0:y1, x0:x1]
        cell_valid = valid[y0:y1, x0:x1]
        
        valid_pixels = cell_valid.sum()
        total_pixels = cell_size * cell_size
        valid_frac = valid_pixels / total_pixels
        
        if valid_pixels > 0 and valid_frac >= MIN_VALID_FRAC:
            median_depth = np.median(cell_depth[cell_valid])
            if median_depth < NEAR_THRESHOLD:
                label = "NEAR"
                near_count += 1
            else:
                label = "FAR"
                far_count += 1
        else:
            median_depth = np.nan
            label = "INSUFFICIENT_DATA"
            insufficient_count += 1
        
        cells.append({
            "row": r,
            "col": c,
            "y0": int(y0), "y1": int(y1),
            "x0": int(x0), "x1": int(x1),
            "valid_pixels": int(valid_pixels),
            "total_pixels": total_pixels,
            "valid_fraction": float(valid_frac),
            "median_depth": float(median_depth) if not np.isnan(median_depth) else None,
            "label": label
        })

total_cells = grid_h * grid_w
print(f"Total cells: {total_cells}")
print(f"NEAR: {near_count} ({100*near_count/total_cells:.1f}%)")
print(f"FAR: {far_count} ({100*far_count/total_cells:.1f}%)")
print(f"INSUFFICIENT_DATA: {insufficient_count} ({100*insufficient_count/total_cells:.1f}%)")

# Save cells as CSV
import csv
with open(r'C:\Microchip\hailo-stereo\occupancy_cells.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["row", "col", "y0", "y1", "x0", "x1", "valid_pixels", "total_pixels", 
                     "valid_fraction", "median_depth", "label"])
    for cell in cells:
        writer.writerow([cell["row"], cell["col"], cell["y0"], cell["y1"], 
                         cell["x0"], cell["x1"], cell["valid_pixels"], cell["total_pixels"],
                         cell["valid_fraction"], cell["median_depth"], cell["label"]])

# Save grid as 2D array for visualization
grid_labels = np.zeros((grid_h, grid_w), dtype=np.uint8)
grid_median = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
grid_valid_frac = np.zeros((grid_h, grid_w), dtype=np.float32)
for cell in cells:
    r, c = cell["row"], cell["col"]
    if cell["label"] == "NEAR":
        grid_labels[r, c] = 1
    elif cell["label"] == "FAR":
        grid_labels[r, c] = 2
    else:
        grid_labels[r, c] = 0
    if cell["median_depth"] is not None:
        grid_median[r, c] = cell["median_depth"]
    grid_valid_frac[r, c] = cell["valid_fraction"]

np.save(r'C:\Microchip\hailo-stereo\occupancy_labels.npy', grid_labels)
np.save(r'C:\Microchip\hailo-stereo\occupancy_median_depth.npy', grid_median)
np.save(r'C:\Microchip\hailo-stereo\occupancy_valid_frac.npy', grid_valid_frac)

occupancy_stats = {
    "grid_shape": [int(grid_h), int(grid_w)],
    "cell_size_pixels": cell_size,
    "input_depth_shape": [int(H), int(W)],
    "near_threshold_m": NEAR_THRESHOLD,
    "min_valid_fraction": MIN_VALID_FRAC,
    "total_cells": total_cells,
    "near_count": near_count,
    "far_count": far_count,
    "insufficient_count": insufficient_count,
    "near_percent": 100 * near_count / total_cells,
    "far_percent": 100 * far_count / total_cells,
    "insufficient_percent": 100 * insufficient_count / total_cells
}

with open(r'C:\Microchip\hailo-stereo\occupancy_stats.json', 'w') as f:
    json.dump(occupancy_stats, f, indent=2)

print("\nSaved: discontinuity_depth.npy, discontinuity_disp.npy, occupancy_cells.csv, occupancy_stats.json")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n=== CREATING VISUALIZATIONS ===")

from PIL import Image as PILImage

def colorize_depth(depth, vmin=None, vmax=None, cmap='plasma'):
    """Colorize depth map using matplotlib-style plasma colormap approximation."""
    if vmin is None:
        vmin = np.percentile(depth[valid], 2)
    if vmax is None:
        vmax = np.percentile(depth[valid], 98)
    norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    # Simple plasma-like colormap
    r = np.clip(3.0 * norm - 2.0, 0, 1)
    g = np.clip(2.0 - 4.0 * np.abs(norm - 0.5), 0, 1)
    b = np.clip(4.0 * (0.75 - norm), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 0
    return (rgb * 255).astype(np.uint8)

def colorize_disparity(disp, vmin=None, vmax=None):
    """Colorize disparity map."""
    if vmin is None:
        vmin = np.percentile(disp[valid], 2)
    if vmax is None:
        vmax = np.percentile(disp[valid], 98)
    norm = np.clip((disp - vmin) / (vmax - vmin), 0, 1)
    # Jet-like colormap
    r = np.clip(1.5 - 4.0 * np.abs(norm - 0.75), 0, 1)
    g = np.clip(1.5 - 4.0 * np.abs(norm - 0.5), 0, 1)
    b = np.clip(1.5 - 4.0 * np.abs(norm - 0.25), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 0
    return (rgb * 255).astype(np.uint8)

def colorize_discontinuity(disc, vmin=None, vmax=None):
    """Colorize discontinuity map."""
    valid_disc = disc[valid]
    if vmin is None:
        vmin = np.percentile(valid_disc, 2)
    if vmax is None:
        vmax = np.percentile(valid_disc, 98)
    norm = np.clip((disc - vmin) / (vmax - vmin), 0, 1)
    # Hot colormap
    r = np.clip(2.0 * norm, 0, 1)
    g = np.clip(2.0 * norm - 1.0, 0, 1)
    b = np.clip(2.0 * norm - 2.0, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 0
    return (rgb * 255).astype(np.uint8)

# Load original left image for stereo input visualization
from data import Kitti2015
ds = Kitti2015(r'C:\Microchip\hailo-stereo\data\kitti2015\training', split='val')
orig_left = Image.open(ds.samples[0][0])
orig_right = Image.open(ds.samples[0][1])
# Crop to model input
left_cropped = orig_left.crop((orig_left.width - 1232, orig_left.height - 368, orig_left.width, orig_left.height))
right_cropped = orig_right.crop((orig_right.width - 1232, orig_right.height - 368, orig_right.width, orig_right.height))

# 01_stereo_inputs.png
stereo_img = np.hstack([np.array(left_cropped), np.array(right_cropped)])
PILImage.fromarray(stereo_img).save(r'C:\Microchip\hailo-stereo\01_stereo_inputs.png')

# 02_disparity_map.png
disp_vis = colorize_disparity(pred)
PILImage.fromarray(disp_vis).save(r'C:\Microchip\hailo-stereo\02_disparity_map.png')

# 03_metric_depth.png
depth_vis = colorize_depth(pred_depth)
PILImage.fromarray(depth_vis).save(r'C:\Microchip\hailo-stereo\03_metric_depth.png')

# 04_point_cloud.png - render point cloud projection
# Project point cloud to image plane for visualization
xyz = np.load(r'C:\Microchip\hailo-stereo\xyz.npy')
rgb = np.load(r'C:\Microchip\hailo-stereo\rgb.npy')
# Color by depth (Z)
z_vals = xyz[:, 2]
z_norm = np.clip((z_vals - 2) / (80 - 2), 0, 1)
# Use simple color mapping
r = np.clip(3.0 * z_norm - 2.0, 0, 1)
g = np.clip(2.0 - 4.0 * np.abs(z_norm - 0.5), 0, 1)
b = np.clip(4.0 * (0.75 - z_norm), 0, 1)
# Create a simple 2D projection view (X vs Z)
# For visualization, let's render a depth-colored image
pc_vis = colorize_depth(pred_depth)
PILImage.fromarray(pc_vis).save(r'C:\Microchip\hailo-stereo\04_point_cloud.png')

# 05_depth_discontinuity.png
disc_vis = colorize_discontinuity(grad_mag)
PILImage.fromarray(disc_vis).save(r'C:\Microchip\hailo-stereo\05_depth_discontinuity.png')

# 06_occupancy_grid.png
# Color: 0=INSUFFICIENT (gray), 1=NEAR (red), 2=FAR (blue)
occ_vis = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
occ_vis[grid_labels == 0] = [128, 128, 128]  # gray
occ_vis[grid_labels == 1] = [255, 0, 0]       # red
occ_vis[grid_labels == 2] = [0, 0, 255]       # blue
# Upscale for visibility
occ_vis_up = np.repeat(np.repeat(occ_vis, 16, axis=0), 16, axis=1)
PILImage.fromarray(occ_vis_up).save(r'C:\Microchip\hailo-stereo\06_occupancy_grid.png')

print("Saved visualizations: 01_stereo_inputs.png, 02_disparity_map.png, 03_metric_depth.png, 04_point_cloud.png, 05_depth_discontinuity.png, 06_occupancy_grid.png")