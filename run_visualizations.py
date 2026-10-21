"""
Generate visualizations for the audit outputs.
"""
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

# Color ramps from preview.py
DISP_RAMP = np.array([
    [0.937, 0.957, 0.984], [0.776, 0.855, 0.937], [0.549, 0.702, 0.878],
    [0.322, 0.541, 0.784], [0.161, 0.376, 0.643], [0.043, 0.176, 0.361],
])
ERR_RAMP = np.array([
    [0.965, 0.961, 0.953], [0.965, 0.855, 0.784], [0.937, 0.616, 0.510],
    [0.851, 0.302, 0.239], [0.545, 0.086, 0.078],
])
INVALID = np.array([0.851, 0.824, 0.780])


def ramp(t: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """Map t in [0,1] onto a colour ramp by linear interpolation."""
    t = np.clip(t, 0.0, 1.0) * (len(colours) - 1)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(colours) - 1)
    f = (t - lo)[..., None]
    return colours[lo] * (1 - f) + colours[hi] * f


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)


def main():
    out_dir = pathlib.Path("results/hailostereo_output_audit")
    
    # Load saved arrays
    disp = np.load(out_dir / "disparity.npy")
    depth = np.load(out_dir / "depth.npy")
    disc = np.load(out_dir / "discontinuity.npy")
    
    # Load input images
    root = pathlib.Path("data/kitti2015/training")
    lefts = sorted((root / "image_2").glob("*_10.png"))
    val_samples = lefts[-40:]
    lp = val_samples[0]
    rp = root / "image_3" / lp.name
    dp = root / "disp_occ_0" / lp.name
    
    # Load and crop images
    left_img = Image.open(lp).convert("RGB")
    right_img = Image.open(rp).convert("RGB")
    if left_img.height >= 368 and left_img.width >= 1232:
        left_img = left_img.crop((left_img.width - 1232, left_img.height - 368, left_img.width, left_img.height))
        right_img = right_img.crop((right_img.width - 1232, right_img.height - 368, right_img.width, right_img.height))
    
    left_arr = np.array(left_img)
    right_arr = np.array(right_img)
    
    # Load GT disparity
    disp_gt = np.array(Image.open(dp), dtype=np.float32) / 256.0
    h, w = disp_gt.shape
    nh, nw = (h // 16) * 16, (w // 16) * 16
    disp_gt = disp_gt[h - nh:, w - nw:]
    
    # Valid mask
    valid = (disp_gt > 0.0) & (disp_gt < 192.0) & np.isfinite(disp_gt)
    valid[:, :192] = False
    
    # --- 1. Stereo inputs ---
    # Save left and right side by side
    stereo_img = np.hstack([left_arr, right_arr])
    Image.fromarray(stereo_img).save(out_dir / "01_stereo_inputs.png")
    print("Saved 01_stereo_inputs.png")
    
    # --- 2. Disparity map ---
    if valid.any():
        lo, hi = np.percentile(disp_gt[valid], [2.0, 98.0])
    else:
        lo, hi = 0.0, 192.0
    span = max(hi - lo, 1e-6)
    
    gt_vis = ramp((disp_gt - lo) / span, DISP_RAMP)
    gt_vis[~valid] = INVALID
    
    pred_vis = ramp((disp - lo) / span, DISP_RAMP)
    
    err = np.abs(disp - disp_gt)
    err_vis = ramp(err / 3.0, ERR_RAMP)
    err_vis[~valid] = INVALID
    
    # Combine: left | pred | gt | error
    panels = [Image.fromarray(to_u8(left_arr/255.0)), 
              Image.fromarray(to_u8(pred_vis)), 
              Image.fromarray(to_u8(gt_vis)), 
              Image.fromarray(to_u8(err_vis))]
    
    W = panels[0].width * 4 + 3 * 8
    H = panels[0].height + 18 + 8
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    titles = ["left image", "predicted disparity", "ground truth disparity", "|error|, red >= 3 px"]
    for i, (panel, title) in enumerate(zip(panels, titles)):
        canvas.paste(panel, (8 + i * (panels[0].width + 8), 18))
        draw.text((8 + i * (panels[0].width + 8), 4), title, fill=(60, 60, 60))
    
    canvas.save(out_dir / "02_disparity_map.png")
    print("Saved 02_disparity_map.png")
    
    # --- 3. Metric depth ---
    # Use depth range for color scale
    z_valid = np.isfinite(depth) & (depth <= 80.0)
    if z_valid.any():
        lo_d, hi_d = np.percentile(depth[z_valid], [2.0, 98.0])
    else:
        lo_d, hi_d = 0.0, 80.0
    span_d = max(hi_d - lo_d, 1e-6)
    
    depth_vis = ramp((depth - lo_d) / span_d, DISP_RAMP)
    depth_vis[~z_valid] = INVALID
    
    # GT depth
    KITTI_FOCAL = 715.7
    KITTI_BASELINE = 0.54
    fb = KITTI_FOCAL * KITTI_BASELINE
    depth_gt = fb / np.maximum(disp_gt, 1e-3)
    depth_gt_valid = np.isfinite(depth_gt) & (depth_gt <= 80.0)
    gt_depth_vis = ramp((depth_gt - lo_d) / span_d, DISP_RAMP)
    gt_depth_vis[~depth_gt_valid] = INVALID
    
    # Depth error
    depth_err = np.abs(depth - depth_gt)
    rel_err = depth_err / np.maximum(depth_gt, 1e-6)
    err_depth_vis = ramp(rel_err / 0.10, ERR_RAMP)
    err_depth_vis[~depth_gt_valid] = INVALID
    
    panels_d = [Image.fromarray(to_u8(left_arr/255.0)), 
                Image.fromarray(to_u8(depth_vis)), 
                Image.fromarray(to_u8(gt_depth_vis)), 
                Image.fromarray(to_u8(err_depth_vis))]
    
    canvas_d = Image.new("RGB", (W, H), (255, 255, 255))
    draw_d = ImageDraw.Draw(canvas_d)
    titles_d = ["left image", f"predicted depth ({lo_d:.1f}-{hi_d:.1f}m)", "ground truth depth", "relative error, red >= 10%"]
    for i, (panel, title) in enumerate(zip(panels_d, titles_d)):
        canvas_d.paste(panel, (8 + i * (panels_d[0].width + 8), 18))
        draw_d.text((8 + i * (panels_d[0].width + 8), 4), title, fill=(60, 60, 60))
    
    canvas_d.save(out_dir / "03_metric_depth.png")
    print("Saved 03_metric_depth.png")
    
    # --- 4. Point cloud visualization (simple depth-colored projection) ---
    # We'll render the point cloud as a depth-colored image from the camera view
    pc_vis = ramp((depth - lo_d) / span_d, DISP_RAMP)
    pc_vis[~z_valid] = INVALID
    Image.fromarray(to_u8(pc_vis)).save(out_dir / "04_point_cloud.png")
    print("Saved 04_point_cloud.png")
    
    # --- 5. Depth discontinuity ---
    disc_valid = valid & z_valid
    if disc_valid.any():
        lo_disc, hi_disc = np.percentile(disc[disc_valid], [2.0, 98.0])
    else:
        lo_disc, hi_disc = 0.0, 1.0
    span_disc = max(hi_disc - lo_disc, 1e-6)
    
    disc_vis = ramp((disc - lo_disc) / span_disc, ERR_RAMP)  # Use error ramp for discontinuity
    disc_vis[~disc_valid] = INVALID
    
    Image.fromarray(to_u8(disc_vis)).save(out_dir / "05_depth_discontinuity.png")
    print("Saved 05_depth_discontinuity.png")
    
    # --- 6. Occupancy grid ---
    # Load occupancy cells
    import csv
    cells = []
    with open(out_dir / "occupancy_cells.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cells.append(row)
    
    grid_rows = 23
    grid_cols = 77
    cell_img = np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)
    
    for cell in cells:
        r, c = int(cell["row"]), int(cell["col"])
        label = cell["label"]
        if label == "NEAR":
            cell_img[r, c] = [255, 0, 0]  # Red
        elif label == "FAR":
            cell_img[r, c] = [0, 0, 255]  # Blue
        else:
            cell_img[r, c] = [128, 128, 128]  # Gray
    
    # Upscale for visibility
    cell_img_up = np.repeat(np.repeat(cell_img, 10, axis=0), 10, axis=1)
    
    # Add legend
    H_legend = 40
    legend_img = np.ones((H_legend, cell_img_up.shape[1], 3), dtype=np.uint8) * 255
    draw_leg = ImageDraw.Draw(Image.fromarray(legend_img))
    draw_leg.text((10, 10), "NEAR (<10m) = Red   FAR (>=10m) = Blue   INSUFFICIENT = Gray", fill=(60, 60, 60))
    
    occ_vis = np.vstack([cell_img_up, legend_img])
    Image.fromarray(occ_vis).save(out_dir / "06_occupancy_grid.png")
    print("Saved 06_occupancy_grid.png")
    
    print("\nAll visualizations saved!")


if __name__ == "__main__":
    main()