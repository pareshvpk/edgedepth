"""
Generate occupancy grid / spatial map from depth prediction.
"""
import pathlib
import sys
import json
import csv

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

ONNX_PATH = pathlib.Path(__file__).resolve().parent / "artifacts" / "hailo_stereo.onnx"


def load_and_preprocess(img_path):
    """Load image and preprocess to match the model's expected input."""
    img = Image.open(img_path).convert("RGB")
    if img.height >= 368 and img.width >= 1232:
        img = img.crop((img.width - 1232, img.height - 368, img.width, img.height))
    else:
        s = 1232 / img.width
        new_h = max(368, round(img.height * s))
        img = img.resize((1232, new_h), Image.BILINEAR)
        img = img.crop((0, img.height - 368, 1232, img.height))
    
    img_arr = np.array(img, dtype=np.float32) / 255.0
    from src.data import MEAN, STD
    img_arr = ((img_arr - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    return img_arr, np.array(img)


def main():
    # Use the first validation pair
    root = pathlib.Path("data/kitti2015/training")
    lefts = sorted((root / "image_2").glob("*_10.png"))
    val_samples = lefts[-40:]
    
    print(f"Found {len(val_samples)} validation samples")
    
    # Load ONNX model
    print(f"Loading ONNX model from {ONNX_PATH}")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    
    # Process first validation sample
    lp = val_samples[0]
    rp = root / "image_3" / lp.name
    dp = root / "disp_occ_0" / lp.name
    
    print(f"\nProcessing {lp.name}")
    
    # Load and preprocess
    left_norm, left_img = load_and_preprocess(lp)
    right_norm, _ = load_and_preprocess(rp)
    
    # Load GT disparity
    disp_gt = np.array(Image.open(dp), dtype=np.float32) / 256.0
    h, w = disp_gt.shape
    nh, nw = (h // 16) * 16, (w // 16) * 16
    disp_gt = disp_gt[h - nh:, w - nw:]
    
    # Run inference
    outputs = sess.run(output_names, {"left": left_norm, "right": right_norm})
    disp = outputs[0][0, 0]
    
    # Camera params for depth
    KITTI_FOCAL = 715.7
    KITTI_BASELINE = 0.54
    fb = KITTI_FOCAL * KITTI_BASELINE
    
    # Convert to depth
    EPS_DISP = 1e-3
    z_pred = fb / np.maximum(disp, EPS_DISP)
    
    # Valid mask
    valid = (disp_gt > 0.0) & (disp_gt < 192.0) & np.isfinite(disp_gt)
    valid[:, :192] = False
    
    # Occupancy grid parameters
    grid_rows = 23  # 368 / 16 = 23
    grid_cols = 77  # 1232 / 16 = 77
    cell_h = 16
    cell_w = 16
    near_threshold = 10.0  # metres
    min_valid_frac = 0.1   # minimum valid pixel fraction
    
    h, w = z_pred.shape
    print(f"Depth map: {h}x{w}")
    print(f"Grid: {grid_rows}x{grid_cols} cells of {cell_h}x{cell_w}")
    print(f"Near threshold: {near_threshold} m")
    print(f"Min valid fraction: {min_valid_frac}")
    
    cells = []
    near_count = 0
    far_count = 0
    insufficient_count = 0
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            
            cell_depth = z_pred[y0:y1, x0:x1]
            cell_valid = valid[y0:y1, x0:x1]
            
            valid_pixels = cell_valid.sum()
            total_pixels = cell_valid.size
            valid_frac = valid_pixels / total_pixels
            
            if valid_pixels > 0:
                median_depth = float(np.median(cell_depth[cell_valid]))
            else:
                median_depth = float('inf')
            
            # Classification
            if valid_frac < min_valid_frac:
                label = "INSUFFICIENT_DATA"
                insufficient_count += 1
            elif median_depth < near_threshold:
                label = "NEAR"
                near_count += 1
            else:
                label = "FAR"
                far_count += 1
            
            cells.append({
                "row": r,
                "col": c,
                "y0": y0, "y1": y1,
                "x0": x0, "x1": x1,
                "valid_pixels": int(valid_pixels),
                "total_pixels": int(total_pixels),
                "valid_fraction": valid_frac,
                "median_depth_m": median_depth if np.isfinite(median_depth) else None,
                "label": label
            })
    
    print(f"\n=== Occupancy Grid Summary ===")
    print(f"Total cells: {len(cells)}")
    print(f"NEAR (< {near_threshold} m): {near_count} ({100*near_count/len(cells):.1f}%)")
    print(f"FAR (>= {near_threshold} m): {far_count} ({100*far_count/len(cells):.1f}%)")
    print(f"INSUFFICIENT_DATA: {insufficient_count} ({100*insufficient_count/len(cells):.1f}%)")
    
    # Save CSV
    with open("results/hailostereo_output_audit/occupancy_cells.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "y0", "y1", "x0", "x1", "valid_pixels", "total_pixels", 
                         "valid_fraction", "median_depth_m", "label"])
        for cell in cells:
            writer.writerow([cell["row"], cell["col"], cell["y0"], cell["y1"], 
                           cell["x0"], cell["x1"], cell["valid_pixels"], cell["total_pixels"],
                           cell["valid_fraction"], cell["median_depth_m"], cell["label"]])
    print("Saved occupancy_cells.csv")
    
    # Save stats
    stats = {
        "depth_map_shape": [h, w],
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "cell_height": cell_h,
        "cell_width": cell_w,
        "near_threshold_m": near_threshold,
        "min_valid_fraction": min_valid_frac,
        "total_cells": len(cells),
        "near_count": near_count,
        "far_count": far_count,
        "insufficient_data_count": insufficient_count,
        "near_percent": 100.0 * near_count / len(cells),
        "far_percent": 100.0 * far_count / len(cells),
        "insufficient_percent": 100.0 * insufficient_count / len(cells),
        "classification_rule": "median depth < near_threshold -> NEAR, else FAR; valid_fraction < min_valid_frac -> INSUFFICIENT_DATA"
    }
    
    with open("results/hailostereo_output_audit/occupancy_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved occupancy_stats.json")


if __name__ == "__main__":
    main()