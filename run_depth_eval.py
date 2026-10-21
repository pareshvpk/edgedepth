"""
Run depth evaluation using the ONNX model with onnxruntime.
"""
import argparse
import pathlib
import sys
import json

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
from data import MEAN, STD, Kitti2015

# KITTI 2015 nominal rectified intrinsics
KITTI_FOCAL = 715.7          # px
KITTI_BASELINE = 0.54        # m

# Range bands, in metres
BANDS = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 80.0))

# The prediction is clamped to [0, MAX_DISP] and the sky region is untrained, so
# disparity can arrive at or near zero. Floor it rather than divide by it.
EPS_DISP = 1e-3

DELTA = 1.25

ONNX_PATH = pathlib.Path(__file__).resolve().parent / "artifacts" / "hailo_stereo.onnx"


def depth_from_disp(disp: np.ndarray, fb: float) -> np.ndarray:
    """Z = f x B / d, with d floored so the reciprocal cannot blow up."""
    return fb / np.maximum(disp, EPS_DISP)


def load_and_preprocess(img_path):
    """Load image and preprocess to match the model's expected input."""
    img = Image.open(img_path).convert("RGB")
    # Crop to 368x1232 bottom-right as the model expects
    if img.height >= 368 and img.width >= 1232:
        img = img.crop((img.width - 1232, img.height - 368, img.width, img.height))
    else:
        # Resize to 1232 wide then crop
        s = 1232 / img.width
        new_h = max(368, round(img.height * s))
        img = img.resize((1232, new_h), Image.BILINEAR)
        img = img.crop((0, img.height - 368, 1232, img.height))
    
    img_arr = np.array(img, dtype=np.float32) / 255.0
    # Normalize
    img_arr = ((img_arr - MEAN) / STD).transpose(2, 0, 1)  # CHW
    return img_arr[None, ...].astype(np.float32)  # Add batch dim


def main():
    # Use the first validation pair
    root = pathlib.Path("data/kitti2015/training")
    ds = Kitti2015(root, split="val")
    
    print(f"Found {len(ds)} validation samples")
    
    # Load ONNX model
    print(f"Loading ONNX model from {ONNX_PATH}")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f"Inputs: {input_names}")
    print(f"Outputs: {output_names}")
    
    fb = KITTI_FOCAL * KITTI_BASELINE
    print(f"f x B = {KITTI_FOCAL:.1f} px x {KITTI_BASELINE:.3f} m = {fb:.2f} px.m")
    
    # Process first validation sample
    sample = ds[0]
    left = sample["left"][None].numpy()
    right = sample["right"][None].numpy()
    disp_gt = sample["disp"].numpy()[0]
    mask = sample["mask"].numpy()[0] > 0.5
    official_mask = disp_gt > 0.0
    
    print(f"Input shapes: left={left.shape}, right={right.shape}")
    print(f"GT shape: {disp_gt.shape}")
    print(f"Masked valid: {mask.sum()}, Official valid: {official_mask.sum()}")
    
    # Run inference
    outputs = sess.run(output_names, {"left": left, "right": right})
    pred = outputs[0][0, 0]  # Remove batch and channel dims
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
    
    # Convert to depth
    z_pred_full = depth_from_disp(pred, fb)
    z_gt_full = depth_from_disp(disp_gt, fb)
    
    print(f"\nDepth prediction range: [{z_pred_full[np.isfinite(z_pred_full)].min():.4f}, {z_pred_full.max():.4f}] m")
    print(f"Depth GT range: [{z_gt_full[np.isfinite(z_gt_full)].min():.4f}, {z_gt_full.max():.4f}] m")
    
    # Save raw depth
    np.save("results/hailostereo_output_audit/depth.npy", z_pred_full.astype(np.float32))
    print("Saved depth.npy")
    
    # Statistics on full depth prediction
    valid_depth = np.isfinite(z_pred_full)
    print(f"\n=== Depth Statistics (full prediction) ===")
    print(f"Valid pixels: {valid_depth.sum()} ({100*valid_depth.sum()/z_pred_full.size:.2f}%)")
    print(f"Min: {z_pred_full[valid_depth].min():.4f} m")
    print(f"Max: {z_pred_full[valid_depth].max():.4f} m")
    print(f"Mean: {z_pred_full[valid_depth].mean():.4f} m")
    print(f"Median: {np.median(z_pred_full[valid_depth]):.4f} m")
    print(f"Std: {z_pred_full[valid_depth].std():.4f} m")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"P{p}: {np.percentile(z_pred_full[valid_depth], p):.4f} m")
    
    # Depth band counts
    print(f"\n=== Depth Band Distribution ===")
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, float('inf'))]:
        if hi == float('inf'):
            count = (z_pred_full >= lo).sum()
        else:
            count = ((z_pred_full >= lo) & (z_pred_full < hi)).sum()
        print(f"  {lo}-{hi if hi != float('inf') else 'inf'} m: {count} ({100*count/z_pred_full.size:.2f}%)")
    
    # Per-band metrics
    protocols = ("masked", "official")
    print(f"\n=== Per-band Metrics ===")
    
    results = {}
    for proto_name, valid_mask in [("masked", mask), ("official", official_mask)]:
        if not valid_mask.any():
            continue
            
        v = valid_mask & (disp_gt > fb / 80.0)  # in-range for depth tables (GT depth <= 80m)
        z_pred = z_pred_full[v]
        z_gt = z_gt_full[v]
        derr = pred[v] - disp_gt[v]
        
        print(f"\n--- {proto_name.upper()} Protocol ---")
        print(f"Evaluated pixels: {v.sum()}")
        
        results[proto_name] = {"bands": {}}
        
        for lo, hi in BANDS:
            sel = (z_gt >= lo) & (z_gt < hi)
            if not sel.any():
                print(f"  {lo}-{hi} m: No ground truth pixels")
                continue
                
            zp = z_pred[sel]
            zg = z_gt[sel]
            de = derr[sel]
            
            d = np.abs(zp - zg)
            rel = d / zg
            ratio = np.maximum(zp / zg, zg / np.maximum(zp, EPS_DISP))
            delta_ok = (ratio < DELTA).sum()
            
            z_mean = float(zg.mean())
            epe = float(np.abs(de).mean())
            floor = z_mean * z_mean / fb * epe
            
            print(f"  {lo}-{hi} m: {sel.sum()} pixels")
            print(f"    MAE: {d.mean():.4f} m")
            print(f"    RMSE: {np.sqrt((d.astype(np.float64) ** 2).mean()):.4f} m")
            print(f"    Median: {np.median(d):.4f} m")
            print(f"    AbsRel: {100.0 * rel.mean():.2f}%")
            print(f"    Delta < 1.25: {100.0 * delta_ok / sel.sum():.2f}%")
            print(f"    EPE: {epe:.4f} px")
            print(f"    Floor: {floor:.4f} m")
            print(f"    Bias dd: {de.mean():+.4f} px")
            print(f"    Bias dZ: {(zp - zg).mean():+.4f} m")
            
            results[proto_name]["bands"][f"{lo}-{hi}"] = {
                "pixels": int(sel.sum()),
                "mae": float(d.mean()),
                "rmse": float(np.sqrt((d.astype(np.float64) ** 2).mean())),
                "median": float(np.median(d)),
                "absrel": float(100.0 * rel.mean()),
                "delta": float(100.0 * delta_ok / sel.sum()),
                "epe": float(epe),
                "floor": float(floor),
                "dbias": float(de.mean()),
                "zbias": float((zp - zg).mean()),
                "z_mean": z_mean
            }
    
    # Save stats
    depth_stats = {
        "focal_px": KITTI_FOCAL,
        "baseline_m": KITTI_BASELINE,
        "fb_px_m": fb,
        "full_stats": {
            "valid_pixels": int(valid_depth.sum()),
            "total_pixels": int(z_pred_full.size),
            "min": float(z_pred_full[valid_depth].min()),
            "max": float(z_pred_full[valid_depth].max()),
            "mean": float(z_pred_full[valid_depth].mean()),
            "median": float(np.median(z_pred_full[valid_depth])),
            "std": float(z_pred_full[valid_depth].std()),
            "percentiles": {f"P{p}": float(np.percentile(z_pred_full[valid_depth], p)) for p in [1, 5, 25, 50, 75, 95, 99]},
            "depth_bands": {f"{lo}-{hi if hi != float('inf') else 'inf'}": int(((z_pred_full >= lo) & (z_pred_full < hi)).sum()) for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, float('inf'))]}
        },
        "band_metrics": results
    }
    
    with open("results/hailostereo_output_audit/depth_stats.json", "w") as f:
        json.dump(depth_stats, f, indent=2)
    print("\nSaved depth_stats.json")
    
    # Also save band metrics as CSV
    import csv
    with open("results/hailostereo_output_audit/depth_band_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["protocol", "band", "pixels", "mae_m", "rmse_m", "median_m", "absrel_pct", "delta_pct", "epe_px", "floor_m", "dbias_px", "zbias_m", "z_mean_m"])
        for proto in protocols:
            if proto in results:
                for band, m in results[proto]["bands"].items():
                    writer.writerow([proto, band, m["pixels"], m["mae"], m["rmse"], m["median"], m["absrel"], m["delta"], m["epe"], m["floor"], m["dbias"], m["zbias"], m["z_mean"]])
    print("Saved depth_band_metrics.csv")


if __name__ == "__main__":
    main()