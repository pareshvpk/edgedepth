"""
Generate depth discontinuity map from disparity prediction.
"""
import pathlib
import sys
import json

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


def compute_discontinuity(depth: np.ndarray) -> np.ndarray:
    """
    Compute depth discontinuity map using finite differences.
    
    Operates on metric depth (Z). Uses central differences in horizontal and vertical
    directions, then computes gradient magnitude.
    
    grad_x = (Z[:, 2:] - Z[:, :-2]) / 2
    grad_y = (Z[2:, :] - Z[:-2, :]) / 2
    magnitude = sqrt(grad_x^2 + grad_y^2)
    
    The result has the same shape as input, with borders handled by forward/backward differences.
    """
    h, w = depth.shape
    disc = np.zeros_like(depth)
    
    # Horizontal gradient (central difference, forward/backward at borders)
    grad_x = np.zeros_like(depth)
    grad_x[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) / 2.0
    grad_x[:, 0] = depth[:, 1] - depth[:, 0]  # forward diff at left border
    grad_x[:, -1] = depth[:, -1] - depth[:, -2]  # backward diff at right border
    
    # Vertical gradient (central difference, forward/backward at borders)
    grad_y = np.zeros_like(depth)
    grad_y[1:-1, :] = (depth[2:, :] - depth[:-2, :]) / 2.0
    grad_y[0, :] = depth[1, :] - depth[0, :]  # forward diff at top border
    grad_y[-1, :] = depth[-1, :] - depth[-2, :]  # backward diff at bottom border
    
    # Gradient magnitude
    disc = np.sqrt(grad_x**2 + grad_y**2)
    
    return disc


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
    z_gt = fb / np.maximum(disp_gt, EPS_DISP)
    
    # Compute discontinuity on predicted depth
    disc_pred = compute_discontinuity(z_pred)
    disc_gt = compute_discontinuity(z_gt)
    
    # Valid mask
    valid = (disp_gt > 0.0) & (disp_gt < 192.0) & np.isfinite(disp_gt)
    valid[:, :192] = False
    
    print(f"\n=== Discontinuity Statistics (Predicted Depth) ===")
    print(f"Shape: {disc_pred.shape}")
    print(f"Valid pixels: {valid.sum()} ({100*valid.sum()/disc_pred.size:.2f}%)")
    print(f"Min: {disc_pred[valid].min():.6f} m/px")
    print(f"Max: {disc_pred[valid].max():.6f} m/px")
    print(f"Mean: {disc_pred[valid].mean():.6f} m/px")
    print(f"Median: {np.median(disc_pred[valid]):.6f} m/px")
    print(f"Std: {disc_pred[valid].std():.6f} m/px")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"P{p}: {np.percentile(disc_pred[valid], p):.6f} m/px")
    
    # Also on GT depth for reference
    print(f"\n=== Discontinuity Statistics (GT Depth) ===")
    print(f"Valid pixels: {valid.sum()}")
    print(f"Min: {disc_gt[valid].min():.6f} m/px")
    print(f"Max: {disc_gt[valid].max():.6f} m/px")
    print(f"Mean: {disc_gt[valid].mean():.6f} m/px")
    print(f"Median: {np.median(disc_gt[valid]):.6f} m/px")
    print(f"Std: {disc_gt[valid].std():.6f} m/px")
    
    # Save discontinuity array
    np.save("results/hailostereo_output_audit/discontinuity.npy", disc_pred.astype(np.float32))
    print("\nSaved discontinuity.npy")
    
    # Save stats
    stats = {
        "input_field": "metric_depth_m",
        "output_shape": list(disc_pred.shape),
        "output_dtype": "float32",
        "formula": "sqrt((dZ/dx)^2 + (dZ/dy)^2) with central differences",
        "horizontal_diff": "central: (Z[:,2:] - Z[:,:-2])/2, forward/backward at borders",
        "vertical_diff": "central: (Z[2:,:] - Z[:-2,:])/2, forward/backward at borders",
        "gradient_type": "magnitude",
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(disc_pred.size),
        "min": float(disc_pred[valid].min()),
        "max": float(disc_pred[valid].max()),
        "mean": float(disc_pred[valid].mean()),
        "median": float(np.median(disc_pred[valid])),
        "std": float(disc_pred[valid].std()),
        "percentiles": {f"P{p}": float(np.percentile(disc_pred[valid], p)) for p in [1, 5, 25, 50, 75, 95, 99]},
        "gt_stats": {
            "mean": float(disc_gt[valid].mean()),
            "median": float(np.median(disc_gt[valid])),
            "std": float(disc_gt[valid].std())
        }
    }
    
    with open("results/hailostereo_output_audit/discontinuity_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved discontinuity_stats.json")


if __name__ == "__main__":
    main()