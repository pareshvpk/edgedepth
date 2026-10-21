"""
Run inference using the ONNX model with onnxruntime.
"""
import argparse
import pathlib
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
from data import MEAN, STD

ONNX_PATH = pathlib.Path(__file__).resolve().parent / "artifacts" / "hailo_stereo.onnx"
CKPT_PATH = pathlib.Path(__file__).resolve().parent / "runs" / "kitti_mixed_fixed768" / "best.pt"


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
    lefts = sorted((root / "image_2").glob("*_10.png"))
    val_samples = lefts[-40:]  # Last 40 are validation
    
    print(f"Found {len(val_samples)} validation samples")
    
    # Load ONNX model
    print(f"Loading ONNX model from {ONNX_PATH}")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f"Inputs: {input_names}")
    print(f"Outputs: {output_names}")
    
    # Process first validation sample
    lp = val_samples[0]
    rp = root / "image_3" / lp.name
    dp = root / "disp_occ_0" / lp.name
    
    print(f"\nProcessing {lp.name}")
    
    left = load_and_preprocess(lp)
    right = load_and_preprocess(rp)
    
    # Load ground truth disparity
    disp_gt = np.array(Image.open(dp), dtype=np.float32) / 256.0
    
    # Crop ground truth same way
    h, w = disp_gt.shape
    nh, nw = (h // 16) * 16, (w // 16) * 16
    disp_gt = disp_gt[h - nh:, w - nw:]
    
    print(f"Input shapes: left={left.shape}, right={right.shape}")
    print(f"GT shape: {disp_gt.shape}")
    
    # Run inference
    outputs = sess.run(output_names, {"left": left, "right": right})
    pred = outputs[0][0, 0]  # Remove batch and channel dims
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"Prediction mean: {pred.mean():.4f}")
    print(f"Prediction std: {pred.std():.4f}")
    
    # Save raw prediction
    np.save("results/hailostereo_output_audit/disparity.npy", pred.astype(np.float32))
    print("Saved disparity.npy")
    
    # Compute statistics
    valid_mask = (disp_gt > 0.0) & (disp_gt < 192.0) & np.isfinite(disp_gt)
    valid_mask[:, :192] = False  # Masked protocol
    
    official_mask = disp_gt > 0.0
    
    print(f"\nValid pixels (masked): {valid_mask.sum()}")
    print(f"Valid pixels (official): {official_mask.sum()}")
    print(f"Total pixels: {pred.size}")
    
    # Statistics on full prediction
    print(f"\n=== Disparity Statistics (full prediction) ===")
    print(f"Min: {pred.min():.4f}")
    print(f"Max: {pred.max():.4f}")
    print(f"Mean: {pred.mean():.4f}")
    print(f"Median: {np.median(pred):.4f}")
    print(f"Std: {pred.std():.4f}")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"P{p}: {np.percentile(pred, p):.4f}")
    
    nan_count = np.isnan(pred).sum()
    inf_count = np.isinf(pred).sum()
    zero_count = (pred == 0).sum()
    neg_count = (pred < 0).sum()
    print(f"NaN: {nan_count} ({100*nan_count/pred.size:.2f}%)")
    print(f"Inf: {inf_count} ({100*inf_count/pred.size:.2f}%)")
    print(f"Zero: {zero_count} ({100*zero_count/pred.size:.2f}%)")
    print(f"Negative: {neg_count} ({100*neg_count/pred.size:.2f}%)")
    
    # Error metrics
    for name, mask in [("masked", valid_mask), ("official", official_mask)]:
        if mask.any():
            err = np.abs(pred[mask] - disp_gt[mask])
            d1 = (err > 3.0) & (err > 0.05 * np.abs(disp_gt[mask]))
            print(f"\n=== {name.upper()} Protocol ===")
            print(f"Pixels: {mask.sum()}")
            print(f"EPE: {err.mean():.4f} px")
            print(f"D1-all: {100.0 * d1.mean():.4f}%")
            
            # Error percentiles
            for p in [1, 5, 25, 50, 75, 95, 99]:
                print(f"Error P{p}: {np.percentile(err, p):.4f} px")
    
    # Save stats
    stats = {
        "scene": lp.name,
        "shape": list(pred.shape),
        "dtype": "float32",
        "full_stats": {
            "min": float(pred.min()),
            "max": float(pred.max()),
            "mean": float(pred.mean()),
            "median": float(np.median(pred)),
            "std": float(pred.std()),
            "percentiles": {f"P{p}": float(np.percentile(pred, p)) for p in [1, 5, 25, 50, 75, 95, 99]},
            "nan_count": int(nan_count),
            "inf_count": int(inf_count),
            "zero_count": int(zero_count),
            "neg_count": int(neg_count),
            "total_pixels": int(pred.size),
        },
        "masked_protocol": {},
        "official_protocol": {}
    }
    
    for name, mask in [("masked", valid_mask), ("official", official_mask)]:
        if mask.any():
            err = np.abs(pred[mask] - disp_gt[mask])
            d1 = (err > 3.0) & (err > 0.05 * np.abs(disp_gt[mask]))
            stats[f"{name}_protocol"] = {
                "valid_pixels": int(mask.sum()),
                "epe_px": float(err.mean()),
                "d1_percent": float(100.0 * d1.mean()),
                "error_percentiles": {f"P{p}": float(np.percentile(err, p)) for p in [1, 5, 25, 50, 75, 95, 99]}
            }
    
    import json
    with open("results/hailostereo_output_audit/disparity_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved disparity_stats.json")


if __name__ == "__main__":
    main()