"""
Generate point cloud from disparity prediction.
"""
import pathlib
import sys
import json

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

ONNX_PATH = pathlib.Path(__file__).resolve().parent / "artifacts" / "hailo_stereo.onnx"

# Nominal KITTI camera parameters
HEIGHT, WIDTH = 368, 1232
MAX_DISP = 192
KITTI_FOCAL = 715.7            # px
KITTI_BASELINE = 0.54          # m
KITTI_CX = 609.56              # px, full frame
KITTI_CY = 172.85              # px, full frame
MIN_DISP = 0.5


def load_and_preprocess(img_path):
    """Load image and preprocess to match the model's expected input."""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.width, img.height
    
    # Crop to 368x1232 bottom-right as the model expects
    if img.height >= HEIGHT and img.width >= WIDTH:
        img = img.crop((img.width - WIDTH, img.height - HEIGHT, img.width, img.height))
        dx, dy = orig_w - WIDTH, orig_h - HEIGHT
    else:
        # Resize to 1232 wide then crop
        s = WIDTH / img.width
        new_h = max(HEIGHT, round(img.height * s))
        img = img.resize((WIDTH, new_h), Image.BILINEAR)
        img = img.crop((0, img.height - HEIGHT, WIDTH, img.height))
        dx, dy = 0, new_h - HEIGHT  # Not exactly right but for calibration adjustment
    
    # For point cloud we need the original uint8 image for coloring
    return img, orig_w, orig_h, dx, dy


def prepare_camera(focal, cx, cy, baseline, orig_h, orig_w, dx, dy):
    """Adjust camera parameters for the cropped/resized input."""
    if orig_h >= HEIGHT and orig_w >= WIDTH:
        # Bottom-right crop
        cx_adj = cx - dx
        cy_adj = cy - dy
        f_adj = focal
    else:
        # Resize then crop
        s = WIDTH / orig_w
        f_adj = focal * s
        cx_adj = cx * s - dx
        cy_adj = cy * s - dy
    
    return f_adj, cx_adj, cy_adj, baseline


def disparity_to_depth(disp: np.ndarray, f: float, baseline: float) -> np.ndarray:
    """Z = f x B / d. Pixels below MIN_DISP come back as +inf."""
    z = np.full(disp.shape, np.inf, dtype=np.float32)
    ok = disp >= MIN_DISP
    z[ok] = f * baseline / disp[ok]
    return z


def depth_to_points(z: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    """Back-project every pixel: [H,W] depth -> [H,W,3] XYZ in metres."""
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (u - cx) * z / f
    y = (v - cy) * z / f
    return np.stack([x, y, z], axis=-1)


def write_ply(path: pathlib.Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Binary little-endian PLY; opens in MeshLab, CloudCompare, Open3D."""
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(rec)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(rec.tobytes())


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
    
    # Load and preprocess images
    left_img, orig_w, orig_h, dx, dy = load_and_preprocess(lp)
    right_img, _, _, _, _ = load_and_preprocess(rp)
    
    # Adjust camera parameters
    f_adj, cx_adj, cy_adj, baseline = prepare_camera(
        KITTI_FOCAL, KITTI_CX, KITTI_CY, KITTI_BASELINE,
        orig_h, orig_w, dx, dy
    )
    
    print(f"Original image: {orig_w}x{orig_h}")
    print(f"Camera (adjusted): f={f_adj:.2f} px, cx={cx_adj:.2f}, cy={cy_adj:.2f}, B={baseline:.3f} m")
    print(f"f x B = {f_adj * baseline:.2f} px.m")
    
    # Preprocess for model (normalized)
    left_arr = np.array(left_img, dtype=np.float32) / 255.0
    from src.data import MEAN, STD
    left_norm = ((left_arr - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    right_arr = np.array(right_img, dtype=np.float32) / 255.0
    right_norm = ((right_arr - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    
    # Run inference
    outputs = sess.run(output_names, {"left": left_norm, "right": right_norm})
    disp = outputs[0][0, 0]
    
    print(f"Disparity range: [{disp.min():.2f}, {disp.max():.2f}] px")
    
    # Convert to depth
    z = disparity_to_depth(disp, f_adj, baseline)
    
    # Build point cloud
    max_depth = 80.0
    drop_border = True
    
    keep = np.isfinite(z) & (z <= max_depth)
    stats = {"pixels": int(disp.size),
             "beyond_range": int((~keep).sum())}
    
    if drop_border:
        border = np.zeros_like(keep)
        border[:, :MAX_DISP] = True
        stats["border_dropped"] = int((keep & border).sum())
        keep &= ~border
    
    xyz = depth_to_points(z, f_adj, cx_adj, cy_adj)[keep]
    rgb = np.array(left_img)[keep]
    stats["points"] = int(keep.sum())
    
    print(f"\nPoint cloud statistics:")
    for k, v in stats.items():
        print(f"  {k:<18} {v:>9,}  ({100 * v / stats['pixels']:.1f}%)")
    
    if len(xyz) > 0:
        lo, hi = xyz.min(0), xyz.max(0)
        print(f"  Extent: X {lo[0]:.1f}..{hi[0]:.1f}  Y {lo[1]:.1f}..{hi[1]:.1f}  Z {lo[2]:.1f}..{hi[2]:.1f} m")
    
    # Save point cloud
    out_path = pathlib.Path("results/hailostereo_output_audit/pointcloud.ply")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out_path, xyz.astype(np.float32), rgb.astype(np.uint8))
    print(f"\nWrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    
    # Save stats
    pc_stats = {
        "camera": {
            "focal_px": f_adj,
            "cx_px": cx_adj,
            "cy_px": cy_adj,
            "baseline_m": baseline,
            "fb_px_m": f_adj * baseline
        },
        "input_image": {
            "original_width": orig_w,
            "original_height": orig_h,
            "cropped_width": WIDTH,
            "cropped_height": HEIGHT
        },
        "disparity": {
            "min_px": float(disp.min()),
            "max_px": float(disp.max()),
            "mean_px": float(disp.mean()),
            "median_px": float(np.median(disp))
        },
        "depth": {
            "min_m": float(z[np.isfinite(z)].min()),
            "max_m": float(z[np.isfinite(z)].max()),
            "mean_m": float(z[np.isfinite(z)].mean()),
            "median_m": float(np.median(z[np.isfinite(z)]))
        },
        "pointcloud": stats,
        "extent": {
            "x_min": float(lo[0]) if len(xyz) > 0 else 0,
            "x_max": float(hi[0]) if len(xyz) > 0 else 0,
            "y_min": float(lo[1]) if len(xyz) > 0 else 0,
            "y_max": float(hi[1]) if len(xyz) > 0 else 0,
            "z_min": float(lo[2]) if len(xyz) > 0 else 0,
            "z_max": float(hi[2]) if len(xyz) > 0 else 0
        } if len(xyz) > 0 else {}
    }
    
    with open("results/hailostereo_output_audit/pointcloud_stats.json", "w") as f:
        json.dump(pc_stats, f, indent=2)
    print("Saved pointcloud_stats.json")


if __name__ == "__main__":
    main()