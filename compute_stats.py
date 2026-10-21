import sys
sys.path.insert(0, r'C:\Microchip\hailo-stereo\src')
import torch
from model import HailoStereo, build_model, MAX_DISP
from data import Kitti2015, MEAN, STD, _valid_mask, _loss_weight
from pointcloud import Camera, disparity_to_depth, depth_to_points, build_cloud, prepare
import numpy as np
import json
from PIL import Image

# Load model
blob = torch.load(r'C:\Microchip\hailo-stereo\runs\kitti_mixed_fixed768\best.pt', map_location='cpu', weights_only=False)
model = build_model(blob)
model.load_state_dict(blob.get('model', blob), strict=True)
model.eval()

# Load a sample (first validation sample)
ds = Kitti2015(r'C:\Microchip\hailo-stereo\data\kitti2015\training', split='val')
s = ds[0]

# Run inference
with torch.no_grad():
    pred = model(s['left'][None], s['right'][None])[0, 0].cpu().numpy()

gt = s["disp"].numpy()[0]
mask = s["mask"].numpy()[0] > 0.5
official_mask = gt > 0

# Get original images
lp, rp, dp = ds.samples[0]
orig_left = Image.open(lp)
orig_right = Image.open(rp)

# Crop info (from data.py _finish method)
h, w = gt.shape
nh, nw = (h // 16) * 16, (w // 16) * 16
crop_y, crop_x = h - nh, w - nw

print("=== INPUT METADATA ===")
print(f"Scene: {lp.name}")
print(f"Original size: {orig_left.size}")
print(f"Crop coordinates: x={crop_x}, y={crop_y}, w={nw}, h={nh}")
print(f"Cropped size: {nw}x{nh}")
print(f"Model input size: 1232x368")
print(f"Channels: 3 (RGB)")
print(f"Normalization: ImageNet MEAN={MEAN}, STD={STD}")
print(f"Disparity range: [0, {MAX_DISP}] pixels")
print(f"Number of disparity hypotheses: {24}")

# Disparity statistics
print("\n=== DISPARITY STATISTICS (pred) ===")
print(f"Shape: {pred.shape}")
print(f"Dtype: {pred.dtype}")
print(f"Min: {pred.min():.4f}")
print(f"Max: {pred.max():.4f}")
print(f"Mean: {pred.mean():.4f}")
print(f"Median: {np.median(pred):.4f}")
print(f"Std: {pred.std():.4f}")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"P{p}: {np.percentile(pred, p):.4f}")

# Invalid values
nan_count = np.isnan(pred).sum()
inf_count = np.isinf(pred).sum()
zero_count = (pred == 0).sum()
neg_count = (pred < 0).sum()
print(f"NaN count: {nan_count} ({100*nan_count/pred.size:.2f}%)")
print(f"Inf count: {inf_count} ({100*inf_count/pred.size:.2f}%)")
print(f"Zero count: {zero_count} ({100*zero_count/pred.size:.2f}%)")
print(f"Negative count: {neg_count} ({100*neg_count/pred.size:.2f}%)")

# GT statistics
print("\n=== GT DISPARITY STATISTICS ===")
print(f"Shape: {gt.shape}")
print(f"Min: {gt.min():.4f}")
print(f"Max: {gt.max():.4f}")
print(f"Mean: {gt.mean():.4f}")
print(f"Median: {np.median(gt):.4f}")

# EPE and D1 metrics
masked_pred = pred[mask]
masked_gt = gt[mask]
official_pred = pred[official_mask]
official_gt = gt[official_mask]

def compute_metrics(pred, gt, name):
    err = np.abs(pred - gt)
    epe = err.mean()
    d1 = ((err > 3.0) & (err > 0.05 * np.abs(gt))).mean() * 100
    print(f"\n{name} protocol:")
    print(f"  Valid pixels: {len(err)}")
    print(f"  EPE: {epe:.4f} px")
    print(f"  D1-all: {d1:.2f}%")
    return epe, d1

compute_metrics(masked_pred, masked_gt, "MASKED")
compute_metrics(official_pred, official_gt, "OFFICIAL")

# Disparity error percentiles
print("\n=== DISPARITY ERROR PERCENTILES (official) ===")
err = np.abs(official_pred - official_gt)
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"P{p}: {np.percentile(err, p):.4f} px")

# Depth conversion
KITTI_FOCAL = 715.7
KITTI_BASELINE = 0.54
fb = KITTI_FOCAL * KITTI_BASELINE

# Camera calibration for cropped input
orig_h, orig_w = orig_left.size[1], orig_left.size[0]
cam = Camera(f=KITTI_FOCAL, cx=609.56, cy=172.85, baseline=KITTI_BASELINE)
cam_cropped, how = cam.for_input(orig_h, orig_w)
print(f"\n=== CAMERA CALIBRATION ===")
print(f"Original: f={KITTI_FOCAL} px, cx=609.56, cy=172.85, B={KITTI_BASELINE} m")
print(f"Cropped: f={cam_cropped.f:.2f} px, cx={cam_cropped.cx:.2f}, cy={cam_cropped.cy:.2f}, B={cam_cropped.baseline:.3f} m")
print(f"Crop method: {how}")
print(f"f*B = {fb:.2f} px*m")

# Depth from disparity
EPS_DISP = 1e-3
pred_depth = fb / np.maximum(pred, EPS_DISP)
gt_depth = fb / np.maximum(gt, EPS_DISP)

# Valid depth pixels (disparity > 0 and in range)
valid_depth_mask = (gt > 0) & (gt < MAX_DISP) & np.isfinite(gt) & (gt >= fb / 80.0)

print("\n=== DEPTH STATISTICS (pred) ===")
print(f"Shape: {pred_depth.shape}")
print(f"Min: {pred_depth.min():.4f} m")
print(f"Max: {pred_depth.max():.4f} m")
print(f"Mean: {pred_depth.mean():.4f} m")
print(f"Median: {np.median(pred_depth):.4f} m")
print(f"Std: {pred_depth.std():.4f} m")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"P{p}: {np.percentile(pred_depth, p):.4f} m")

# Valid/invalid pixels
valid_pred_depth = pred_depth[valid_depth_mask]
print(f"\nValid depth pixels (GT in range, <= 80m): {valid_depth_mask.sum()} ({100*valid_depth_mask.sum()/pred_depth.size:.2f}%)")
print(f"Invalid depth pixels: {(~valid_depth_mask).sum()} ({100*(~valid_depth_mask).sum()/pred_depth.size:.2f}%)")

# Depth bands
bands = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, np.inf)]
print("\n=== DEPTH BAND COUNTS (GT) ===")
for lo, hi in bands:
    if hi == np.inf:
        mask_band = (gt_depth >= lo)
        label = f">{lo} m"
    else:
        mask_band = (gt_depth >= lo) & (gt_depth < hi)
        label = f"{lo}-{hi} m"
    count = (mask_band & valid_depth_mask).sum()
    print(f"  {label}: {count} ({100*count/valid_depth_mask.sum():.2f}% of valid)")

# Depth error metrics per band
print("\n=== DEPTH ERROR METRICS PER BAND (official protocol, GT depth <= 80m) ===")
for lo, hi in bands[:-1]:  # exclude >80m
    mask_band = (gt_depth >= lo) & (gt_depth < hi) & valid_depth_mask
    if mask_band.sum() == 0:
        print(f"  {lo}-{hi} m: no valid pixels")
        continue
    z_pred = pred_depth[mask_band]
    z_gt = gt_depth[mask_band]
    abs_err = np.abs(z_pred - z_gt)
    rel_err = abs_err / z_gt
    rmse = np.sqrt((abs_err ** 2).mean())
    delta = (np.maximum(z_pred / z_gt, z_gt / np.maximum(z_pred, EPS_DISP)) < 1.25).mean() * 100
    print(f"  {lo}-{hi} m: n={mask_band.sum():,}, MAE={abs_err.mean():.4f}m, RMSE={rmse:.4f}m, Median={np.median(abs_err):.4f}m, AbsRel={rel_err.mean()*100:.2f}%, delta<1.25={delta:.2f}%")

# Point cloud
print("\n=== POINT CLOUD ===")
left_u8 = prepare(Image.open(lp))
xyz, rgb, stats = build_cloud(pred, left_u8, cam_cropped, max_depth=80.0, drop_border=True)
print(f"Original pixels: {stats['pixels']:,}")
print(f"Valid points: {stats['points']:,} ({100*stats['points']/stats['pixels']:.2f}%)")
print(f"Beyond range: {stats['beyond_range']:,} ({100*stats['beyond_range']/stats['pixels']:.2f}%)")
print(f"Border dropped: {stats.get('border_dropped', 0):,}")
if len(xyz) > 0:
    print(f"X range: {xyz[:,0].min():.2f} .. {xyz[:,0].max():.2f} m")
    print(f"Y range: {xyz[:,1].min():.2f} .. {xyz[:,1].max():.2f} m")
    print(f"Z range: {xyz[:,2].min():.2f} .. {xyz[:,2].max():.2f} m")
    print(f"X mean: {xyz[:,0].mean():.2f} m, std: {xyz[:,0].std():.2f} m")
    print(f"Y mean: {xyz[:,1].mean():.2f} m, std: {xyz[:,1].std():.2f} m")
    print(f"Z mean: {xyz[:,2].mean():.2f} m, std: {xyz[:,2].std():.2f} m")

# Save arrays
np.save(r'C:\Microchip\hailo-stereo\pred_disp.npy', pred.astype(np.float32))
np.save(r'C:\Microchip\hailo-stereo\gt_disp.npy', gt.astype(np.float32))
np.save(r'C:\Microchip\hailo-stereo\pred_depth.npy', pred_depth.astype(np.float32))
np.save(r'C:\Microchip\hailo-stereo\gt_depth.npy', gt_depth.astype(np.float32))
np.save(r'C:\Microchip\hailo-stereo\xyz.npy', xyz.astype(np.float32))
np.save(r'C:\Microchip\hailo-stereo\rgb.npy', rgb.astype(np.uint8))

# Save metadata
metadata = {
    "scene": lp.name,
    "original_size": [orig_w, orig_h],
    "crop": {"x": int(crop_x), "y": int(crop_y), "w": int(nw), "h": int(nh)},
    "model_input_size": [1232, 368],
    "channels": 3,
    "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
    "disparity_range": [0, int(MAX_DISP)],
    "num_disparity_hypotheses": 24,
    "camera": {
        "focal": KITTI_FOCAL,
        "baseline": KITTI_BASELINE,
        "cx_full": 609.56,
        "cy_full": 172.85,
        "focal_cropped": cam_cropped.f,
        "cx_cropped": cam_cropped.cx,
        "cy_cropped": cam_cropped.cy,
        "fb": fb
    },
    "disparity_stats": {
        "pred": {"min": float(pred.min()), "max": float(pred.max()), "mean": float(pred.mean()),
                 "median": float(np.median(pred)), "std": float(pred.std())},
        "gt": {"min": float(gt.min()), "max": float(gt.max()), "mean": float(gt.mean()),
               "median": float(np.median(gt))},
        "epe_masked": float(np.abs(masked_pred - masked_gt).mean()),
        "epe_official": float(np.abs(official_pred - official_gt).mean()),
        "d1_masked": float(((np.abs(masked_pred - masked_gt) > 3.0) & (np.abs(masked_pred - masked_gt) > 0.05 * np.abs(masked_gt))).mean() * 100),
        "d1_official": float(((np.abs(official_pred - official_gt) > 3.0) & (np.abs(official_pred - official_gt) > 0.05 * np.abs(official_gt))).mean() * 100),
        "valid_pixels_masked": int(mask.sum()),
        "valid_pixels_official": int(official_mask.sum())
    },
    "depth_stats": {
        "pred": {"min": float(pred_depth.min()), "max": float(pred_depth.max()), "mean": float(pred_depth.mean()),
                 "median": float(np.median(pred_depth)), "std": float(pred_depth.std())},
        "gt": {"min": float(gt_depth.min()), "max": float(gt_depth.max()), "mean": float(gt_depth.mean()),
               "median": float(np.median(gt_depth))},
        "valid_pixels": int(valid_depth_mask.sum()),
        "invalid_pixels": int((~valid_depth_mask).sum())
    },
    "pointcloud_stats": stats
}

with open(r'C:\Microchip\hailo-stereo\metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n=== SAVED FILES ===")
print("pred_disp.npy, gt_disp.npy, pred_depth.npy, gt_depth.npy, xyz.npy, rgb.npy, metadata.json")