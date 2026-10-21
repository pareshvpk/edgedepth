import sys
sys.path.insert(0, r'C:\Microchip\hailo-stereo\src')
import torch
from model import HailoStereo, build_model, MAX_DISP
from data import Kitti2015, MEAN, STD, _valid_mask
from pointcloud import Camera, disparity_to_depth, depth_to_points, build_cloud, prepare
import numpy as np
import json
from PIL import Image

# Load model
blob = torch.load(r'C:\Microchip\hailo-stereo\runs\kitti_mixed_fixed768\best.pt', map_location='cpu', weights_only=False)
model = build_model(blob)
model.load_state_dict(blob.get('model', blob), strict=True)
model.eval()

# Load validation dataset
ds = Kitti2015(r'C:\Microchip\hailo-stereo\data\kitti2015\training', split='val')
print(f"Dataset size: {len(ds)}")

KITTI_FOCAL = 715.7
KITTI_BASELINE = 0.54
fb = KITTI_FOCAL * KITTI_BASELINE

# Accumulators for all scenes
all_scenes = []
scene_metrics = []

for i in range(len(ds)):
    s = ds[i]
    lp, rp, dp = ds.samples[i]
    
    with torch.no_grad():
        pred = model(s['left'][None], s['right'][None])[0, 0].cpu().numpy()
    
    gt = s["disp"].numpy()[0]
    mask = s["mask"].numpy()[0] > 0.5
    official_mask = gt > 0
    
    # Disparity metrics
    masked_pred = pred[mask]
    masked_gt = gt[mask]
    official_pred = pred[official_mask]
    official_gt = gt[official_mask]
    
    masked_epe = np.abs(masked_pred - masked_gt).mean()
    masked_d1 = ((np.abs(masked_pred - masked_gt) > 3.0) & (np.abs(masked_pred - masked_gt) > 0.05 * np.abs(masked_gt))).mean() * 100
    official_epe = np.abs(official_pred - official_gt).mean()
    official_d1 = ((np.abs(official_pred - official_gt) > 3.0) & (np.abs(official_pred - official_gt) > 0.05 * np.abs(official_gt))).mean() * 100
    
    # Depth metrics
    pred_depth = fb / np.maximum(pred, 1e-3)
    gt_depth = fb / np.maximum(gt, 1e-3)
    valid_depth = (gt > 0) & (gt < MAX_DISP) & np.isfinite(gt) & (gt >= fb / 80.0)
    
    # Per-band depth metrics
    bands = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80)]
    band_metrics = {}
    for lo, hi in bands:
        band_mask = (gt_depth >= lo) & (gt_depth < hi) & valid_depth
        if band_mask.sum() > 0:
            z_pred = pred_depth[band_mask]
            z_gt = gt_depth[band_mask]
            abs_err = np.abs(z_pred - z_gt)
            rel_err = abs_err / z_gt
            rmse = np.sqrt((abs_err ** 2).mean())
            delta = (np.maximum(z_pred / z_gt, z_gt / np.maximum(z_pred, 1e-3)) < 1.25).mean() * 100
            band_metrics[f"{lo}-{hi}"] = {
                "n": int(band_mask.sum()),
                "mae": float(abs_err.mean()),
                "rmse": float(rmse),
                "median": float(np.median(abs_err)),
                "absrel": float(rel_err.mean() * 100),
                "delta125": float(delta)
            }
    
    scene_metrics.append({
        "scene": lp.name,
        "masked_epe": float(masked_epe),
        "masked_d1": float(masked_d1),
        "official_epe": float(official_epe),
        "official_d1": float(official_d1),
        "masked_pixels": int(mask.sum()),
        "official_pixels": int(official_mask.sum()),
        "valid_depth_pixels": int(valid_depth.sum()),
        "band_metrics": band_metrics
    })
    
    if i % 10 == 0:
        print(f"  Scene {i}: {lp.name}, official EPE={official_epe:.4f}, D1={official_d1:.2f}%")

# Aggregate metrics
print("\n=== AGGREGATE METRICS (40 validation scenes) ===")

# Pixel-weighted
total_masked_pixels = sum(s["masked_pixels"] for s in scene_metrics)
total_official_pixels = sum(s["official_pixels"] for s in scene_metrics)
weighted_masked_epe = sum(s["masked_epe"] * s["masked_pixels"] for s in scene_metrics) / total_masked_pixels
weighted_masked_d1 = sum(s["masked_d1"] * s["masked_pixels"] for s in scene_metrics) / total_masked_pixels
weighted_official_epe = sum(s["official_epe"] * s["official_pixels"] for s in scene_metrics) / total_official_pixels
weighted_official_d1 = sum(s["official_d1"] * s["official_pixels"] for s in scene_metrics) / total_official_pixels

print(f"Total masked pixels: {total_masked_pixels:,}")
print(f"Total official pixels: {total_official_pixels:,}")
print(f"Weighted masked EPE: {weighted_masked_epe:.4f} px")
print(f"Weighted masked D1: {weighted_masked_d1:.2f}%")
print(f"Weighted official EPE: {weighted_official_epe:.4f} px")
print(f"Weighted official D1: {weighted_official_d1:.2f}%")

# Scene-averaged
print(f"\nScene-averaged masked EPE: {np.mean([s['masked_epe'] for s in scene_metrics]):.4f} px")
print(f"Scene-averaged masked D1: {np.mean([s['masked_d1'] for s in scene_metrics]):.2f}%")
print(f"Scene-averaged official EPE: {np.mean([s['official_epe'] for s in scene_metrics]):.4f} px")
print(f"Scene-averaged official D1: {np.mean([s['official_d1'] for s in scene_metrics]):.2f}%")

# Per-band aggregated
print("\n=== AGGREGATE DEPTH BAND METRICS ===")
for band_name in ["0-5", "5-10", "10-20", "20-40", "40-80"]:
    all_n = sum(s["band_metrics"].get(band_name, {}).get("n", 0) for s in scene_metrics)
    if all_n == 0:
        print(f"  {band_name} m: no valid pixels")
        continue
    weighted_mae = sum(s["band_metrics"].get(band_name, {}).get("mae", 0) * s["band_metrics"].get(band_name, {}).get("n", 0) for s in scene_metrics) / all_n
    weighted_rmse = np.sqrt(sum(s["band_metrics"].get(band_name, {}).get("rmse", 0)**2 * s["band_metrics"].get(band_name, {}).get("n", 0) for s in scene_metrics) / all_n)
    weighted_median = np.median([s["band_metrics"].get(band_name, {}).get("median", 0) for s in scene_metrics if s["band_metrics"].get(band_name, {}).get("n", 0) > 0])
    weighted_absrel = sum(s["band_metrics"].get(band_name, {}).get("absrel", 0) * s["band_metrics"].get(band_name, {}).get("n", 0) for s in scene_metrics) / all_n
    weighted_delta = sum(s["band_metrics"].get(band_name, {}).get("delta125", 0) * s["band_metrics"].get(band_name, {}).get("n", 0) for s in scene_metrics) / all_n
    print(f"  {band_name} m: n={all_n:,}, MAE={weighted_mae:.4f}m, RMSE={weighted_rmse:.4f}m, Median={weighted_median:.4f}m, AbsRel={weighted_absrel:.2f}%, delta<1.25={weighted_delta:.2f}%")

# Save scene metrics
with open(r'C:\Microchip\hailo-stereo\scene_metrics.json', 'w') as f:
    json.dump(scene_metrics, f, indent=2)

# Also save as CSV
import csv
with open(r'C:\Microchip\hailo-stereo\scene_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["scene", "masked_epe", "masked_d1", "official_epe", "official_d1", 
                     "masked_pixels", "official_pixels", "valid_depth_pixels"])
    for s in scene_metrics:
        writer.writerow([s["scene"], s["masked_epe"], s["masked_d1"], s["official_epe"], s["official_d1"],
                         s["masked_pixels"], s["official_pixels"], s["valid_depth_pixels"]])

print("\nSaved scene_metrics.json and scene_metrics.csv")