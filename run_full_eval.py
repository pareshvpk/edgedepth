"""
Run full validation set evaluation.
"""
import pathlib
import sys
import json

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
from src.data import Kitti2015, MEAN, STD

ONNX_PATH = pathlib.Path(__file__).resolve().parent / "artifacts" / "hailo_stereo.onnx"

KITTI_FOCAL = 715.7
KITTI_BASELINE = 0.54
MAX_DISP = 192
fb = KITTI_FOCAL * KITTI_BASELINE


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
    img_arr = ((img_arr - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    return img_arr


def depth_from_disp(disp: np.ndarray) -> np.ndarray:
    return fb / np.maximum(disp, 1e-3)


def main():
    # Load ONNX model
    print(f"Loading ONNX model from {ONNX_PATH}")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    
    # Load validation dataset
    root = pathlib.Path("data/kitti2015/training")
    ds = Kitti2015(root, split="val")
    
    print(f"Found {len(ds)} validation samples")
    
    # Accumulators
    protocols = ("masked", "official")
    
    # Disparity metrics
    disp_acc = {k: [0.0, 0.0, 0] for k in protocols}  # [sum_epe, sum_d1, count]
    
    # Depth metrics per band
    BANDS = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 80.0))
    DELTA = 1.25
    EPS_DISP = 1e-3
    
    class Band:
        def __init__(self):
            self.abs_err = []
            self.rel_err = []
            self.disp_err = []
            self.disp_bias = []
            self.z_bias = []
            self.z_gt = []
            self.delta_ok = 0
            self.n = 0
        
        def add(self, z_pred, z_gt, disp_err):
            if z_gt.size == 0:
                return
            d = np.abs(z_pred - z_gt)
            self.abs_err.append(d.astype(np.float32))
            self.rel_err.append((d / z_gt).astype(np.float32))
            self.disp_err.append(np.abs(disp_err).astype(np.float32))
            self.disp_bias.append(disp_err.astype(np.float32))
            self.z_bias.append((z_pred - z_gt).astype(np.float32))
            self.z_gt.append(z_gt.astype(np.float32))
            ratio = np.maximum(z_pred / z_gt, z_gt / np.maximum(z_pred, EPS_DISP))
            self.delta_ok += int((ratio < DELTA).sum())
            self.n += int(z_gt.size)
        
        def summary(self):
            if self.n == 0:
                return None
            abs_err = np.concatenate(self.abs_err)
            rel_err = np.concatenate(self.rel_err)
            disp_err = np.concatenate(self.disp_err)
            disp_bias = np.concatenate(self.disp_bias)
            z_bias = np.concatenate(self.z_bias)
            z_gt = np.concatenate(self.z_gt)
            z_mean = float(z_gt.mean())
            epe = float(disp_err.mean())
            return {
                "n": self.n,
                "mae": float(abs_err.mean()),
                "rmse": float(np.sqrt((abs_err.astype(np.float64) ** 2).mean())),
                "median": float(np.median(abs_err)),
                "absrel": 100.0 * float(rel_err.mean()),
                "delta": 100.0 * self.delta_ok / self.n,
                "epe": epe,
                "floor": z_mean * z_mean / fb * epe,
                "z_mean": z_mean,
                "dbias": float(disp_bias.mean()),
                "zbias": float(z_bias.mean()),
            }
    
    depth_bands = {k: {b: Band() for b in BANDS} for k in protocols}
    depth_overall = {k: Band() for k in protocols}
    
    # Per-scene results
    scene_results = []
    
    for i in range(len(ds)):
        s = ds[i]
        left = s["left"][None].numpy()
        right = s["right"][None].numpy()
        disp_gt = s["disp"].numpy()[0]
        mask = s["mask"].numpy()[0] > 0.5
        official = disp_gt > 0.0
        
        # Run inference
        outputs = sess.run(output_names, {"left": left, "right": right})
        pred = outputs[0][0, 0]
        
        # Disparity metrics
        valid = {"masked": mask, "official": official}
        scene_disp = {}
        for k in protocols:
            v = valid[k]
            if v.any():
                err = np.abs(pred[v] - disp_gt[v])
                d1 = (err > 3.0) & (err > 0.05 * np.abs(disp_gt[v]))
                epe = float(err.mean())
                d1_pct = 100.0 * float(d1.mean())
                n = int(v.sum())
                disp_acc[k][0] += epe * n
                disp_acc[k][1] += d1_pct * n
                disp_acc[k][2] += n
                scene_disp[k] = {"epe": epe, "d1": d1_pct, "pixels": n}
            else:
                scene_disp[k] = {"epe": float("nan"), "d1": float("nan"), "pixels": 0}
        
        # Depth metrics
        z_pred_full = depth_from_disp(pred)
        z_gt_full = depth_from_disp(disp_gt)
        derr_full = pred - disp_gt
        
        scene_depth = {}
        for k in protocols:
            v = valid[k]
            v_depth = v & (disp_gt > fb / 80.0)
            if v_depth.any():
                z_pred = z_pred_full[v_depth]
                z_gt = z_gt_full[v_depth]
                derr = derr_full[v_depth]
                depth_overall[k].add(z_pred, z_gt, derr)
                for lo, hi in BANDS:
                    sel = (z_gt >= lo) & (z_gt < hi)
                    depth_bands[k][(lo, hi)].add(z_pred[sel], z_gt[sel], derr[sel])
                scene_depth[k] = {"pixels": int(v_depth.sum())}
            else:
                scene_depth[k] = {"pixels": 0}
        
        scene_results.append({
            "scene": ds.samples[i][0].name,
            "disparity": scene_disp,
            "depth": scene_depth
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(ds)}")
    
    # Print aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE VALIDATION RESULTS (40 scenes)")
    print("=" * 60)
    
    print(f"\n{'protocol':10s} {'pixels':>12s} {'EPE':>9s} {'D1-all':>9s}")
    for k in protocols:
        e, d, n = disp_acc[k]
        if n > 0:
            print(f"{k:10s} {n:12,d} {e/n:8.3f}p {d/n:8.2f}%")
    
    print("\n--- Depth Metrics (Official Protocol) ---")
    print(f"{'range':>12s} {'pixels':>11s} {'MAE':>9s} {'RMSE':>8s} {'median':>8s} {'AbsRel':>8s} {'d<1.25':>8s} {'EPE':>8s} {'floor':>9s} {'bias dd':>9s} {'bias dZ':>9s}")
    for lo, hi in BANDS:
        b = depth_bands["official"][(lo, hi)].summary()
        if b:
            print(f"{lo:.0f}-{hi:.0f} m: {b['n']:11,d} {b['mae']:8.2f}m {b['rmse']:7.2f}m {b['median']:7.2f}m {b['absrel']:7.2f}% {b['delta']:7.2f}% {b['epe']:7.3f}p {b['floor']:8.2f}m {b['dbias']:+8.2f}p {b['zbias']:+8.2f}m")
    
    overall = depth_overall["official"].summary()
    if overall:
        print(f"{'all <= 80 m':>12s} {overall['n']:11,d} {overall['mae']:8.2f}m {overall['rmse']:7.2f}m {overall['median']:7.2f}m {overall['absrel']:7.2f}% {overall['delta']:7.2f}% {overall['epe']:7.3f}p {overall['floor']:8.2f}m {overall['dbias']:+8.2f}p {overall['zbias']:+8.2f}m")
    
    print("\n--- Depth Metrics (Masked Protocol) ---")
    print(f"{'range':>12s} {'pixels':>11s} {'MAE':>9s} {'RMSE':>8s} {'median':>8s} {'AbsRel':>8s} {'d<1.25':>8s} {'EPE':>8s} {'floor':>9s} {'bias dd':>9s} {'bias dZ':>9s}")
    for lo, hi in BANDS:
        b = depth_bands["masked"][(lo, hi)].summary()
        if b:
            print(f"{lo:.0f}-{hi:.0f} m: {b['n']:11,d} {b['mae']:8.2f}m {b['rmse']:7.2f}m {b['median']:7.2f}m {b['absrel']:7.2f}% {b['delta']:7.2f}% {b['epe']:7.3f}p {b['floor']:8.2f}m {b['dbias']:+8.2f}p {b['zbias']:+8.2f}m")
    
    overall = depth_overall["masked"].summary()
    if overall:
        print(f"{'all <= 80 m':>12s} {overall['n']:11,d} {overall['mae']:8.2f}m {overall['rmse']:7.2f}m {overall['median']:7.2f}m {overall['absrel']:7.2f}% {overall['delta']:7.2f}% {overall['epe']:7.3f}p {overall['floor']:8.2f}m {overall['dbias']:+8.2f}p {overall['zbias']:+8.2f}m")
    
    # Save scene metrics
    with open("results/hailostereo_output_audit/scene_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "protocol", "disparity_epe_px", "disparity_d1_pct", "disparity_pixels", "depth_pixels"])
        for sr in scene_results:
            for k in protocols:
                d = sr["disparity"][k]
                dp = sr["depth"][k]
                writer.writerow([sr["scene"], k, d["epe"], d["d1"], d["pixels"], dp["pixels"]])
    
    # Save aggregate metrics
    agg = {
        "n_scenes": len(ds),
        "disparity": {k: {"epe": disp_acc[k][0]/disp_acc[k][2], "d1": disp_acc[k][1]/disp_acc[k][2], "pixels": disp_acc[k][2]} for k in protocols if disp_acc[k][2] > 0},
        "depth_bands": {}
    }
    for k in protocols:
        agg["depth_bands"][k] = {}
        for lo, hi in BANDS:
            b = depth_bands[k][(lo, hi)].summary()
            if b:
                agg["depth_bands"][k][f"{lo}-{hi}"] = b
        agg["depth_bands"][k]["all"] = depth_overall[k].summary()
    
    with open("results/hailostereo_output_audit/aggregate_metrics.json", "w") as f:
        json.dump(agg, f, indent=2)
    
    print("\nSaved scene_metrics.csv and aggregate_metrics.json")


if __name__ == "__main__":
    import csv
    main()