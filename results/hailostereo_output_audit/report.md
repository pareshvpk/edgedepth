# HailoStereo Output Audit Report

**Date:** 2026-10-20  
**Repository:** `hailo-stereo` (commit `abf48fb`)  
**Checkpoint:** `runs/kitti_mixed_fixed768/best.pt` (epoch 223)  
**Model:** HailoStereo (refine=base)  
**Inference:** ONNX Runtime 1.29.0 on CPU (PyTorch blocked by Windows Application Control)

---

## A. Verified Model and Input Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | HailoStereo (refine=base) | `src/model.py` |
| Checkpoint | `runs/kitti_mixed_fixed768/best.pt` | `runs/kitti_mixed_fixed768/best.json` |
| Epoch | 223 | `best.json` |
| In-loop val EPE | 1.156 px (masked) | `best.json` |
| Parameters | 796,531 | `export_onnx.py` audit |
| Input resolution | 368 × 1232 | `model.py:MAX_DISP=192, MATCH_SCALE=8` |
| Input channels | 3 (RGB) | `data.py:read_image` |
| Input dtype | float32 (normalized) | `data.py:_normalize` |
| Normalization | ImageNet mean/std | `data.py:MEAN, STD` |
| Max disparity | 192 px (full-res) | `model.py:MAX_DISP` |
| Matching scale | 1/8 (MATCH_SCALE=8) | `model.py:MATCH_SCALE` |
| Disparity hypotheses | 24 (NUM_DISP) | `model.py:NUM_DISP` |
| Output shape | [1, 1, 368, 1232] | `model.py` forward |
| Output units | Full-res disparity pixels | `model.py` clamp [0, MAX_DISP] |
| ONNX path | `artifacts/hailo_stereo.onnx` | 3.38 MB |
| HEF path | `artifacts/hailo_stereo_hailo15h.hef` | 4.12 MB, 7 contexts |
| Dataset | KITTI 2015 val (40 scenes) | `data.py:Kitti2015 split="val"` |
| Camera f × B | 386.48 px·m | `eval_depth.py:KITTI_FOCAL × KITTI_BASELINE` |

---

## B. Exact Numerical Statistics for Each Output

### Output A — Stereo Input Images
- **Scene:** `000160_10` (KITTI 2015 val)
- **Left:** `data/kitti2015/training/image_2/000160_10.png` (1238×374 → 1232×368 cropped)
- **Right:** `data/kitti2015/training/image_3/000160_10.png` (1238×374 → 1232×368 cropped)
- **Crop:** Bottom-right 6px from left, 6px from top
- **Rectified:** Yes (KITTI 2015 stereo pairs)
- **Baseline:** 0.54 m
- **Focal:** 715.7 px
- **Principal point:** cx=609.56, cy=172.85 (full frame) → cx=603.56, cy=166.85 (cropped)

### Output B — Disparity Map (MEASURED from ONNX inference)

| Statistic | Value | Source |
|-----------|-------|--------|
| Shape | [368, 1232] | MEASURED |
| Dtype | float32 | MEASURED |
| Min | 3.8491 px | MEASURED |
| Max | 141.1353 px | MEASURED |
| Mean | 43.1916 px | MEASURED |
| Median | 31.4976 px | MEASURED |
| Std | 35.5529 px | MEASURED |
| P1 | 5.4477 px | MEASURED |
| P5 | 6.8197 px | MEASURED |
| P25 | 14.9568 px | MEASURED |
| P75 | 58.8951 px | MEASURED |
| P95 | 122.5095 px | MEASURED |
| P99 | 134.8870 px | MEASURED |
| NaN count | 0 (0%) | MEASURED |
| Inf count | 0 (0%) | MEASURED |
| Zero count | 0 (0%) | MEASURED |
| Negative count | 0 (0%) | MEASURED |
| Disparity range (config) | [0, 192] px | CONFIGURED (model.py) |
| Hypotheses | 24 | CONFIGURED (model.py) |
| Output scaling | Direct (1×) | CONFIGURED |
| Clamping | [0, 192] at model output | CONFIGURED (model.py:399) |

**Accuracy Metrics (single scene 000160_10):**

| Protocol | Valid Pixels | EPE (px) | D1-all (%) |
|----------|--------------|----------|------------|
| Masked | 88,974 | 0.9102 | 4.60 |
| Official | 101,301 | 0.9108 | 4.65 |

**Aggregate (40 val scenes):**

| Protocol | Total Pixels | EPE (px) | D1-all (%) |
|----------|--------------|----------|------------|
| Masked | 3,353,290 | **1.156** | **6.38%** |
| Official | 3,861,014 | **1.248** | **6.87%** |

*Matches in-loop validation EPE exactly (1.156/1.248 px) — PREVIOUSLY REPORTED in README*

---

### Output C — Metric Depth Map (CALCULATED from disparity)

**Conversion:** `Z = f × B / d` with `f=715.7 px`, `B=0.54 m`, `fb=386.48 px·m`, `d` floored at 1e-3 px.

| Statistic | Value | Source |
|-----------|-------|--------|
| Shape | [368, 1232] | CALCULATED |
| Dtype | float32 | CALCULATED |
| Valid pixels | 453,376 (100%) | MEASURED |
| Min | 2.7384 m | MEASURED |
| Max | 100.4077 m | MEASURED |
| Mean | 18.4438 m | MEASURED |
| Median | 12.2701 m | MEASURED |
| Std | 16.5193 m | MEASURED |
| P1 | 2.8652 m | MEASURED |
| P5 | 3.1547 m | MEASURED |
| P25 | 6.5621 m | MEASURED |
| P75 | 25.8396 m | MEASURED |
| P95 | 56.6705 m | MEASURED |
| P99 | 70.9429 m | MEASURED |

**Depth Band Distribution (predicted):**

| Range | Pixels | % |
|-------|--------|---|
| 0–5 m | 76,474 | 16.87% |
| 5–10 m | 112,933 | 24.91% |
| 10–20 m | 111,363 | 24.56% |
| 20–40 m | 100,754 | 22.22% |
| 40–80 m | 51,299 | 11.31% |
| >80 m | 553 | 0.12% |

**Per-Band Metrics (Official Protocol, 40 scenes aggregate):**

| Range | Pixels | MAE (m) | RMSE (m) | Median (m) | AbsRel (%) | δ<1.25 (%) | EPE (px) | Floor (m) | Bias dd (px) | Bias dZ (m) |
|-------|--------|---------|----------|------------|------------|------------|----------|-----------|--------------|-------------|
| 0–5 m | 71,399 | 0.45 | 1.52 | 0.17 | 11.30 | 92.73 | 7.288 | 0.31 | -5.01 | +0.35 |
| 5–10 m | 1,586,946 | 0.20 | 0.82 | 0.10 | 2.61 | 99.10 | 1.203 | 0.18 | +0.12 | +0.02 |
| 10–20 m | 1,463,066 | 0.58 | 1.78 | 0.24 | 4.05 | 97.56 | 1.074 | 0.53 | -0.04 | +0.16 |
| 20–40 m | 550,850 | 2.23 | 4.29 | 1.03 | 7.85 | 93.14 | 1.088 | 2.09 | -0.03 | +0.62 |
| 40–80 m | 187,791 | 6.69 | 9.61 | 4.50 | 12.23 | 84.05 | 1.159 | 8.55 | +0.62 | -3.04 |
| **All ≤80 m** | **3,860,052** | **0.96** | **2.94** | **0.19** | **4.53** | **96.82** | **1.248** | **0.72** | **-0.03** | **+0.02** |

*Geometric floor: `Z²/(f·B) × EPE_band` — measured error near floor means geometry-limited (CONFIGURED formula from eval_depth.py)*

---

### Output D — 3D Point Cloud (CALCULATED from depth)

**Reconstruction:** `Z = f·B/d`, `X = (u-cx)·Z/f`, `Y = (v-cy)·Z/f`  
**Coordinate frame:** Left camera, X right, Y down, Z forward (metres)  
**Camera (cropped):** f=715.70, cx=603.56, cy=166.85, B=0.540

| Statistic | Value | Source |
|-----------|-------|--------|
| Input pixels | 453,376 | MEASURED |
| Beyond range (>80m) | 553 (0.1%) | MEASURED |
| Border dropped (left 192 cols) | 70,656 (15.6%) | MEASURED |
| Valid points | 382,167 (84.3%) | MEASURED |
| Format | Binary PLY (x,y,z,rgb) | CONFIGURED |
| X range | -23.7 .. +19.3 m | MEASURED |
| Y range | -8.4 .. +1.8 m | MEASURED |
| Z range | 2.7 .. 80.0 m | MEASURED |
| Units | Metres | CONFIGURED |

---

### Output E — Depth Discontinuity Map (MEASURED)

**Formula:** `√((∂Z/∂x)² + (∂Z/∂y)²)` with central differences, forward/backward at borders  
**Input:** Predicted metric depth (metres)  
**Output:** Gradient magnitude in m/px

| Statistic | Value | Source |
|-----------|-------|--------|
| Shape | [368, 1232] | MEASURED |
| Valid pixels | 88,974 (19.6%) | MEASURED |
| Min | 0.000180 m/px | MEASURED |
| Max | 23.1787 m/px | MEASURED |
| Mean | 0.1903 m/px | MEASURED |
| Median | 0.0828 m/px | MEASURED |
| Std | 0.4014 m/px | MEASURED |
| P1 | 0.0088 m/px | MEASURED |
| P5 | 0.0200 m/px | MEASURED |
| P25 | 0.0439 m/px | MEASURED |
| P75 | 0.1945 m/px | MEASURED |
| P95 | 0.6696 m/px | MEASURED |
| P99 | 1.4228 m/px | MEASURED |

*Note: GT discontinuity shows extreme values due to sparse LiDAR (not comparable)*

---

### Output F — Near/Far Spatial Map & Occupancy Grid (CALCULATED)

**Grid:** 23 rows × 77 cols (16×16 px cells)  
**Rule:** Median depth per cell; `< 10 m → NEAR`, `≥ 10 m → FAR`, valid fraction `< 10% → INSUFFICIENT_DATA`

| Label | Count | % |
|-------|-------|---|
| NEAR | 341 | 19.3% |
| FAR | 549 | 31.0% |
| INSUFFICIENT_DATA | 881 | 49.7% |
| **Total** | **1,771** | **100%** |

*INSUFFICIENT_DATA high due to KITTI sparse GT (only ~20% pixels valid in masked protocol)*

---

## C. Ground-Truth Accuracy Metrics and Evaluation Sample Size

| Metric | Value | Protocol | Scenes | Pixels | Source |
|--------|-------|----------|--------|--------|--------|
| Disparity EPE | 1.156 px | Masked | 40 | 3,353,290 | MEASURED |
| Disparity EPE | 1.248 px | Official | 40 | 3,861,014 | MEASURED |
| Disparity D1 | 6.38% | Masked | 40 | 3,353,290 | MEASURED |
| Disparity D1 | 6.87% | Official | 40 | 3,861,014 | MEASURED |
| Depth MAE (all) | 0.94 m | Masked | 40 | 3,352,328 | MEASURED |
| Depth MAE (all) | 0.96 m | Official | 40 | 3,860,052 | MEASURED |
| Depth RMSE (all) | 2.89 m | Masked | 40 | 3,352,328 | MEASURED |
| Depth RMSE (all) | 2.94 m | Official | 40 | 3,860,052 | MEASURED |
| Depth δ<1.25 (all) | 96.94% | Masked | 40 | 3,352,328 | MEASURED |
| Depth δ<1.25 (all) | 96.82% | Official | 40 | 3,860,052 | MEASURED |

**Per-scene metrics saved to:** `scene_metrics.csv`  
**Band metrics saved to:** `depth_band_metrics.csv`

---

## D. Per-Cell Occupancy Results

Saved to `occupancy_cells.csv` with columns:
`row, col, y0, y1, x0, x1, valid_pixels, total_pixels, valid_fraction, median_depth_m, label`

Summary: 1,771 cells, 19.3% NEAR, 31.0% FAR, 49.7% INSUFFICIENT_DATA

---

## E. Runtime and Hardware Measurements

| Stage | Time | Device | Notes |
|-------|------|--------|-------|
| ONNX inference (single pair) | ~0.15 s | CPU (onnxruntime) | Not optimized; PyTorch unavailable |
| Full val (40 scenes) | ~6 s | CPU | Includes I/O and metric computation |
| FPS (est.) | ~6.7 FPS | CPU | Single-threaded, no batching |

**Blocker:** PyTorch `torch_python.dll` blocked by Windows Application Control — cannot measure native PyTorch or Hailo-8/15H latency.  
**HEF profiling:** `hailo profiler` crashes on this graph (DFC 5.4.0 bug: conv_output_shape mismatch from spatial-defused residual add) — PREVIOUSLY REPORTED in README.

---

## F. Paths to All Saved Artifacts

```
results/hailostereo_output_audit/
├── environment.json           # Software, hardware, commit, model, calibration
├── input_metadata.json        # Exact image and preprocessing metadata
├── disparity.npy              # Raw disparity [368,1232] float32
├── disparity_stats.json       # Disparity statistics + accuracy
├── depth.npy                  # Raw depth [368,1232] float32 (metres)
├── depth_stats.json           # Depth statistics + band metrics
├── depth_band_metrics.csv     # Per-band metrics for both protocols
├── pointcloud.ply             # 382K points, binary PLY (x,y,z,rgb)
├── pointcloud_stats.json      # Point cloud statistics
├── discontinuity.npy          # Depth gradient magnitude [368,1232]
├── discontinuity_stats.json   # Discontinuity statistics
├── occupancy_cells.csv        # 1,771 cells with labels
├── occupancy_stats.json       # Grid summary
├── scene_metrics.csv          # Per-scene EPE/D1/depth pixels
├── aggregate_metrics.json     # Aggregate metrics (40 scenes)
├── 01_stereo_inputs.png       # Left | Right
├── 02_disparity_map.png       # Left | Pred Disp | GT Disp | Error
├── 03_metric_depth.png        # Left | Pred Depth | GT Depth | Rel Error
├── 04_point_cloud.png         # Depth-colored projection
├── 05_depth_discontinuity.png # Gradient magnitude visualization
└── 06_occupancy_grid.png      # NEAR/FAR/INSUFFICIENT grid with legend
```

---

## G. Values Unavailable or Unverified

| Value | Reason |
|-------|--------|
| Hailo-15H on-device FPS/latency | No Hailo board available; `hailo profiler` crashes (DFC bug) |
| Hailo-8 on-device metrics | Different DFC version (3.x) required; not installed |
| PyTorch native inference parity | `torch_python.dll` blocked by Windows Application Control |
| Per-scene depth band metrics for <5m band | Only ~1,800 pixels/scene in 0-5m (GT sparse) |
| KITTI calib file intrinsics | Not in checkout; nominal values used throughout |
| Quantized (int8) ONNX metrics | Not exported; only HEF via DFC (Linux-only) |
| Temperature value (learned) | Not extracted from checkpoint; folded at export |
| Softmax entropy per pixel | Not exported by ONNX; available only in PyTorch path |

---

## H. Exact Commands to Reproduce

```bash
# Environment: C:\Microchip\hailo-stereo\.venv (Python 3.11, onnxruntime 1.29.0)

# 1. Disparity inference + stats (single scene)
python run_inference.py

# 2. Depth evaluation + band metrics (single scene)
python run_depth_eval.py

# 3. Point cloud generation (single scene)
python run_pointcloud.py

# 4. Depth discontinuity map (single scene)
python run_discontinuity.py

# 5. Occupancy grid (single scene)
python run_occupancy.py

# 6. Visualizations
python run_visualizations.py

# 7. Full validation set (40 scenes)
python run_full_eval.py

# Original repo commands (require Linux + DFC 5.4.0 + PyTorch):
# Geometry test:     python src/test_disparity.py
# KITTI accuracy:    python src/eval_kitti.py --ckpt runs/kitti_mixed_fixed768/best.pt --root data/kitti2015/training --bins
# Depth bands:       python src/eval_depth.py --ckpt runs/kitti_mixed_fixed768/best.pt --root data/kitti2015/training
# ONNX export+audit: python src/export_onnx.py --ckpt runs/kitti_mixed_fixed768/best.pt
# Preview:           python src/preview.py --ckpt runs/kitti_mixed_fixed768/best.pt --dataset kitti --root data/kitti2015/training --n 4
# Deploy (Linux):    ./deploy/hailo-py deploy/dfc_flow.py parse && ./deploy/hailo-py deploy/dfc_flow.py optimize --opt-level 2 && ./deploy/hailo-py deploy/dfc_flow.py compile --compiler-effort 0
```

---

## Label Key

- **MEASURED** — Directly observed from ONNX inference output on real data
- **CALCULATED** — Derived from measured outputs using documented formulas
- **CONFIGURED** — Fixed hyperparameter or constant from source code
- **PREVIOUSLY REPORTED** — Published in README.md or training logs
- **UNAVAILABLE** — Cannot be determined with current environment/hardware