# EdgeDepth-8-dense24 on the Raspberry Pi 5 + Hailo-8

40 KITTI 2015 val pairs, scored with the same masks and pixel-pooled metrics
as `deploy/dfc_flow.py` (`kitti_val_pairs`, `score`). The HEF takes RAW uint8
RGB 368x1232x3 `left`/`right` (normalization is inside the HEF) and outputs
uint8 disparity 368x1232x1, which HailoRT dequantizes to float32.

Tested on: Pi 5, Hailo-8 AI HAT+, HailoRT 4.23.0, `dtparam=pciex1_gen=3`.

## 1. Pack pairs (host)

```bash
python deploy/pi/pack_val_pairs.py --out deploy/pi/kitti_val40.npz
```

## 2. Run the HEF (on the Pi)

`run_hef.py` needs only numpy + `hailo_platform`, so it is copied standalone.

```bash
scp release/EdgeDepth-8-dense24/hailo8/EdgeDepth-8_hailo8.hef \
    deploy/pi/kitti_val40.npz deploy/pi/run_hef.py <user>@<pi>:~/edgedepth/
ssh <user>@<pi> 'cd ~/edgedepth && python run_hef.py --hef EdgeDepth-8_hailo8.hef \
    --pairs kitti_val40.npz --out pred_dense24.npz'
scp <user>@<pi>:~/edgedepth/pred_dense24.npz deploy/pi/
```

## 3. Score (host)

```bash
python deploy/pi/score_preds.py --pairs deploy/pi/kitti_val40.npz \
    --pred dense24=deploy/pi/pred_dense24.npz --out deploy/pi/dense24_results.json
python deploy/pi/full_metrics.py --pairs deploy/pi/kitti_val40.npz \
    --pred dense24=deploy/pi/pred_dense24.npz --out deploy/pi/full_metrics.json
```

`full_metrics.py` adds disparity RMSE, depth MAE/RMSE, AbsRel, δ thresholds
and near/medium/far range bands.

## Demos

- `demo_kitti.py`: runs the HEF live on the Pi over the packed pairs and prints per-frame EPE / latency.
- `demo_kitti_view.py`: same, with a disparity/depth view (`--pred-npz` replays saved predictions without the chip).
- `viewer2/stereo_view.py`: host-side viewer for `kitti_val40.npz` + `pred_dense24.npz`.

## Results

| File | Contents |
|---|---|
| `dense24_results.json` | EPE / D1 (masked + official), per-image EPE, FPS |
| `full_metrics.json` | full disparity + depth metrics; also scores the earlier EdgeDepth HEF and Model Zoo `stereonet` for comparison |
| `bench_dense24.json` | 200-frame latency breakdown (send / device / post), memory |
| `bench_shipped.json`, `bench_zoo.json` | the same benchmark for the two comparison models |
| `figures/` | accuracy, latency/FPS, error vs. distance, accuracy vs. latency, utilization, latency distribution |
| `pointclouds/` | Hailo-8 point clouds for KITTI frames 160, 173, 186 (`.ply` + `.png`) |

Headline: **1.437 px EPE / D1 6.69% (official), 1.372 px / 6.17% (masked),
99.4 ms, 10.07 FPS.**

FPS values for the comparison models in `full_metrics.json` were measured at
PCIe Gen2. Use the Gen3 numbers in `bench_*.json` (earlier HEF 7.50 FPS,
stereonet 5.19 FPS).

The Pi link copies at ~0.3 MB/s, so compress predictions with
`np.savez_compressed` on the Pi before `scp`.
