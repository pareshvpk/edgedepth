# Pi on-device smoke test (40 KITTI val pairs, two Hailo-8 HEFs)

Metrics reuse `deploy/dfc_flow.py` (`kitti_val_pairs`, `score`, pixel-weighted
aggregation). Both HEFs take RAW uint8 RGB 368x1232x3 LEFT/RIGHT and output
uint8 disparity 368x1232x1.

## 1. Pack pairs (host)

```bash
python deploy/pi/pack_val_pairs.py --out deploy/pi/kitti_val40.npz
```

## 2. Run each HEF (on the Pi)

```bash
scp deploy/pi/kitti_val40.npz deploy/pi/run_hef.py pi@<pi>:~/
ssh pi@<pi> 'python run_hef.py --hef hailo_stereo_hailo8.hef --pairs kitti_val40.npz --out pred_ours.npz'
ssh pi@<pi> 'python run_hef.py --hef stereonet.hef --pairs kitti_val40.npz --out pred_zoo.npz'
scp pi@<pi>:~/pred_ours.npz pi@<pi>:~/pred_zoo.npz deploy/pi/
```

HEF paths: ours `deploy/build_hailo8_shipped/hailo_stereo_hailo8.hef`,
zoo `artifacts/stereonet.hef` (copy the one under test to the Pi first).

## 3. Score (host)

```bash
python deploy/pi/score_preds.py --pairs deploy/pi/kitti_val40.npz \
  --pred ours=deploy/pi/pred_ours.npz --pred zoo=deploy/pi/pred_zoo.npz
# table on stdout, JSON in deploy/pi/smoke_results.json
```
