# EdgeDepth-8 (v1.0)

Stereo disparity / depth network for edge deployment (Raspberry Pi 5 + Hailo-8 AI HAT+).
The "8" refers to the Hailo-8 target and the 1/8-resolution matching.

**Source run:** `runs/kft_both_s60/best.pt` (epoch 58). Original checkpoint left in place.
Files here are copies: `EdgeDepth-8.pt` (PyTorch checkpoint), `EdgeDepth-8.onnx` (= `runs/final_candidates/s60ep58.onnx`).

## Accuracy: KITTI 2015 val (40 pairs), float

| Protocol | EPE      | D1-all | Accuracy (100 - D1) |
|----------|----------|--------|---------------------|
| official | 1.067 px | 5.06%  | 94.9%               |
| masked   | 0.987 px | 4.53%  | 95.5%               |

## Model
- ~0.91 M parameters; 24 disparity hypotheses at 1/8 resolution, group-wise cost (groups 8)
- arch: num_disp 24, groups 8, match_ch 32, feat_ch 32, n_res 6, agg_ch 64, cost `both` (training args list cost `sub`)
- Trained: SceneFlow pretrain → KITTI 2015 + 2012 mixed fine-tune (60 ep, lr 5e-5, batch 4, crop 320x768, strong aug, EMA 0.999, scale aug 1.0-1.5 p0.5)

## Status
- Best of all variants tried (refine v2, v3, A' all finished ~1.12 official).
- Hailo-8 compile: not yet done (see notes on the hailo8 compile blocker).
