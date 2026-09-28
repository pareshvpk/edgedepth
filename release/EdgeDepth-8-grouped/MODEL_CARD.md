# EdgeDepth-8-grouped (v1.0)

Same math as EdgeDepth-8, repackaged for the Hailo-8 dataflow compiler.
No retraining: weights are remapped, not relearned.

**Source:** `release/EdgeDepth-8/EdgeDepth-8.pt` (arch: {'num_disp': 24, 'groups': 8, 'match_ch': 32, 'cost': 'both', 'feat_ch': 32, 'n_res': 6, 'agg_ch': 64, 'refine': 'base'})

## What changed

CostVolume.forward looped over 24 disparity hypotheses: per hypothesis an
ew_sub -> grouped 1x1 (`reduce`) and an ew_mult -> grouped 1x1
(`reduce_corr`), then a concat of 48 small tensors (~100 tiny layers in the
Hailo graph; the 'both' arch failed Hailo-8 allocation).

The grouped model (`src/model_grouped.py`) fuses that into wide layers:

1. `L = torch.cat([left] * D, 1)` -> `[B, D*C, h, w]`
2. `R = torch.cat(_shift_all(right, D), 1)` (same static pad-once/slice-many
   shift, reused by import)
3. `S = L - R` (one ew_sub), `M = L * R` (one ew_mult)
4. `X = torch.cat([S, M], 1)` -> one `Conv2d(2*D*C, 2*D*G, 1,
   groups=2*D*G)`: first D*G groups = `reduce.weight` tiled D times, next
   D*G = `reduce_corr.weight` tiled D times.

The fused conv emits `[all sub (d-major) | all corr]` instead of the original
per-d `[sub | corr]` interleave. No permute op is added to the graph; the
INPUT channels of `aggregation.stem`'s convolution are permuted at conversion
time instead. (`cat([left]*D)` is used rather than `repeat` so the export
stays free of the shape-dynamic Tile op.)

## Equivalence (must be <= 1e-3 px)

| check | max |disp_orig - disp_grouped| |
|---|---|---|
| random 1x3x368x1232 (seed 0) | 2.899e-04 px |
| KITTI 2015 val, worst of 40 pairs | 4.730e-04 px (pair 7) |
| onnxruntime vs torch, worst of 2 pairs | 6.027e-04 px |

## Accuracy: KITTI 2015 val (40 pairs), float

| Protocol | original EPE | grouped EPE | original D1-all | grouped D1-all |
|----------|--------------|-------------|-----------------|----------------|
| official | 1.067 px | 1.067 px | 5.06% | 5.06% |
| masked   | 0.987 px | 0.987 px | 4.53% | 4.53% |

Grouped matches original to 3 decimals on both protocols.

## Op counts (ONNX node types)

`EdgeDepth-8.onnx` vs `EdgeDepth-8-grouped.onnx`:

| op | original | grouped |
|---|---|---|
| Conv | 115 | 68 |
| Sub | 24 | 1 |
| Mul | 27 | 4 |
| Concat | 27 | 7 |
| Slice | 24 | 24 |
| TOTAL nodes | 417 | 304 |

## MACs (`model.count_ops`, 368x1232)

| stage | original GMAC | grouped GMAC |
|---|---|---|
| refine1 | 4.505 | 4.505 |
| aggregation | 3.819 | 3.819 |
| refine2 | 2.791 | 2.791 |
| features | 2.089 | 2.089 |
| refine4 | 1.779 | 1.779 |
| cost_volume | 0.011 | 0.011 |
| soft_argmin | 0.000 | 0.000 |

| TOTAL | 14.994 | 14.994 |

## Files

- `EdgeDepth-8-grouped.pt` (state_dict + arch + `"cost_layout": "grouped"` + source note)
- `EdgeDepth-8-grouped.onnx` (opset 14, inputs `left`/`right`, output
  `disparity`, fixed 368x1232, fused soft-argmin -- same export
  settings as `src/export_onnx.py`)
