# EdgeDepth-8-prered (v1.0)

Same math as EdgeDepth-8, repackaged for the Hailo-8 dataflow compiler.
No retraining: weights are remapped, not relearned.

**Source:** `release/EdgeDepth-8/EdgeDepth-8.pt` (arch: {'num_disp': 24, 'groups': 8, 'match_ch': 32, 'cost': 'both', 'feat_ch': 32, 'n_res': 6, 'agg_ch': 64, 'refine': 'base'})

## Why not grouped

EdgeDepth-8-grouped (`src/model_grouped.py`, GroupedCostVolume) fails Hailo-8
allocation: its cost volume is a 1536-channel tensor (cat[sub 768 | corr 768])
feeding one grouped 1x1 conv (1536 -> 384, groups=384), which the compiler
re-splits into 24 pieces and cannot route. This model keeps the identical
output channel layout (`[all sub (d-major) | all corr (d-major)]`, so the same
stem input-channel permutation applies) but narrows the tensors that feed it.

## What changed

Original per hypothesis d (`CostVolume` in `src/model.py`, cost="both";
`reduce` / `reduce_corr` are `Conv2d(C, G, 1, groups=G, bias=False)`, C=32,
G=8, D=24):

- `sub_d = reduce(left - shift_d(right))`
- `corr_d = reduce_corr(left * shift_d(right))`

Because `reduce` is linear, has NO bias, is 1x1, and the shift pads with zeros
(`reduce(0)=0`):

- `reduce(left - shift_d(right)) == reduce(left) - shift_d(reduce(right))`

`PreReducedCostVolume` (`src/model_prered.py`) computes:

1. `rl = reduce(left)` -> `[B, G, h, w]`, `rr = reduce(right)` (same weights
   as the original `reduce`, copied not tiled)
2. `sub = cat([rl]*D, 1) - cat(_shift_all(rr, D), 1)` -> `[B, D*G, h, w]`
   (one narrow ew_sub on 192 channels, not 1536)
3. `big_left = cat([left]*D, 1)`, `big_right = cat(_shift_all(right, D), 1)`
   -> `[B, D*C, h, w]` (768 ch)
4. `corr = fused_corr(big_left * big_right)` with
   `Conv2d(D*C, D*G, 1, groups=D*G)` whose weight is `reduce_corr.weight`
   tiled D times
5. `cat([sub, corr], 1)` -> `[B, 2*D*G, h, w]` = 384 ch, same layout as
   GroupedCostVolume, so the aggregation.stem input-channel permutation
   (`_stem_channel_perm`, reused by import) applies unchanged.

(`cat([x]*D)` is used rather than `repeat` so the export stays free of the
shape-dynamic Tile op; `_shift_all`, `_stem_channel_perm`, `load_original`
and `strip_module_prefix` are reused by import.)

## Equivalence (must be <= 1e-3 px)

| check | max |disp_orig - disp_prered| |
|---|---|---|
| random 1x3x368x1232 (seed 0) | 2.136e-04 px |
| KITTI 2015 val, worst of 40 pairs | 8.011e-04 px (pair 14) |
| onnxruntime vs torch, worst of 2 pairs | 5.569e-04 px |

## Accuracy: KITTI 2015 val (40 pairs), float

| Protocol | original EPE | prered EPE | original D1-all | prered D1-all |
|----------|--------------|-------------|-----------------|----------------|
| official | 1.067 px | 1.067 px | 5.06% | 5.06% |
| masked   | 0.987 px | 0.987 px | 4.53% | 4.53% |

Prered matches original to 3 decimals on both protocols.

## Op counts (ONNX node types)

`EdgeDepth-8.onnx` vs `EdgeDepth-8-grouped.onnx` vs `EdgeDepth-8-prered.onnx`:

| op | original | grouped | prered |
|---|---|---|---|
| Conv | 115 | 68 | 70 |
| Sub | 24 | 1 | 1 |
| Mul | 27 | 4 | 4 |
| Concat | 27 | 7 | 10 |
| Slice | 24 | 24 | 48 |
| TOTAL nodes | 417 | 304 | 430 |

## ONNX channel widths of the cost-volume tensors

Measured by ONNX shape inference at matching resolution
(46x154 for the fixed 368x1232 export).
`sub_in`/`mul_in` = widest Sub/Mul operand channels,
`concat_out` = widest Concat tensor channels,
`conv_in` = widest matching-resolution Conv input channels.

| cost-volume tensor | original ch | grouped ch | prered ch |
|---|---|---|---|
| sub_in | 32 | 768 | 192 |
| mul_in | 32 | 768 | 768 |
| concat_out | 384 | 1536 | 768 |
| conv_in | 384 | 1536 | 768 |

Statically, the prered graph replaces the grouped 1536-channel
`cat[sub 768 | corr 768]` with a 192-channel sub path
(`cat([rl]*D)` / shifted `rr`, 8 -> 192 ch) alongside the 768-channel corr
path (`big_left`/`big_right`/product, 768 ch), concatenated once to the same
384-channel `[sub 192 | corr 192]` cost volume the stem consumes.

## MACs (`model.count_ops`, 368x1232)

| stage | original GMAC | prered GMAC |
|---|---|---|
| refine1 | 4.505 | 4.505 |
| aggregation | 3.819 | 3.819 |
| refine2 | 2.791 | 2.791 |
| features | 2.089 | 2.089 |
| refine4 | 1.779 | 1.779 |
| cost_volume | 0.011 | 0.006 |
| soft_argmin | 0.000 | 0.000 |

| TOTAL | 14.994 | 14.989 |

## Files

- `EdgeDepth-8-prered.pt` (state_dict + arch + `"cost_layout": "prered"` + source note)
- `EdgeDepth-8-prered.onnx` (opset 14, inputs `left`/`right`, output
  `disparity`, fixed 368x1232, fused soft-argmin -- same export
  settings as `src/export_onnx.py`)
