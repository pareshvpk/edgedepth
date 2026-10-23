# EdgeDepth (v1.0)

Same math as the trained network (and its EdgeDepth-dense variant), repackaged so the Hailo-8
dataflow compiler can split the graph into contexts. No retraining: weights
are remapped, not relearned.

**Source:** `runs/kft_both_s60/best.pt`, epoch 58 (arch: {'num_disp': 24, 'groups': 8, 'match_ch': 32, 'cost': 'both', 'feat_ch': 32, 'n_res': 6, 'agg_ch': 64, 'refine': 'base'})

## Why not dense

EdgeDepth-dense (`DenseCorrCostVolume` in `src/model_dense.py`) calls
`_shift_all` twice -- once on `reduce(right)` (sub path, G=8 ch) and once on
`right` (corr path, C=32 ch) -- exporting 48 Slice layers, which the Hailo-8
compiler cannot split into contexts. This model shifts only once.

## What changed

Dense per the dense model (`DenseCorrCostVolume`, D=24, C=32, G=8):

- `sub = cat([rl]*D, 1) - cat(_shift_all(rr, D), 1)` with `rl = reduce(left)`,
  `rr = reduce(right)` (`reduce` = `Conv2d(C, G, 1, groups=G, bias=False)`)
- `corr = fused_corr(big_left * big_right)` with `big_left = cat([left]*D)`,
  `big_right = cat(_shift_all(right, D))` (dense `Conv2d(D*C, D*G, 1)`)

Because `reduce` is linear, has NO bias, is 1x1, and the shift pads with zeros
(`reduce(0)=0`): `shift_d(reduce(R)) == reduce(shift_d(R))` and
`reduce(L) - reduce(R_d) == reduce(L - R_d)`.

`DenseSubCorrCostVolume` (`src/model_dense24.py`) computes:

1. `big_left = cat([left]*D, 1)` -> `[B, D*C, h, w]` (768 ch)
2. `big_right = cat(_shift_all(right, D), 1)` -> `[B, D*C, h, w]` (the ONLY
   24 slices)
3. `sub = fused_sub(big_left - big_right)` with DENSE
   `Conv2d(D*C, D*G, 1, groups=1)` whose weight is the block-diagonal
   expansion of `reduce.weight` tiled D times (row `d*G+g` nonzero only at
   columns `d*C+g*4 .. d*C+g*4+3`)
4. `corr = fused_corr(big_left * big_right)` with the SAME dense weight as
   `DenseCorrCostVolume.fused_corr` (copied, 768 -> 192)
5. `cat([sub, corr], 1)` -> `[B, 2*D*G, h, w]` = 384 ch, same layout as
   dense, so the aggregation stem is copied unchanged.

(`cat([x]*D)` is used rather than `repeat` so the export stays free of the
shape-dynamic Tile op; `_shift_all`, `expand_grouped_to_dense`,
`_stem_channel_perm`, `load_original` and `strip_module_prefix` are reused
by import; `src/model_dense.py` and `src/model.py` are never modified.)

## Equivalence (must be <= 1e-3 px)

| check | max |disp_dense - disp_dense24| |
|---|---|
| random 1x3x368x1232 (seed 0) | 2.060e-04 px |
| KITTI 2015 val, worst of 40 pairs | 6.409e-04 px (pair 14) |
| onnxruntime vs torch, worst of 2 pairs | 6.027e-04 px |

## Accuracy: KITTI 2015 val (40 pairs), float

| Protocol | dense EPE | EdgeDepth EPE | dense D1-all | EdgeDepth D1-all |
|----------|--------------|-------------|-----------------|----------------|
| official | 1.067 px | 1.067 px | 5.06% | 5.06% |
| masked   | 0.987 px | 0.987 px | 4.53% | 4.53% |

EdgeDepth matches dense to 3 decimals on both protocols
(expected: 0.987 masked / 1.067 official EPE).

## Op counts (ONNX node types)

`EdgeDepth-original.onnx` vs `EdgeDepth-grouped.onnx` vs `EdgeDepth-prered.onnx` vs `EdgeDepth-dense.onnx` vs `EdgeDepth.onnx`:

| op | original | grouped | prered | dense | EdgeDepth |
|---|---|---|---|---|---|
| Conv | 115 | 68 | 70 | 70 | 69 |
| Sub | 24 | 1 | 1 | 1 | 1 |
| Mul | 27 | 4 | 4 | 4 | 4 |
| Concat | 27 | 7 | 10 | 10 | 7 |
| Slice | 24 | 24 | 48 | 48 | 24 |
| TOTAL nodes | 417 | 304 | 430 | 430 | 305 |

The EdgeDepth graph has exactly 24 Slice nodes (one `_shift_all`).

## ONNX channel widths of the cost-volume tensors

Measured by ONNX shape inference at matching resolution
(46x154 for the fixed 368x1232 export).
`sub_in`/`mul_in` = widest Sub/Mul operand channels,
`concat_out` = widest Concat tensor channels,
`conv_in` = widest matching-resolution Conv input channels.

| cost-volume tensor | original ch | grouped ch | prered ch | dense ch | EdgeDepth ch |
|---|---|---|---|---|---|
| sub_in | 32 | 768 | 192 | 192 | 768 |
| mul_in | 32 | 768 | 768 | 768 | 768 |
| concat_out | 384 | 1536 | 768 | 768 | 768 |
| conv_in | 384 | 1536 | 768 | 768 | 768 |

Statically, the EdgeDepth graph keeps the dense layout -- stacked 768-channel
`big_left`/`big_right` tensors feeding one Sub and one Mul, then the two
dense 1x1s, concatenated once to the same 384-channel
`[sub 192 | corr 192]` cost volume the stem consumes.

## MACs (`model.count_ops`, 368x1232)

| stage | dense GMAC | EdgeDepth GMAC |
|---|---|---|
| refine1 | 4.505 | 4.505 |
| aggregation | 3.819 | 3.819 |
| refine2 | 2.791 | 2.791 |
| features | 2.089 | 2.089 |
| refine4 | 1.779 | 1.779 |
| cost_volume | 1.045 | 2.089 |
| soft_argmin | 0.000 | 0.000 |

| TOTAL | 16.028 | 17.072 |

## Files

- `EdgeDepth.pt` (state_dict + arch + `"cost_layout": "dense24"` + source note)
- `EdgeDepth.onnx` (opset 14, inputs `left`/`right`, output
  `disparity`, fixed 368x1232, fused soft-argmin -- same export
  settings as `src/export_onnx.py`)

## Hailo-8 HEF

- HEF: `hailo8/EdgeDepth_hailo8.hef` (23.16 MB, 7 contexts,
  sha256 `96fd7441b8a8193306f899b81c116831a8edd562c9e6f11d21f74c919f575069`)
- int8 emulated (40 KITTI 2015 val pairs): EPE masked 1.240 / official 1.329,
  D1 official 6.32% (fp_optimized 0.987 / 1.067 / 5.06%)
- Full compile story: `hailo8/COMPILE_NOTES.md`
