# hailo_stereo.onnx -- operator-level graph stats

Source `artifacts\hailo_stereo.onnx`: 347 nodes, 101 initializers (0.80M weights), opset 14, 316 operator-to-operator edges (+106 Constant-op consumers = 422 tensor edges).
Graphs: `hailo_stereo_graph.dot` (+`.svg`/`.png`), `hailo_stereo_graph.mmd`, module overview `hailo_stereo_modules.dot` (+`.svg`/`.png`).

## Top-level flow (verified from tensor connectivity)

Branch A (un-suffixed nodes) consumes `right`, branch B (`_1` suffixes) consumes `left`; both copies share every weight initializer (see shared table below).

```
IO:right / IO:left -> stem2/4/8 -> res.0..5 -> head | head_1
branch A (right) -> /features/head -> Concat[zero-pad, right-feat]
  -> 24x Slice @shift d -> Sub(head_1 - slice)  (left-branch grid)
  -> 24x reduce Conv 1x1 (grouped 32->8) -> Concat_1 (192ch cost volume)
  -> aggregation (stem, res_a/b, down, res_c/d, Resize+merge, Add,
     res_e, out 3x3 -> 24 channel hypotheses)
  -> Softmax -> /soft_argmin/index Conv 24->1 (disparity @1/8 res)
refine4/2/1 (x3): Resize(scale 2) -> Mul(2.0) -> Concat with the
  branch-B (left) stem4 (r4) / stem2 (r2) feature / raw IO:left (r1)
  -> blocks -> out
  -> Add   (up-samples disparity 1/8 -> 1/4 -> 1/2 -> 1/1)
Clip -> IO:disparity
```
Every disparity hypothesis is an explicit Slice/Sub/Conv arm (`/cost_volume/*0..*23`); there is no disparity-search op in ONNX -- the argmin IS the soft-argmin index conv.

## PyTorch cross-check

Against `HailoStereo` (`src/model.py`) rebuilt from `runs\kitti_mixed_fixed768\best.pt` -- the checkpoint `artifacts/hailo_stereo.onnx` was exported from (BN folds reproduced, the soft-argmin temperature folded with its trained sign, and a full torch-vs-onnxruntime run).

| check | result | detail |
|---|---|---|
| torch params with an ONNX initializer of the same name are bit-identical | PASS | 9 of 101 initializers covered by this rule (head, reduce, refine-stage outs, index ramp) |
| every Conv2d+BatchNorm2d fold materialised by constant folding matches a torch pair exactly | PASS | 46 folded ONNX convs x 46 matched, temperature t = 1.9305 pre-scaled onto the aggregation cost head |
| siamese tower: 2 conv-node copies per torch conv, sharing 1 initializer set | PASS | 16 torch convs -> 16 + 16 nodes, 31 initializers on 2 consumers each (features.head.weight is one of them) |
| cost volume: the shared reduce initializer has 1 consumer per hypothesis arm | PASS | reduce on 24 nodes, 24 Sub, 24 Slice, D=24 |
| soft-argmin index ramp == arange(D) on both sides | PASS | [0..23] on both sides |
| refine ladder: 3 stages, one Mul(2.0) each | PASS | 3 RefineStage, 3 Mul(2.0) |
| final Clip bounds == [0, MAX_DISP] = [0.0, 192.0] | PASS | [0.0, 192.0] |
| numeric parity: torch forward vs onnxruntime (randn 1x3x368x1232, seed 0) | PASS | max |diff| = 4.43e-04 px (tolerance 1e-02), shapes (1, 1, 368, 1232) |

## UNRECOVERED from the ONNX / limitations of the graph evidence

Every row is verified directly against `artifacts\hailo_stereo.onnx`; status follows the report.md labeling convention (UNAVAILABLE = not in the file at all, FOLDED → gone during export, OUTSIDE = not part of this artifact, INFERRED = only recoverable by matching, not read off).

| value | status | reason |
|---|---|---|
| torch module names for the folded conv/BN weights | INFERRED | 92 of 101 initializers are auto-renamed `onnx::Conv_NNN` (only 9 keep real names); identity of the 61 folded conv nodes follows only from the numeric fold-matching in the cross-check above |
| BatchNorm running_mean / running_var / eps | FOLDED | merged into each conv's W/b during export; the raw BN tensors are absent from the graph and the fold is one-way |
| soft-argmin temperature scalar | FOLDED | pre-scaled with a sign flip onto the aggregation cost head during constant folding (see the fold row above); the scalar itself is not stored |
| training state (epoch, optimizer, loss history) | UNAVAILABLE | inference-only export; no `metadata_props`, empty doc_string |
| dynamic / alternate input resolutions | UNAVAILABLE | shapes are frozen at 1x3x368x1232 at export time |
| preprocessing (ImageNet mean/std, 16-multiple crop, left-first ordering) | OUTSIDE | no normalization ops in the graph; the caller must reproduce it |
| confidence / entropy surface | UNAVAILABLE | the 24-way softmax is computed internally (`/soft_argmin/Softmax`) and consumed by the index conv; IO exposes only `disparity` |
| quantized variants (Hailo-8 / Hailo-15H INT8) | OUTSIDE | this artifact is the float32 opset 14 graph; the INT8 profiles live in the HEF artifacts under `deploy/` |

## Clusters (first 1-2 path components of the node names)

| cluster | nodes |
|---|---|
| cost_volume | 171 |
| features/res | 60 |
| refine4/blocks | 15 |
| refine2/blocks | 10 |
| refine1/blocks | 10 |
| refine4 | 6 |
| refine2 | 6 |
| refine1 | 6 |
| aggregation/res_a | 5 |
| aggregation/res_b | 5 |
| aggregation/res_c | 5 |
| aggregation/res_d | 5 |
| aggregation/res_e | 5 |
| features/stem2 | 4 |
| features/stem4 | 4 |
| features/stem8 | 4 |
| aggregation | 3 |
| aggregation/stem | 2 |
| aggregation/down | 2 |
| aggregation/merge | 2 |
| soft_argmin | 2 |
| refine4/stem | 2 |
| refine2/stem | 2 |
| refine1/stem | 2 |
| features/head | 1 |
| features/head_1 | 1 |
| aggregation/out | 1 |
| refine4/out | 1 |
| refine2/out | 1 |
| refine1/out | 1 |
| Constant | 1 |
| Constant_1 | 1 |
| Clip | 1 |

## Ops

| op | count |
|---|---|
| Constant | 106 |
| Conv | 91 |
| LeakyRelu | 60 |
| Add | 28 |
| Slice | 24 |
| Sub | 24 |
| Concat | 5 |
| Resize | 4 |
| Mul | 3 |
| Softmax | 1 |
| Clip | 1 |

## Initializers consumed by more than one node (the siamese-weight-sharing evidence)

| tensor | consumers |
|---|---|
| cost_volume.reduce.weight | 24 |
| onnx::Conv_855 | 2 |
| onnx::Conv_856 | 2 |
| onnx::Conv_858 | 2 |
| onnx::Conv_859 | 2 |
| onnx::Conv_861 | 2 |
| onnx::Conv_862 | 2 |
| onnx::Conv_864 | 2 |
| onnx::Conv_865 | 2 |
| onnx::Conv_867 | 2 |
| onnx::Conv_868 | 2 |
| onnx::Conv_870 | 2 |
| onnx::Conv_871 | 2 |
| onnx::Conv_873 | 2 |
| onnx::Conv_874 | 2 |
| onnx::Conv_876 | 2 |
| onnx::Conv_877 | 2 |
| onnx::Conv_879 | 2 |
| onnx::Conv_880 | 2 |
| onnx::Conv_882 | 2 |
| onnx::Conv_883 | 2 |
| onnx::Conv_885 | 2 |
| onnx::Conv_886 | 2 |
| onnx::Conv_888 | 2 |
| onnx::Conv_889 | 2 |
| onnx::Conv_891 | 2 |
| onnx::Conv_892 | 2 |
| onnx::Conv_894 | 2 |
| onnx::Conv_895 | 2 |
| onnx::Conv_897 | 2 |
| onnx::Conv_898 | 2 |
| features.head.weight | 2 |
