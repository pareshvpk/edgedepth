# EdgeDepth — Hailo-8 compile notes (2026-10-06)

## What the HEF is

EdgeDepth (s60 ep58, cost='both', float EPE masked 0.987 / official 1.067,
D1 4.53% masked / 5.06% official on 40 KITTI 2015 val pairs).

It is an exact rewrite of the cost volume, with the same weights and outputs
(max diff 6.4e-4 px).

Target: Hailo-8 (Pi 5 AI HAT+ 26 TOPS), DFC 3.34 (venv `~/.venvs/hailo-dfc3`).

## Build that produced the HEF

`dfc_flow.py parse --opt-level 1` (checker_cfg disabled), then `optimize`
(6 min), then `compile --compiler-effort 0 --no-width-split`.
Build dir `deploy/build_hailo8_edgedepth/`.

Results: split into 7 contexts in about 3 min (partitioner: 183 iterations,
3m 38s); mapping 8 min 10 s; kernels + HEF 24 s; total 8 min 49 s.
HEF 23.16 MB (23,159,143 bytes).

Bandwidth per frame (from `compile_e0nws.log`):

| stream | Mbps / frame |
|---|---|
| inputs | 20.75 |
| outputs | 3.46 |
| inter-context | 594.9 (shipped HEF: 930) |
| DDR | 0 |

Log verbatim:

```
[info] Bandwidth of model inputs: 20.7539 Mbps, outputs: 3.45898 Mbps (for a single frame)
[info] Bandwidth of DDR buffers: 0.0 Mbps (for a single frame)
[info] Bandwidth of inter context tensors: 594.906 Mbps (for a single frame)
[info] Successful Mapping (allocation time: 8m 10s)
[info] Successful Compilation (compilation time: 24s)
  wrote deploy/build_hailo8_edgedepth/EdgeDepth_hailo8_e0nws.hef (23.16 MB) in 520s
```

Exact commands:

```
export HAILO_VENV=$HOME/.venvs/hailo-dfc3
COMMON="--hw-arch hailo8 --onnx release/EdgeDepth/EdgeDepth.onnx --build-dir <B> --float-ref 0.987,1.067,4.53"
./deploy/hailo-py deploy/dfc_flow.py parse $COMMON --opt-level 1
echo "model_optimization_config(checker_cfg, policy=disabled)" >> <B>/hailo_stereo.resolved.alls
./deploy/hailo-py deploy/dfc_flow.py optimize $COMMON
./deploy/hailo-py deploy/dfc_flow.py compile $COMMON --compiler-effort 0 --no-width-split --hef <B>/EdgeDepth_hailo8_e0nws.hef
```

(Run long DFC steps under `script -q -c "..." /dev/null > log` so the output
flushes live. Keep the laptop plugged in.)

## int8 emulation (laptop, dfc_flow.py emulate, 40 val pairs, masked EPE / official EPE / official D1)

| context | masked EPE | official EPE | official D1 |
|---|---|---|---|
| fp_optimized | 0.987 | 1.067 | 5.06% |
| quantized (int8) | **1.240** | **1.329** | **6.32%** (+0.253 px masked) |

For reference, EdgeDepth original layout at opt-level 2: 1.196 / 1.282.
Shipped model int8: 1.333 / 1.430; on the Pi: 1.613 / 1.687 @ 3.95 FPS.
opt-level 1 was used because the opt-level 2 QFT ran out of memory on these graphs.

## Compiler warning to verify on the device

conv56_d2_sd0 (from /refine4/blocks/blocks.2/conv2/Conv) with element-wise addition:
"requires the output_shape of conv and of the add to be equal",
add [-1,92,308,32] vs conv [-1,92,308,3].
It does not appear in any earlier build.

Log verbatim:

```
[warning] conv layer conv56_d2_sd0 (translated from /refine4/blocks/blocks.2/conv2/Conv) with element-wise addition requires the output_shape of conv and of the add to be equal
add_output_shape='[-1, 92, 308, 32]', conv_output_shape=[-1, 92, 308, 3]
```

## History of the Hailo-8 compile attempts (why the rewrite was needed)

| variant | change | result |
|---|---|---|
| EdgeDepth original layout (254 Hailo layers) | none | e0 + no-width-split: context 6 hit the 1 h watchdog (09-25) |
| EdgeDepth-grouped (164 layers) | 24 branches fused into one grouped 1x1 conv | partitioned in 5 min; context 2 (grouped conv) failed mapping |
| EdgeDepth-prered (217 layers, 48 Slice) | reduce before shift | never partitioned (6 h 51 min, on battery) |
| EdgeDepth-dense (48 Slice) | prered + dense block-diagonal corr conv | never partitioned: 45 min (09-26); 2 h 06 min, 1,495 rejected partitions (10-06) |
| **EdgeDepth (24 Slice, 305 ONNX nodes)** | shift once; sub = dense conv(big_left - big_right) | e0: 7 contexts in 7.5 min, but context 3 hit the 1 h watchdog; **e0 + --no-width-split: HEF in 8 min 49 s** |

Root cause: Hailo-8 runs each context as a spatial pipeline. A cost volume with
48 parallel shift slices needs too many resources and inter-context tensors, so
the rejected partitions were "too many resources", "initial utilization
overflow" and "too many inputs/outputs".

EdgeDepth identity: reduce has no bias and the shift pads with zeros, so
shift(reduce(R)) == reduce(shift(R)) and reduce(L)-reduce(R_d) == reduce(L-R_d).
17.07 GMAC (dense 16.03, original 14.99).

## Per-context cluster utilization (from compile_e0nws.log, ANSI codes and \r stripped)

```
hailo_stereo_context_0 utilization:
+-----------+---------------------+---------------------+--------------------+
| Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
+-----------+---------------------+---------------------+--------------------+
| cluster_0 | 93.8%               | 26.6%               | 24.2%              |
| cluster_1 | 31.3%               | 10.9%               | 14.8%              |
| cluster_2 | 100%                | 28.1%               | 25.8%              |
| cluster_3 | 93.8%               | 23.4%               | 21.9%              |
| cluster_4 | 50%                 | 17.2%               | 18.8%              |
| cluster_5 | 25%                 | 6.3%                | 9.4%               |
| cluster_6 | 56.3%               | 28.1%               | 20.3%              |
| cluster_7 | 100%                | 46.9%               | 35.9%              |
+-----------+---------------------+---------------------+--------------------+
| Total     | 68.8%               | 23.4%               | 21.4%              |
+-----------+---------------------+---------------------+--------------------+

hailo_stereo_context_1 utilization:
+-----------+---------------------+---------------------+--------------------+
| Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
+-----------+---------------------+---------------------+--------------------+
| cluster_0 | 37.5%               | 9.4%                | 35.9%              |
| cluster_1 | 100%                | 39.1%               | 28.9%              |
| cluster_2 | 56.3%               | 20.3%               | 51.6%              |
| cluster_3 | 37.5%               | 25%                 | 35.2%              |
| cluster_4 | 87.5%               | 21.9%               | 24.2%              |
| cluster_5 | 100%                | 56.3%               | 57.8%              |
| cluster_6 | 100%                | 25%                 | 36.7%              |
| cluster_7 | 100%                | 25%                 | 28.1%              |
+-----------+---------------------+---------------------+--------------------+
| Total     | 77.3%               | 27.7%               | 37.3%              |
+-----------+---------------------+---------------------+--------------------+

hailo_stereo_context_2 utilization:
+-----------+---------------------+---------------------+--------------------+
| Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
+-----------+---------------------+---------------------+--------------------+
| cluster_0 | 25%                 | 7.8%                | 11.7%              |
| cluster_1 | 50%                 | 28.1%               | 27.3%              |
| cluster_2 | 62.5%               | 29.7%               | 36.7%              |
| cluster_3 | 56.3%               | 31.3%               | 32.8%              |
| cluster_4 | 43.8%               | 35.9%               | 34.4%              |
| cluster_5 | 31.3%               | 9.4%                | 13.3%              |
| cluster_6 | 87.5%               | 45.3%               | 45.3%              |
| cluster_7 | 50%                 | 17.2%               | 20.3%              |
+-----------+---------------------+---------------------+--------------------+
| Total     | 50.8%               | 25.6%               | 27.7%              |
+-----------+---------------------+---------------------+--------------------+

hailo_stereo_context_3 utilization:
+-----------+---------------------+---------------------+--------------------+
| Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
+-----------+---------------------+---------------------+--------------------+
| cluster_0 | 50%                 | 15.6%               | 40.6%              |
| cluster_1 | 50%                 | 21.9%               | 73.4%              |
| cluster_2 | 100%                | 32.8%               | 64.8%              |
| cluster_3 | 37.5%               | 14.1%               | 29.7%              |
| cluster_4 | 68.8%               | 26.6%               | 51.6%              |
| cluster_5 | 25%                 | 6.3%                | 21.9%              |
| cluster_6 | 100%                | 34.4%               | 68%                |
| cluster_7 | 68.8%               | 23.4%               | 39.1%              |
+-----------+---------------------+---------------------+--------------------+
| Total     | 62.5%               | 21.9%               | 48.6%              |
+-----------+---------------------+---------------------+--------------------+

hailo_stereo_context_4 utilization:
+-----------+---------------------+---------------------+--------------------+
| Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
+-----------+---------------------+---------------------+--------------------+
| cluster_0 | 87.5%               | 28.1%               | 29.7%              |
| cluster_1 | 100%                | 40.6%               | 38.3%              |
| cluster_2 | 43.8%               | 26.6%               | 32.8%              |
| cluster_3 | 50%                 | 18.8%               | 16.4%              |
| cluster_4 | 100%                | 81.3%               | 46.1%              |
| cluster_6 | 87.5%               | 28.1%               | 26.6%              |
+-----------+---------------------+---------------------+--------------------+
| Total     | 58.6%               | 27.9%               | 23.7%              |
+-----------+---------------------+---------------------+--------------------+

hailo_stereo_context_5 utilization:
+-----------+---------------------+---------------------+--------------------+
| Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
+-----------+---------------------+---------------------+--------------------+
| cluster_0 | 75%                 | 26.6%               | 44.5%              |
| cluster_1 | 12.5%               | 4.7%                | 7%                 |
| cluster_2 | 100%                | 35.9%               | 60.2%              |
| cluster_3 | 37.5%               | 10.9%               | 17.2%              |
| cluster_4 | 25%                 | 6.3%                | 9.4%               |
| cluster_5 | 100%                | 29.7%               | 26.6%              |
| cluster_6 | 100%                | 34.4%               | 65.6%              |
| cluster_7 | 93.8%               | 34.4%               | 51.6%              |
+-----------+---------------------+---------------------+--------------------+
| Total     | 68%                 | 22.9%               | 35.3%              |
+-----------+---------------------+---------------------+--------------------+

hailo_stereo_context_6 utilization:
+-----------+---------------------+---------------------+--------------------+
| Cluster   | Control Utilization | Compute Utilization | Memory Utilization |
+-----------+---------------------+---------------------+--------------------+
| cluster_0 | 81.3%               | 32.8%               | 46.1%              |
| cluster_1 | 37.5%               | 18.8%               | 25.8%              |
| cluster_2 | 31.3%               | 17.2%               | 28.9%              |
| cluster_3 | 100%                | 39.1%               | 50%                |
| cluster_4 | 81.3%               | 43.8%               | 69.5%              |
| cluster_5 | 75%                 | 51.6%               | 53.9%              |
| cluster_6 | 50%                 | 18.8%               | 32%                |
| cluster_7 | 56.3%               | 20.3%               | 32%                |
+-----------+---------------------+---------------------+--------------------+
| Total     | 64.1%               | 30.3%               | 42.3%              |
+-----------+---------------------+---------------------+--------------------+
```

## Next

An on-device test on the Pi 5 (deploy/pi/: run_hef.py, score_preds.py;
40 pairs) against the shipped HEF; check the refine4 warning.

## Files in this folder

- `EdgeDepth_hailo8.hef` — copy of
  `deploy/build_hailo8_edgedepth/EdgeDepth_hailo8_e0nws.hef`
  (sha256 `96fd7441b8a8193306f899b81c116831a8edd562c9e6f11d21f74c919f575069`)
- `emulator_results.json` — copy of the int8 emulation scores
- `hailo_stereo.resolved.alls` — parse + optimization script used
- `hailo_stereo.compile.alls` — allocator settings used
  (`compiler_optimization_level=0`, `width_splitter_defuse=disabled`)
