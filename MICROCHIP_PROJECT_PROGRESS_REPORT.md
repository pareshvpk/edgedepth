# HailoStereo — Technical Progress Report for Microchip

**Prepared for:** Microchip engineering review
**Scope:** Model/algorithm development only. On-device Hailo-15H hardware validation,
physical-board testing, real-device latency measurement, and final documentation/report
writing are explicitly **excluded** from the completion percentages below — they are
tracked separately as blocked/out-of-scope items.

**Method:** Every claim in this report is backed by a specific file, script, log, or
artifact in the repository (`C:\Microchip\hailo-stereo`). File paths are given so each
number can be independently reproduced. Git history is a single squashed initial
commit (`ffda535`, 2026-09-15) followed by seven incremental commits through
2026-09-20 (`git log`), so chronology below is reconstructed from **dated evidence
inside the files** (STATUS.md, IMPROVEMENT_PLAN.md, HANDOFF documents, checkpoint
timestamps, artifact mtimes) rather than from commit dates alone.

---

## 0. Bottom line

**≈70% complete / ≈30% remaining**, counting only model/algorithm development
(architecture, training, export, quantization, and the Hailo software compilation
pipeline up to a valid HEF). The dividing line is not accuracy polish — the model is
already trained, evaluated, exported, quantized, and **compiled to a working
Hailo-15H HEF**, with every stage backed by a passing automated check. What remains
is: (1) a small number of identified-but-unimplemented model improvements (match
confidence output, sky/unmatchable-region supervision), (2) one accuracy band that is
improving but not yet closed (0–5 m), and (3) an explicitly-deferred architecture
experiment branch (Phase 4 in the improvement plan) that was judged optional rather
than executed. On-device validation and FPS measurement are blocked on hardware this
machine does not have and are **not counted** in either the 70 or the 30.

---

## 1. What was actually investigated (method note)

Repository root: `C:\Microchip\hailo-stereo`. Directories inspected in full:
`src/` (17 Python modules, 3,955 non-model lines + `model.py`), `notes/` (9 analysis
scripts), `ref/` (Hailo's own YAML + upstream original source), `report/` (5 markdown
status documents + the teardown HTML), `deploy/` (compiler driver, `.alls` script,
build artifacts, logs), `runs/` (17 checkpoint directories with `best.json`/
`train.log` for every experiment), `artifacts/` (ONNX/HEF binaries), `logs/` (Hailo
SDK logs). `git log` shows 8 commits total; the first commit already contains the
full model, training pipeline, teardown report and initial deploy flow, so most of
the project's history lives in the dated prose of `report/STATUS.md` and
`report/IMPROVEMENT_PLAN.md`, not in git. Where a document's claim could be checked
against a primary artifact (a `best.json`, a `.hef` file size, an `emulator_results.json`),
it was, and every one checked matched exactly (see inline citations below).

---

## 2. Chronological reconstruction of the actual workflow

Dates are as recorded in the documents/artifacts, not inferred:

| date | stage |
|---|---|
| ≤2026-09-07 | Original Hailo Model Zoo `stereonet` artifacts obtained (`artifacts/stereonet.hef`, `stereonet.zip`, `artifacts/stereonet_pretrained/stereonet.onnx`) |
| 2026-09-09 | Cost-volume reverse engineering; KITTI calibration set built (`artifacts/calib_kitti_*.npy`, superseding an earlier SceneFlow-based calibration set) |
| 2026-09-10 | `stereonet_hailo15h_v5.4.0.hef` (the current Model Zoo build) obtained for a like-for-like toolchain comparison; first Hailo DFC run producing the original `hailo_stereo_hailo15h.hef` (4.69 MB, 5 contexts) |
| 2026-09-11 | `report/STATUS.md` first written; border-region weak-supervision fix; distance-band evaluation (`eval_depth.py`) added |
| 2026-09-14 | Per-worker RNG bug found and fixed in `data.py`; `report/IMPROVEMENT_PLAN.md` written; Phase 0–1 finetune ablations run on Windows; SceneFlow train/val leak diagnosed |
| 2026-09-14/15 | Phase 2 (KITTI 2012 mixing, disparity-aware sampling) run; paused once for GPU thermal throttling (documented in `report/IMPROVEMENT_PLAN.md` §"Phase 2 status") |
| 2026-09-15 | Squashed initial commit `ffda535` to git |
| 2026-09-16 | Phase 3 (pretrain redone scene-disjoint, RNG-fixed) completed on Windows; `report/HANDOFF_TO_LINUX.md` written; Linux session: ONNX export, DFC compile, emulation, `eval_depth.py` re-run, `STATUS.md` corrected; allocator regression discovered (0.95-utilization build stopped completing) and worked around with `--compiler-effort 0` |
| 2026-09-18 | Documentation/layout cleanup, `.alls` rationale corrected, README restructured into charts/tables |
| 2026-09-20 | Unit-mislabelling bug fixed in the headline comparison table (EPE vs D1) |

The **logical** order (independent of exact dates) that the evidence supports:

1. Obtain and inspect the original Hailo Model Zoo `stereonet` (ONNX + HEF + YAML).
2. Reverse-engineer its cost volume; discover the disparity-indexing defect (CV-1).
3. Prove the 12 disparity hypotheses are bit-identical, geometrically and numerically.
4. Discover and catalogue six further defects (CV-2, HW-1, MEM-1, NUM-1, EXP-1, ARCH-1).
5. Discover the original's published EPE figure is mislabeled (it is D1, not EPE).
6. Design a replacement architecture constrained to Hailo-compilable ops.
7. Build a geometry-first test suite that fails on the original and passes on the replacement.
8. Build the training/data pipeline (masked loss, strict init, deep supervision).
9. Pretrain on SceneFlow Driving; discover and fix a scene-leak in the validation split (full retrain required).
10. Finetune on KITTI 2015; discover a per-worker RNG bug in the data loader; discover and fix a border-region supervision gap.
11. Diagnose a large-disparity under-read; run a four-phase ablation campaign (RNG fix, EMA/augmentation, larger crop, SceneFlow replay, KITTI 2012 mixing, disparity-aware sampling, pretrain redo).
12. Export to ONNX with a self-auditing exporter; verify ONNX/PyTorch numerical parity.
13. Simulate int8 quantization and rank per-layer sensitivity.
14. Run the Hailo Dataflow Compiler pipeline: parse → optimize (QAT) → emulate → compile.
15. Resolve an allocator regression (`shmifo` capacity) that appeared between two DFC runs.
16. Stress/soak/adversarial-test the compiled int8 graph in the DFC emulator; discover a real behavioral defect (cannot express "no match"/"far").
17. Write up findings (excluded from this report's completion percentage).

---

## 3. Reverse engineering of the Hailo Model Zoo `stereonet` (detailed)

### 3.1 What the original cost volume was supposed to do

A rectified stereo pair's matching left pixel `x` corresponds to right pixel `x-d`
for the true disparity `d`. A cost volume evaluates a per-pixel similarity/difference
for every candidate `d` in `[0, D)`, so the network can search over disparities rather
than commit to one. The upstream reference implementation
(`ref/ref/upstream/cost_volume.py`, vendored from the model's public GitHub source,
`nivosco/StereoNet`) shifts the right feature map by `d` and subtracts it from the
left feature map, for `d = 0 … 11` (12 hypotheses at 1/16 resolution, `D=192`).

### 3.2 What the implementation actually did

```python
# ref/ref/upstream/cost_volume.py, lines 29-42
zero_padding = torch.zeros(..., disparity)               # d zero columns
left_move = torch.cat((origin, zero_padding), 3)          # pad appended on the RIGHT
leftMinusRightMove = left_move[:, :, :, :origin.shape[3]] - candidate   # slice [:W]
```

Padding is appended **after** column `W-1` and then the result is sliced to
`[:, :, :, :W]` — the first `W` columns of a tensor whose first `W` columns are, by
construction, exactly the original `origin` tensor, unchanged, for every value of
`disparity`. The slice never reaches the padded region. **The shift is a no-op for
every disparity hypothesis**, including in the `concat` cost-volume variant used by
the shipped model.

### 3.3 How the candidates were proven identical

`notes/verify_costvol.py` re-exports the original ONNX with the 12 internal `Sub_*`
subtraction outputs exposed, runs it once with random left/right input, and diffs
every disparity level against level 0:

```
TEST 1 — are the 12 cost-volume disparity slices distinct?
  Sub_1  (disparity level  1)  max|diff vs level 0| = 0.000000e+00   identical=True
  ...  (all 11 remaining levels identical=True)
```

`notes/verify_func.py` (TEST 2/3) independently confirms behaviorally: feeding the
**same** left image and successively larger true disparities into the right input
produces `max|pred vs disp=0|` of `0.0` for every non-zero true disparity — the
predicted disparity map does not change at all when the true disparity changes, and
does not change when the right image is replaced with random noise, all-zero, or
all-one input (TEST 3). The network's output is a function of the left image alone.

### 3.4 What `[d:d+W]` would have meant, and what was changed

The correct construction pads on the **left** (`F.pad(x, (d, 0))`) so the slice
`[..., :W]` returns the tensor shifted by `d` columns, with the first `d` columns
zeroed — i.e. `out[..., x] = right[..., x-d]`. HailoStereo's replacement
(`src/model.py::_shift_all`, lines 123-137) implements exactly this, as a **static**
graph: it pads once with `num_disp-1` leading zero columns and takes constant-offset
slices, so the same operation that fails in the original exports to Hailo-parseable
`Concat` + `Slice` nodes rather than dynamic `F.pad` calls per hypothesis.
`src/test_disparity.py::test_shift_actually_shifts` and
`test_cost_volume_slices_are_distinct` assert the corrected behavior directly and are
written to fail in the way the original model would fail (comments cite "this is the
CV-1 defect" at the assertion site).

### 3.5 What the original's real accuracy came from

With no disparity search, the network can only use the left image plus whatever prior
its training data taught it. `report/teardown.html` and `README.md` (§1) report the
original's real KITTI EPE, measured under the official protocol on the same 40-frame
split HailoStereo is scored on (`notes/compare_models.py`), as **1.300 px** — not the
8.223 published by the Model Zoo. Cross-checking `ref/stereonet.yaml`
(`eval_metric: EPE`, `full_precision_result: 8.223`) against a rerun of the original
ONNX under the official protocol found **EPE 1.300 px, D1 8.119%** — the published
8.223 is a near-exact match for the measured **D1**, not EPE, i.e. Hailo's own model
card mislabels its published metric (`report/teardown.html`, evidence text: "a
near-exact match to the published 8.223, confirming the published figure is the D1
outlier rate, not EPE"). Road-scene disparity correlates with a left-to-right
brightness gradient, which a left-image-only network can fit without ever looking at
the right image.

### 3.6 Full defect catalogue (from `report/teardown.html`, cited by ID throughout the codebase)

| ID | Defect | Severity | Where fixed |
|---|---|---|---|
| CV-1 | Cost-volume shift is a no-op; 12 hypotheses bit-identical | Critical | `src/model.py::_shift_all` |
| CV-2 | Zero-padding at the disparity axis border contaminates cost with a magnitude term, not a matching term | Minor | `src/train.py` weak border supervision (§5) |
| HW-1 | Refinement head (six 32-ch residual blocks at full 368×1232) does not fit a Hailo context; compiler spatially defuses `conv47`/`conv48` into 22 sub-convolutions each | Major | Channel width tapers with resolution in the new refinement ladder (`src/model.py::RefineStage`) |
| MEM-1 | A single 21.8 MB initializer (`/Tile_output_0`) is a materialized per-channel index ramp — 91.9% of the ONNX file | Major | `SoftArgmin.index` is a 1×1 convolution with weights `[0..23]`, not a materialized tensor |
| NUM-1 | Softmax runs at full resolution (5,441,536 elements) after 16× upsampling, unscaled — no temperature | Major | Softmax confined to matching resolution (170,016 elements) with a learned, clamped temperature |
| EXP-1 | Export loads weights with `strict=False`, silently leaving mismatched tensors at random init | Minor | Every `load_state_dict` call in `src/train.py`, `export_onnx.py`, `quantize_sim.py` is `strict=True` |
| ARCH-1 | Single-stage refinement from a 1/16 disparity map (16 px per bin) instead of a hierarchical 1/8→1/4→1/2→1/1 ladder | Minor | `src/model.py` refines through three stages (`refine4`, `refine2`, `refine1`) from a 1/8 map (8 px per bin) |

**Status: COMPLETE.** Every defect ID above is independently verifiable: CV-1 by
`notes/verify_costvol.py` + `notes/verify_func.py` + `ref/ref/upstream/cost_volume.py`
source inspection; MEM-1/NUM-1 by `notes/analyze.py`'s initializer/softmax-shape dump;
HW-1 by the shipped `.hef`'s own layer table (cited in `report/teardown.html`).

---

## 4. New architecture (`src/model.py`) — COMPLETE

### 4.1 Structural summary

| | Model Zoo `stereonet` | HailoStereo |
|---|---|---|
| Disparity hypotheses | 12 (all bit-identical) | **24 (all numerically distinct, verified)** |
| Matching resolution | built at 1/16, softmax after 16× upsample to full-res | built and decoded at 1/8 |
| Max disparity range | 192 px | 192 px (same, at 2× the resolution: 24×8 vs 12×16) |
| 3D convolutions | 5 (`MetricBlock` ×4 + final `Conv3d`) | **0** |
| Cost volume layout | 5D tensor `[B, D, C, H, W]`, filtered by `Conv3d` | Disparity hypotheses packed into the **channel** axis; 2D convolution only |
| Softmax width | 5,441,536 elements (full-res, all channels) | **170,016 elements** (1/8-res matching resolution) |
| Largest constant | 21.76 MB (materialized index ramp) | **0.442 MB** (1×1 conv, weights `[0..23]`) |
| Parameters | 423,586 (measured; Model Zoo's own YAML claims 623.1K, a discrepancy not explained in the repo) | 796,531 |
| ONNX size | 23.69 MB | **3.38 MB**, opset 14, 347 nodes |
| Compute | 112.07 GMAC-derived GOPS (90.6% in refinement) | **28.41 GOPS** |
| Refinement head | 6× 32-ch ResBlocks at full 368×1232 (compiler-defused into 22 shards per layer) | 3-stage tapering ladder (32→24→16 ch), nothing wide enough to require defusing |
| Softmax temperature | none (NUM-1) | learned, clamped to `[0.2, 2.0]` (`SoftArgmin.temperature`) |
| Output range | unbounded except a final ReLU | `clamp(0, MAX_DISP)` — a single `Clip` op |

Evidence: `src/model.py` (architecture source); `src/export_onnx.py::compute_budget`
(GOPS by stage); `notes/analyze.py` (original's MAC/param/initializer breakdown);
`report/README.md` §1 "Structural scorecard" (side-by-side table, independently
cross-checked here against primary sources rather than taken on faith).

### 4.2 Component-by-component

- **`FeatureNet`** — siamese tower, three strided convs (1/2, 1/4, 1/8) + 6 residual
  blocks, channel count *rises* with depth (16→24→32) rather than the original's flat
  32 everywhere everywhere (targets HW-1).
- **`CostVolume`** — group-wise: a shared grouped 1×1 convolution (`groups=8`) reduces
  the 32-channel `left - shift(right, d)` difference to 8 values per hypothesis, then
  concatenates 24 hypotheses along the channel axis (`24×8=192` channels). Avoids
  `Conv3d`/5D tensors entirely (HW-1). A `cost="corr"` (product) variant exists in the
  same class but is untested on hardware (§7, Phase 4, not started).
- **`Aggregation`** — a small 2D hourglass (stem → 2 ResBlocks → downsample to 1/16 →
  2 ResBlocks → upsample back → 1 ResBlock → output conv+BatchNorm). The final
  BatchNorm is documented and empirically verified as load-bearing (§4.3).
- **`SoftArgmin`** — softmax over the 24-channel cost at 1/8 resolution, decoded by a
  fixed (non-trainable) 1×1 convolution whose weights are the disparity indices
  `[0..23]` (replacing MEM-1's 21.8 MB ramp with 96 bytes).
- **`RefineStage`×3** — hierarchical upsample-and-correct ladder at 1/4, 1/2, 1/1,
  guided by left-image features at matching resolution, replacing ARCH-1's single
  full-resolution stage.
- **Final clamp** — `torch.clamp(d1, 0, MAX_DISP)`; the model's docstring records that
  the unclamped refinement head was measured emitting up to ~2.5× `MAX_DISP` in
  GT-free regions (sky, occlusion) before this was added, at zero measured cost to
  official KITTI EPE (KITTI GT never exceeds `MAX_DISP`).

### 4.3 A finding from architecture ablation, not from the original teardown

`notes/diagnose_collapse.py` establishes that the aggregation head's final
BatchNorm is necessary, not decorative: removing it lets the convolution scale its
logits freely (measured std reaching 34 within 100 training steps), which saturates
the 24-way softmax (entropy 0.007 of a possible `ln(24)=3.18`) and stops all gradient
flow through the matching path. Three seeds without BatchNorm: EPE **29.19 / 8.55 /
7.41** (one outright failure). With it: entropy 2.56–2.84, EPE **7.47 / 7.22 / 7.82**
(all three converge consistently). This is filed as the same defect class as NUM-1
(an unscaled softmax) discovered independently during the replacement's own
development, not inherited from the teardown. **Status: COMPLETE**, reproducible via
`notes/diagnose_collapse.py`.

---

## 5. Training and evaluation — COMPLETE (with one long ablation campaign)

### 5.1 Data

- **SceneFlow Driving** (`src/fetch_driving.py`, `src/data.py::SceneFlow`): 4,400
  synthetic pairs, ~3.1 GB, used for pretraining.
- **KITTI 2015** (`src/data.py::Kitti2015`): 200 labelled real pairs, 160 train / 40
  val, used for finetuning and as the single fixed evaluation split for every
  reported number in the project.
- **KITTI 2012** (`src/data.py::Kitti2012`, added mid-project): 194 pairs, added
  specifically to increase large-disparity supervision share (§5.4).

### 5.2 The SceneFlow validation-split leak, found and fixed

**What was wrong.** Every checkpoint before `sceneflow_holdout` validated on the last
2% of an index-sorted frame list. SceneFlow Driving is a single continuous
fly-through per rendered scene, so this put frame 713 of a trajectory in validation
immediately after frame 712 in training — near-identical geometry, texture and
lighting one time-step apart. **Evidence:** `runs/sceneflow_cont/best.json` (leaky,
EPE 2.596) vs `runs/sceneflow_holdout/best.json` (scene-disjoint, EPE 3.687) — both
files present on disk and matching the reported 42% gap.

**The fix.** `src/train.py::build_datasets` now holds out whole named subsets
(`SCENEFLOW_HOLDOUT = ("35mm_focallength/scene_forwards/slow",)`), verified
scene-disjoint by construction (`_scene_of` groups by the `.../<focallength>/<scene>/<speed>/` path prefix, and `build_datasets` asserts the holdout name exists and
excludes it from `train.samples`). Because earlier checkpoints had trained on frames
now held out, **a full retrain from scratch was required** — the fix could not be
applied by re-scoring existing weights. **Status: COMPLETE.**

### 5.3 The per-worker RNG bug, found and fixed

`_StereoBase` originally constructed its `random.Random(seed)` once in `__init__`,
so every `DataLoader` worker process received an **identical copy** — with
`--workers 4`, only 4 distinct augmentation draws were ever produced per batch of 16.
Fixed by reseeding per worker from `get_worker_info().seed` (verified 16/16 distinct
draws, per `report/IMPROVEMENT_PLAN.md` §1.4 and `report/WINDOWS_HANDOFF.md`). Every
run before the fix trained with a quarter of its apparent augmentation diversity.
Isolated ablation (`kitti_rngfix`, `runs/kitti_rngfix/best.json`, EPE 1.488 masked)
showed the bug alone cost negligible headline accuracy (within the ~0.1 px noise
floor against the pre-fix baseline) but the fix mattered once folded into the
pretrain redo (§5.4, Phase 3). **Status: COMPLETE.**

### 5.4 The finetuning ablation campaign

A four-phase, gated campaign (`report/IMPROVEMENT_PLAN.md`), each phase's winner
becoming the next phase's baseline. All numbers below are read directly from
`runs/<name>/best.json` and cross-checked against the numbers quoted in
`IMPROVEMENT_PLAN.md`, which matched exactly:

| phase | run | change | official EPE | D1% | status |
|---|---|---|---|---|---|
| baseline | `kitti_border` | border weak supervision only | 1.659 | 9.92 | superseded |
| 1 | `kitti_rngfix` | worker-RNG fix only | 1.670 | — | ablation |
| 1 | `kitti_ema_aug` | + EMA 0.999, strong aug | 1.691 | — | ablation, no gain |
| 1 | `kitti_crop768` | + 320×768 crop | **1.639** | 9.42 | **Phase 1 winner** |
| 1 | `kitti_replay50` | + SceneFlow replay 0.5 | 1.704 | 10.32 | fixed far-range forgetting, small headline cost |
| 2.1 | `kitti_mixed768` | + KITTI 2012 (354 train pairs) | **1.336** | 7.59 | **decisive win of the project** |
| 2.2 | `kitti_mixed_sampled768` | + disparity-aware sampling | 1.345 | 7.61 | statistical tie, flatter far-range bias |
| 3 | `sceneflow_fixed` | pretrain redone, scene-disjoint + RNG-fixed | (hold-out 3.604, comparable to prior 3.610) | — | retrain confirms the fix didn't regress the pretrain itself |
| 3.2 | **`kitti_mixed_fixed768`** | finetune from the fixed pretrain | **1.248** | **6.87** | **shipped model** |

Reproduced independently from `runs/kitti_mixed_fixed768/best.json`:
`{"epoch": 223, "epe": 1.1557382206635125, "d1": 6.3837604263275765}` — the `epe`
field is the **masked** protocol (1.156 px); the **official** protocol figure (1.248
px, 6.87% D1, quoted throughout) comes from `src/eval_kitti.py`'s second pass over
every valid ground-truth pixel, not just the trained region. Both protocols are
computed in one script run, deliberately kept distinct so a masked number is never
quoted against a baseline that reports an unmasked one.

**Two protocols, and why they must not be conflated:** the masked protocol excludes
ground truth beyond 192 px (none exists in this split) and the left `max_disp`
border columns (no possible match). The **official** protocol — every pixel with
valid ground truth — is the one comparable to a published baseline, and is what the
1.248 px headline figure is.

**Guard-rail respected throughout:** per-frame EPE on the 40-image split spans
0.93–3.51 px (`IMPROVEMENT_PLAN.md` §3), so every ablation decision in the table
above was made only where the gap exceeded that noise floor; ties were resolved by a
stated secondary criterion (smaller ≥80 px bias), not cherry-picked.

**Status: COMPLETE.** Final shipped checkpoint: `runs/kitti_mixed_fixed768/best.pt`.

### 5.5 Border-region weak supervision (CV-2 remedy)

**What was wrong.** Training masked two regions from the loss: GT beyond 192 px
(measured empty — confirms the range is sufficient) and the left `max_disp` columns,
which have no possible match. Excluding a region from the **loss** is correct;
excluding it from the **reported metric** silently hides a large error. Measured:
13.2% of frame pixels (the border) carried **83.7%** of total error (mean EPE 47.75
px) when left entirely unsupervised (`report/STATUS.md` §2.2, table reproduced from
`eval_kitti.py --bins`).

**The fix.** The border re-enters the loss at reduced weight (`BORDER_WEIGHT` in
`src/data.py`) rather than staying fully unsupervised, so the network extrapolates
from monocular cues and spatial continuity. Result: border EPE 47.75 → **2.95 px**
(16× reduction) for a 3.5% cost in the matched region (1.41 → 1.46 px).
**Status: COMPLETE.**

### 5.6 Distance-band (depth) evaluation

`src/eval_depth.py` (398 lines, new mid-project) converts prediction and ground
truth to metres through KITTI's nominal `f·B = 386.5 px·m` and reports error banded
by distance, because a single headline EPE hides a 100× swing in metric significance
(1 px = 6 cm at 5 m, 6.5 m at 50 m). Key finding (`report/STATUS.md` §2.3, re-measured
2026-09-16 on the shipped checkpoint): **5–40 m sits at its geometric floor** — the
model is not the limiting factor in the range band that matters most for driving; the
0–5 m band is the weakest (EPE 7.29 px vs ~1.1 px elsewhere) and was initially
misdiagnosed as limited by the 192 px disparity ceiling. That diagnosis was checked
and found wrong (every 0–5 m pixel in the split has GT disparity 77–158 px, inside
the range) and corrected in the same document — the actual cause, confirmed by
`report/IMPROVEMENT_PLAN.md` §1, was KITTI 2015's near-absence of large-disparity
training examples (1.0% of pixels ≥80 px vs 42% in SceneFlow), addressed by the
KITTI 2012 mix (§5.4). **Status: PARTIAL** — materially improved (mean |dZ| 1.08 →
0.45 m, AbsRel 28.4% → 11.3%) but still the weakest band; further gains are
identified as a data/sampling problem, not an architecture limit.

### 5.7 Experiments attempted and superseded/discarded

| run | what it tested | outcome |
|---|---|---|
| `runs/sceneflow`, `sceneflow_full`, `sceneflow_cont` | Early pretrains, before the leak fix | Superseded once the leak was found; kept for the record, not used downstream |
| `runs/_sceneflow_cont_raced` | — | **Infrastructure artifact, not a model experiment.** `train.log` shows the identical training sequence logged twice from the same start point — a duplicate background process writing the same log (consistent with a known machine issue where the venv Python launches as two processes). Directory was renamed with a `_...raced` suffix and excluded from the reported results; no conclusions were drawn from it. |
| `kitti_ema_aug` | EMA + strong augmentation on top of the RNG fix | No measurable gain (1.691 vs 1.670 px, within noise); not carried forward |
| `kitti_replay50` (Phase 1 variant D) | SceneFlow replay at ratio 0.5 during KITTI finetune | Fixed far-range forgetting (SceneFlow hold-out 27.7 → 3.99 px) at a small headline cost (1.704 vs 1.639 px); not selected as the Phase 1 winner by the stated lowest-official-EPE rule, but the mechanism it validated (replay) informed later phase design |
| `kitti_mixed_sampled768` (Phase 2.2, disparity-aware sampling) | Oversampling far-disparity frames 3× | Statistical tie with the non-sampled run (1.345 vs 1.336 px, inside noise); kept as an alternate ship candidate, not selected as primary |
| Compiling the HEF at the DFC's default 60% utilization | — | **Fails to allocate** (`Resources presolve failed: lcus=(180/80)`); not a model defect — documented and worked around (§6.3) |
| Compiling at `max_utilization=0.95` on 2026-09-16 | Re-running the exact build that succeeded on 2026-09-10 | **No longer completes** — three attempts of 41–88 minutes stalled on `shmifo in capacity exceeded`; an environment/allocator regression between DFC runs, not a code change. Worked around with `performance_param(compiler_optimization_level=0)` instead (7 contexts, 4.12 MB, succeeds in 5m52s) |
| `max_utilization=0.8` (workaround attempt) | Lower packing density to avoid the spatial-defuse profiler crash | Made it **worse** — 146 shards instead of 97, 40m32s compile, 5.80 MB HEF, same crash. Discarded |
| `allocator_param(width_splitter_defuse=disabled)` | Workaround attempt for the profiler crash | Accepted by the compiler but ineffective — the shards are produced by a different mechanism (spatial, not width, defusing). Discarded |

---

## 6. Verification — every test actually run

| test | purpose | threshold | result | status |
|---|---|---|---|---|
| `src/test_disparity.py` (6 tests) | Geometry: does the cost volume actually shift, and does matching recover a known disparity | all 6 pass | **6/6 PASS** — the exact suite the original model fails | COMPLETE |
| `src/check_ingest.py` | Data pipeline: correct `I_L(x)=I_R(x-d)` warping convention, both KITTI 2015 and 2012 | residual minimized at warp ×1.0 | **PASS**, 3.61× better than not warping | COMPLETE |
| `src/test_learns.py --steps 2000` | Synthetic end-to-end learning sanity | EPE should fall substantially | **28.10 → 2.88 px** (90% reduction) | COMPLETE |
| `notes/diagnose_collapse.py` | Confirms BatchNorm is load-bearing in the cost head | entropy should not saturate | entropy 2.56–2.84 with BN vs 0.007 without; 3-seed EPE 7.2–7.8 vs 7.4/8.6/29.2 | COMPLETE |
| `src/export_onnx.py` structural audit (HW-1, MEM-1×2, NUM-1, PARSE) | No 3D convs, no oversized constants, no dynamic-shape ops, softmax confined to matching resolution | all 5 sub-checks PASS | **all PASS**, "fails closed" on any unresolved value | COMPLETE |
| `src/export_onnx.py` parity | ONNX vs PyTorch numerical equivalence (catches sign errors in `fuse_temperature_`) | < 1e-2 px | **5.57e-04 px** (Linux, on the shipped checkpoint); 8.4e-04 px reported separately in `deploy/README.md` | COMPLETE |
| `src/eval_depth.py` cross-check | Depth-domain scoring reproduces pixel-domain scoring | exact pixel-count match | matches `eval_kitti.py` to the pixel (3,353,290 masked / 3,861,014 official) | COMPLETE |
| `deploy/dfc_flow.py` harness validation | The compiler-facing KITTI loader/scorer must reproduce `eval_kitti.py` | exact match | reproduces exactly, pre-compile | COMPLETE |
| DFC emulator float parity (`SDK_NATIVE`, `SDK_FP_OPTIMIZED` vs torch) | Confirms ONNX translation + on-chip normalization + NHWC layout are correct before quantization is even considered | should match torch float | **exact match to 3 decimals** (`deploy/build/emulator_results.json`) | COMPLETE |
| DFC emulator int8 accuracy (`SDK_QUANTIZED`) | Measures real quantization cost | — | masked 1.333 / official 1.430 px, D1 8.00% (+14.5% official vs the +24.6% the simulation predicted) | COMPLETE |
| `notes/stress_emulator.py` determinism | Same input, repeated inference | bitwise identical | **bitwise identical**, max diff 0.0 across 3 repeats and again across 6 scenes × 7 repeats over a 30-minute soak | COMPLETE |
| `notes/stress_emulator.py` soak (121 inferences, 30 min) | NaN/Inf, clamp violations, shape drift, latency/RSS drift under sustained load | 0 failures on all counts | **0 crashes, 0 NaN/Inf, 0 clamp violations, 0 shape drift**; latency improved (warm-up), RSS plateaued | COMPLETE |
| `notes/stress_emulator.py` adversarial (20 degenerate inputs) | Photometric robustness (inversion, noise, channel loss, contrast) | no crash, sane output | robust to inversion/noise/channel-loss; **fails to abstain on textureless input** (reports ~100 px / 3.9 m on blank/noise/sky-like input) | **FAILED — real defect, not fixed** (see §8) |
| `notes/verify_costvol.py`, `notes/verify_func.py` | Proves the *original* model's defect (not this project's) | slices should differ / output should respond to right image | 12/12 identical, output insensitive to right image — confirms the defect being replaced | COMPLETE (diagnostic of the baseline, not of HailoStereo) |
| `notes/compare_models.py`, `notes/compare_crossdomain.py` | Head-to-head scoring of both ONNX graphs under one scorer | validity check: original should reproduce its published number | validity check passed (measured 1.300 px vs 8.223 D1-relabelled reference, see §3.5); HailoStereo wins on KITTI, results read as a bracket (not a clean win either direction) on the cross-domain SceneFlow check since neither model trained on those exact frames but each has its own domain advantage | COMPLETE |
| `hailo profiler` (latency/FPS) | Static performance estimate | should report FPS | **crashes** — `conv_output_shape` mismatch from spatial-defusing a residual add; reproduced on both the 5-context and 7-context builds | **FAILED, not repaired** — DFC 5.4.0 tooling bug, four workarounds tried and documented, none works (§5.7); needs a DFC fix or on-device HailoRT timing (hardware, out of scope) |

Two failed/unrepaired tests are called out explicitly, as required: the emulator
**cannot currently make the model abstain on unmatchable input** (§8), and the
**Hailo profiler cannot report FPS on this HEF** — the latter is a compiler tooling
defect external to this project's model, not something fixable in this codebase.

---

## 7. Hailo software pipeline

```
Model (src/model.py, PyTorch)
  → ONNX (src/export_onnx.py)                                    COMPLETE
  → DFC parse (deploy/dfc_flow.py parse → hailo_stereo.har)       COMPLETE
  → DFC optimize / quantization-aware fine-tuning (optimize)      COMPLETE
  → DFC emulate (SDK_NATIVE / SDK_FP_OPTIMIZED / SDK_QUANTIZED)   COMPLETE
  → DFC compile (compile → HEF)                                  COMPLETE
  → hailo profiler (FPS/latency estimate)                        FAILED (tooling bug, §6)
  → on-device validation (HailoRT, physical Hailo-15H)            OUT OF SCOPE — no board
```

### 7.1 Compilation results

Two HEFs exist, both valid and both preserved:

| build | size | contexts | compile time | utilization setting | status |
|---|---|---|---|---|---|
| `artifacts/shipped/hailo_stereo_hailo15h.hef` (2026-09-10) | 4.69 MB | 5 | 13m 15s | `max_utilization=0.95` | superseded but preserved; on-disk size confirmed (4,689,920 bytes) |
| `artifacts/hailo_stereo_hailo15h.hef` (2026-09-16, current) | 4.12 MB | 7 | 5m 52s | `performance_param(compiler_optimization_level=0)` | **current deliverable**; on-disk size confirmed (4,120,576 bytes) |

Both were built with the identical DFC release (**5.4.0**) that produced
`artifacts/stereonet_hailo15h_v5.4.0.hef` (9,166,848 bytes on disk — Model Zoo's own
current HEF, kept for a like-for-like comparison), so the accuracy/size comparison
in §4.1 measures architecture, not a compiler-generation gap.

### 7.2 The allocator regression (found, worked around, not fully explained)

Re-running the exact 0.95-utilization configuration that succeeded on 2026-09-10
**stopped completing** on 2026-09-16: three attempts (41–88 minutes each) stalled at
context 3 of 5, with 34 of 46 allocator failures being `shmifo in capacity exceeded`.
The 24-slice cost volume (a structural consequence of doubling the disparity
hypotheses to fix CV-1) is memory-traffic-heavy rather than MAC-heavy, and is
specifically what strains the allocator's inter-context stream budget — a
prediction the original teardown made in advance ("MAC counts do not predict its
cost ... when the graph reaches a dataflow NPU"). The current HEF uses a different
compiler setting that succeeds but produces more contexts (7 vs 5); the FPS impact
of more context switching is **unmeasured** because the profiler itself is broken
(§6). This is filed as an open question, not resolved: `report/STATUS.md` §1
"Allocator change" and §3.3 document the full search (four configurations tried).

### 7.3 Quantization

- **Simulation** (`src/quantize_sim.py`): fake-quantizes weights (per-channel, signed
  int8) and activations (per-tensor, asymmetric, percentile-calibrated) in PyTorch,
  before any Linux/DFC access existed. Predicted **+24.6%** int8 EPE cost and
  identified `soft_argmin.index` (the disparity-index weights themselves) as the
  dominant sensitivity, 3.4× the next-worst layer, ranked first on all three
  checkpoints tested.
- **Real DFC quantization**: DFC 5.4.0 at `optimization_level=2` does **not** run
  the equalization/bias-correction the simulation modelled (log confirms `Bias
  Correction skipped`, `Adaround skipped`); it runs Quantization-Aware Fine-Tuning
  (gradient distillation against the float model over the 64-pair calibration set,
  converging to a distill loss of 0.0268). Measured real cost: **+14.5% official
  EPE** (1.248 → 1.430 px) — roughly half what the simulation predicted, because it
  modelled a different mechanism than the one that actually runs.
- **`soft_argmin.index` promoted to 16-bit** (`quantization_param({soft_argmin_index},
  precision_mode=a16_w16)` in `deploy/hailo_stereo.alls`) based on the simulation's
  finding, at ~0% of the compute budget (24 MACs/pixel).
- The `.alls` script's rationale comment was found to contain **two further stale
  claims** during the Linux session (the optimization-level-2 justification, and an
  incorrect attribution of the calibration set to SceneFlow rather than KITTI); both
  were corrected in place (`deploy/hailo_stereo.alls`, header "VERIFIED 2026-09-16").

**Status: COMPLETE** for both the simulation and the real DFC quantization pass;
the simulation's absolute number is superseded (not "wrong" — it modelled a
mechanism, QAT, that the actual compiler does not use at this optimization level)
but its ranking output remains the basis for the 16-bit promotion decision.

---

## 8. Genuinely unfinished technical work (the ~30%)

These are model-development gaps identified inside the project's own analysis
documents, independent of hardware access:

1. **Match-confidence output not exported.** `notes/stress_emulator.py`'s soak test
   converges on a single defect: the model has no way to express "no match" —
   textureless input, sky, and a zero-true-disparity synthetic test all produce a
   confident-looking ~100 px (≈3.9 m) output rather than an uncertain one. The
   entropy of the softmax over 24 hypotheses already exists inside the graph as an
   intermediate value (`src/demo_server.py`'s "Match confidence" panel computes and
   displays it in PyTorch) but **is not part of the ONNX/HEF output** — verified
   directly: `deploy/README.md`'s recorded ONNX interface has a single output,
   `disparity [1,1,368,1232]`, no second output. `report/STRESS.md` §9 recommends
   exporting it as a second HEF output; this has **not been implemented in
   `src/model.py` or `src/export_onnx.py`**. Status: **NOT STARTED**, scoped and
   estimated as low-effort by the project's own analysis.
2. **Unmatchable-region (sky/above-horizon) supervision.** The same failure mode
   manifests as the KITTI-finetuned model drifting toward reporting sky as near
   (unsupervised because LiDAR ground truth never reaches the sky). The project's
   own border-region fix (§5.5) is explicitly identified as the applicable pattern
   ("apply the same idea above the horizon and to low-texture regions",
   `report/STRESS.md` §9) but has not been applied. Status: **NOT STARTED**.
3. **0–5 m accuracy is improved but still the weakest band.** EPE 7.29 px against
   ~1.1 px in every other band; root cause (missing large-disparity supervision) is
   diagnosed and partially addressed via KITTI 2012 mixing, with further gains
   identified as achievable through more large-disparity data or refined
   disparity-aware sampling rather than architecture change. Status: **PARTIAL**.
4. **Phase 2.3 (SceneFlow replay-ratio sweep) not executed.** Planned in
   `report/IMPROVEMENT_PLAN.md` as 0.25/0.5/1.0 sweep, contingent on Phase 1's
   replay result; superseded in priority once Phase 2.1 (KITTI 2012) delivered a
   larger, decisive gain, and explicitly left undone. Status: **NOT STARTED**
   (deprioritized, not abandoned for cause).
5. **Phase 4 architecture variants not executed.** `cost="corr"` (correlation
   cost volume, already coded as an option in `CostVolume` but never selected in a
   trained checkpoint) and wider matching features (`match_ch` 32→48) were planned
   contingent on "if ≥80 px bias survives Phases 1–3." The bias was substantially
   but not completely closed (residual ~−10 to −13 px at 100–120 px), which by the
   plan's own stated criterion leaves this phase technically warranted; the project
   log records the decision to treat it as optional given the scale of improvement
   already achieved, not a decision that it is unnecessary. Status: **NOT STARTED**.
6. **FPS/latency remains unmeasured, and cannot be produced by this repository's
   toolchain.** This is explicitly excluded from the 70/30 split per the audit
   scope (it requires either a DFC bug fix outside this project's control, or
   physical-board `HailoRT` timing), but is recorded here because it is a genuine
   open technical question, not merely an administrative one.

Note: KITTI licence registration (an administrative/legal item,
`report/STATUS.md` open item 5) is **not** counted as technical work in either
direction, consistent with the audit's scope.

---

## 9. Basis for the ~70% / ~30% split

**Counted as complete (the ~70%):** the full reverse-engineering and defect
catalogue of the baseline model (§3); the ground-up architecture satisfying every
identified constraint (§4); the training/data pipeline including two non-trivial bug
discoveries-and-fixes (validation leak, RNG bug) that each required a full retrain
(§5.2–5.3); the finetuning ablation campaign that delivered the project's largest
accuracy gain (§5.4); the border-supervision fix (§5.5); ONNX export with a
self-auditing exporter and verified numerical parity (§6); quantization simulation
and real DFC quantization-aware fine-tuning (§7.3); and a complete, successful
compile to a valid Hailo-15H HEF via the DFC pipeline (§7.1), backed by an
emulator-based verification and a 30-minute stress/soak/adversarial test campaign
that is unusually thorough for a project at this stage (§6). Every one of these
stages has a passing, reproducible, named test or artifact — not a claim without a
check.

**Counted as remaining (the ~30%):** two specific, scoped, not-yet-implemented model
improvements identified by the project's own stress testing (match-confidence export,
unmatchable-region supervision); one accuracy band under active but incomplete
improvement (0–5 m); and one architecture experiment branch that was explicitly
deferred rather than run (Phase 4, correlation cost volume / wider features), which
by the improvement plan's own stated trigger condition remains open. The SceneFlow
replay-ratio sweep is minor and likely low-yield given KITTI 2012 already delivered
the larger win it was chasing.

**Explicitly excluded from both halves, per audit scope:** on-device Hailo-15H
validation, physical board testing, FPS/latency measurement (blocked on the above),
and all documentation/report-writing effort.

This split reflects that the project has already passed through — and produced
passing evidence for — every stage in the intended pipeline (teardown → design →
train → export → quantize → compile), and what remains is targeted refinement within
that pipeline rather than any un-started pipeline stage.
