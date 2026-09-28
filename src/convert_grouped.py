"""
Convert EdgeDepth-8 to EdgeDepth-8-grouped and verify the conversion.

    python src/convert_grouped.py \
        --src release/EdgeDepth-8/EdgeDepth-8.pt \
        --out-dir release/EdgeDepth-8-grouped

Steps:
  a. load the original checkpoint, build the grouped model via
     convert_from_original, both .eval().
  b. Equivalence: random input 1x3x368x1232 (seed 0) AND the 40 KITTI val
     pairs (same loader/crop as src/eval_kitti.py). Reports max
     |disp_orig - disp_grouped| per check; FAILS (exit 1) if > 1e-3 px.
  c. Score both on the 40 pairs with the same metric as src/eval_kitti.py
     (masked + official EPE, D1). Grouped must match to 3 decimals.
  d. Save EdgeDepth-8-grouped.pt (state_dict + arch + cost_layout + source).
  e. Export EdgeDepth-8-grouped.onnx mirroring src/export_onnx.py exactly,
     plus an onnxruntime check on 2 pairs vs torch (< 1e-3 px).
  f. Op-count table (Conv, Sub, Mul, Concat, Slice, total) for the original
     ONNX vs the new ONNX, and MAC count via model.count_ops for both.
  g. Write release/EdgeDepth-8-grouped/MODEL_CARD.md.

Uses CUDA if available; checks the GPU temperature first and waits until it
is below 75 C before doing GPU work.
"""

from __future__ import annotations

import argparse
import collections
import copy
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import count_ops  # noqa: E402
from model_grouped import convert_from_original, load_original  # noqa: E402
from data import Kitti2015  # noqa: E402
from eval_kitti import score  # noqa: E402  (same metric as eval_kitti.py)
from export_onnx import strip_identity  # noqa: E402  (same post-step as export)

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

HEIGHT, WIDTH = 368, 1232
MAX_EQUIV_PX = 1e-3
MAX_ORT_PX = 1e-3


def wait_for_cool_gpu(device: str) -> None:
    """Before GPU work, wait until the GPU temperature is below 75 C."""
    if device == "cpu":
        return
    while True:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
            temps = [float(t) for t in out.stdout.strip().split()
                     if t.strip()[:1].isdigit()]
        except Exception as exc:  # nvidia-smi missing: proceed, nothing to check
            print(f"(gpu temp check skipped: {exc})")
            return
        if not temps:
            return
        hottest = max(temps)
        print(f"gpu temperature: {hottest:.0f} C")
        if hottest < 75:
            return
        print("gpu over 75 C -- waiting 30 s ...")
        time.sleep(30)


def score_split(model, ds, device: str):
    """Score a model on every pair of ds under the masked and official
    protocols, exactly like src/eval_kitti.py. Returns
    {protocol: (EPE, D1-all)} and the per-pair worst error bookkeeping."""
    acc = {"masked": [0.0, 0.0, 0], "official": [0.0, 0.0, 0]}
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            pred = model(s["left"][None].to(device),
                         s["right"][None].to(device))[0, 0].cpu().numpy()
            disp = s["disp"].numpy()[0]
            masked = s["mask"].numpy()[0] > 0.5
            official = disp > 0.0
            for name, m in (("masked", masked), ("official", official)):
                epe, d1, n = score(pred, disp, m)
                if n:
                    acc[name][0] += epe * n
                    acc[name][1] += d1 * n
                    acc[name][2] += n
    return {name: (acc[name][0] / acc[name][2], acc[name][1] / acc[name][2])
            for name in ("masked", "official")}


def onnx_op_counts(path: pathlib.Path) -> collections.Counter:
    import onnx
    graph = onnx.load(str(path)).graph
    hist = collections.Counter(n.op_type for n in graph.node)
    hist["_total"] = len(graph.node)
    return hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="source checkpoint, e.g. release/EdgeDepth-8/EdgeDepth-8.pt")
    ap.add_argument("--out-dir", required=True,
                    help="output dir, e.g. release/EdgeDepth-8-grouped")
    ap.add_argument("--kitti-root", default=str(REPO / "data" / "kitti2015" / "training"),
                    help="KITTI 2015 training root holding image_2/image_3/disp_occ_0")
    ap.add_argument("--orig-onnx", default=None,
                    help="original ONNX for the op table "
                         "(default: <src-stem>.onnx next to --src)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    wait_for_cool_gpu(device)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    # ---- (a) load + convert -------------------------------------------
    print("=" * 68)
    print("(a) load original, build grouped")
    print("=" * 68)
    orig = load_original(args.src).to(device).eval()
    print(f"  source arch: {orig.arch}")
    grouped = convert_from_original(orig).to(device).eval()
    print("  grouped model built by weight remapping (no training)")

    # ---- (b) equivalence ------------------------------------------------
    print("=" * 68)
    print("(b) equivalence: max |disp_orig - disp_grouped|")
    print("=" * 68)
    torch.manual_seed(0)
    left = torch.randn(1, 3, HEIGHT, WIDTH, device=device)
    right = torch.randn(1, 3, HEIGHT, WIDTH, device=device)
    with torch.no_grad():
        rand_diff = float((orig(left, right) - grouped(left, right)).abs().max())
    print(f"  random 1x3x{HEIGHT}x{WIDTH} (seed 0): {rand_diff:.3e} px")
    if rand_diff > MAX_EQUIV_PX:
        failures.append(f"random-input equivalence {rand_diff:.3e} > {MAX_EQUIV_PX:.0e}")

    ds = Kitti2015(args.kitti_root, split="val")
    assert len(ds) == 40, f"expected 40 KITTI val pairs, got {len(ds)}"
    print(f"  KITTI val: {len(ds)} pairs (same loader/crop as eval_kitti.py)")
    worst, worst_idx = 0.0, -1
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            l, r = s["left"][None].to(device), s["right"][None].to(device)
            d = float((orig(l, r) - grouped(l, r)).abs().max())
            if d > worst:
                worst, worst_idx = d, i
    print(f"  KITTI val worst pair: {worst:.3e} px (pair {worst_idx})")
    if worst > MAX_EQUIV_PX:
        failures.append(f"KITTI equivalence {worst:.3e} > {MAX_EQUIV_PX:.0e}")

    # ---- (c) EPE parity --------------------------------------------------
    print("=" * 68)
    print("(c) KITTI val EPE (same metric as eval_kitti.py)")
    print("=" * 68)
    res_orig = score_split(orig, ds, device)
    res_grouped = score_split(grouped, ds, device)
    print(f"  {'protocol':10s} {'model':>10s} {'EPE':>9s} {'D1-all':>9s}")
    for name in ("masked", "official"):
        eo, do = res_orig[name]
        eg, dg = res_grouped[name]
        print(f"  {name:10s} {'original':>10s} {eo:8.3f}p {do:8.2f}%")
        print(f"  {name:10s} {'grouped':>10s} {eg:8.3f}p {dg:8.2f}%")
        if round(eo, 3) != round(eg, 3):
            failures.append(
                f"{name} EPE mismatch: {eo:.4f} vs {eg:.4f} (must match to 3 decimals)")
    print("  expected original: 0.987 masked / 1.067 official")

    # ---- (d) save checkpoint ---------------------------------------------
    print("=" * 68)
    print("(d) save checkpoint")
    print("=" * 68)
    ckpt_path = out_dir / "EdgeDepth-8-grouped.pt"
    torch.save({
        "model": {k: v.cpu() for k, v in grouped.state_dict().items()},
        "arch": dict(grouped.arch),
        "cost_layout": "grouped",
        "source_ckpt": str(args.src),
        "source_arch": dict(orig.arch),
        "note": ("EdgeDepth-8 with the 24-branch cost='both' cost volume fused "
                 "into one grouped 1x1 (GroupedCostVolume) plus a stem "
                 "input-channel permutation. Same math, no retraining. "
                 "See MODEL_CARD.md."),
    }, ckpt_path)
    print(f"  wrote {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.2f} MB)")

    # ---- (e) ONNX export (mirrors export_onnx.py) -------------------------
    print("=" * 68)
    print("(e) ONNX export + onnxruntime check")
    print("=" * 68)
    onnx_path = out_dir / "EdgeDepth-8-grouped.onnx"
    export_model = convert_from_original(load_original(args.src)).eval()
    reference_state = copy.deepcopy(export_model.state_dict())
    export_model.fuse_temperature_()  # same fusion as export_onnx.py
    zl = torch.zeros(1, 3, HEIGHT, WIDTH)
    zr = torch.zeros(1, 3, HEIGHT, WIDTH)
    # Same arguments as src/export_onnx.py (opset 14, fixed zeros input,
    # input/output names, constant folding, non-dynamo exporter). The
    # `dynamo=False` keyword only exists on newer torch; pass it when the
    # installed exporter accepts it, otherwise the default IS the legacy
    # (non-dynamo) path export_onnx.py selects.
    import inspect as _inspect
    export_kw: dict = dict(input_names=["left", "right"],
                           output_names=["disparity"],
                           opset_version=14, do_constant_folding=True)
    if "dynamo" in _inspect.signature(torch.onnx.export).parameters:
        export_kw["dynamo"] = False
    torch.onnx.export(export_model, (zl, zr), str(onnx_path), **export_kw)
    strip_identity(onnx_path)  # same post-step as export_onnx.py
    print(f"  wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")

    try:
        import onnxruntime as ort
    except ImportError:
        failures.append("onnxruntime unavailable; torch-vs-onnx NOT checked")
        ort = None
    if ort is not None:
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
        worst_ort = 0.0
        with torch.no_grad():
            for i in range(2):  # 2 val pairs, torch vs onnx
                s = ds[i]
                l = s["left"][None].to(device)
                r = s["right"][None].to(device)
                expected = grouped(l, r)[0, 0].cpu().numpy()
                got = sess.run(["disparity"],
                               {"left": l.cpu().numpy(),
                                "right": r.cpu().numpy()})[0][0, 0]
                worst_ort = max(worst_ort, float(np.abs(got - expected).max()))
        print(f"  onnxruntime vs torch (2 pairs): worst {worst_ort:.3e} px")
        if worst_ort > MAX_ORT_PX:
            failures.append(f"onnx parity {worst_ort:.3e} > {MAX_ORT_PX:.0e}")

    # ---- (f) op-count + MAC tables ----------------------------------------
    print("=" * 68)
    print("(f) op counts and MACs")
    print("=" * 68)
    orig_onnx = (pathlib.Path(args.orig_onnx) if args.orig_onnx else
                 pathlib.Path(args.src).parent / (pathlib.Path(args.src).stem + ".onnx"))
    assert orig_onnx.exists(), f"original ONNX not found: {orig_onnx}"
    keys = ["Conv", "Sub", "Mul", "Concat", "Slice"]
    h0, h1 = onnx_op_counts(orig_onnx), onnx_op_counts(onnx_path)
    op_rows = [(k, h0.get(k, 0), h1.get(k, 0)) for k in keys]
    op_rows.append(("TOTAL nodes", h0["_total"], h1["_total"]))
    print(f"  {'op':12s} {'orig':>8s} {'grouped':>8s}")
    for k, a, b in op_rows:
        print(f"  {k:12s} {a:8d} {b:8d}")

    mac_rows = []
    for name, m in (("original", orig), ("grouped", grouped)):
        totals, total = count_ops(m, HEIGHT, WIDTH, device=device)
        mac_rows.append((name, totals, total))
        print(f"  {name}: {total / 1e9:.2f} GMAC")
    order = sorted(mac_rows[0][1].items(), key=lambda kv: -kv[1])
    print(f"  {'stage':18s} {'orig GMAC':>10s} {'grouped GMAC':>12s}")
    for stage, _ in order:
        a = mac_rows[0][1].get(stage, 0)
        b = mac_rows[1][1].get(stage, 0)
        print(f"  {stage:18s} {a / 1e9:10.3f} {b / 1e9:12.3f}")

    # ---- (g) MODEL_CARD ----------------------------------------------------
    print("=" * 68)
    print("(g) MODEL_CARD.md")
    print("=" * 68)
    eo_m, do_m = res_orig["masked"]
    eo_o, do_o = res_orig["official"]
    eg_m, dg_m = res_grouped["masked"]
    eg_o, dg_o = res_grouped["official"]
    op_table = "\n".join(
        f"| {k} | {a} | {b} |" for k, a, b in op_rows)
    mac_table = "\n".join(
        f"| {stage} | {mac_rows[0][1].get(stage, 0) / 1e9:.3f} "
        f"| {mac_rows[1][1].get(stage, 0) / 1e9:.3f} |"
        for stage, _ in order)
    card = f"""# EdgeDepth-8-grouped (v1.0)

Same math as EdgeDepth-8, repackaged for the Hailo-8 dataflow compiler.
No retraining: weights are remapped, not relearned.

**Source:** `{args.src}` (arch: {orig.arch})

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
| random 1x3x368x1232 (seed 0) | {rand_diff:.3e} px |
| KITTI 2015 val, worst of 40 pairs | {worst:.3e} px (pair {worst_idx}) |
| onnxruntime vs torch, worst of 2 pairs | {worst_ort if ort is not None else float('nan'):.3e} px |

## Accuracy: KITTI 2015 val (40 pairs), float

| Protocol | original EPE | grouped EPE | original D1-all | grouped D1-all |
|----------|--------------|-------------|-----------------|----------------|
| official | {eo_o:.3f} px | {eg_o:.3f} px | {do_o:.2f}% | {dg_o:.2f}% |
| masked   | {eo_m:.3f} px | {eg_m:.3f} px | {do_m:.2f}% | {dg_m:.2f}% |

Grouped matches original to 3 decimals on both protocols.

## Op counts (ONNX node types)

`{orig_onnx.name}` vs `{onnx_path.name}`:

| op | original | grouped |
|---|---|---|
{op_table}

## MACs (`model.count_ops`, {HEIGHT}x{WIDTH})

| stage | original GMAC | grouped GMAC |
|---|---|---|
{mac_table}

| TOTAL | {mac_rows[0][2] / 1e9:.3f} | {mac_rows[1][2] / 1e9:.3f} |

## Files

- `EdgeDepth-8-grouped.pt` (state_dict + arch + `"cost_layout": "grouped"` + source note)
- `EdgeDepth-8-grouped.onnx` (opset 14, inputs `left`/`right`, output
  `disparity`, fixed {HEIGHT}x{WIDTH}, fused soft-argmin -- same export
  settings as `src/export_onnx.py`)
"""
    (out_dir / "MODEL_CARD.md").write_text(card)
    print(f"  wrote {out_dir / 'MODEL_CARD.md'}")

    print("=" * 68)
    if failures:
        print("FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED -- grouped model is equivalent, scores match, artifacts written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
