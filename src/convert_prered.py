"""
Convert EdgeDepth-8 to EdgeDepth-8-prered and verify the conversion.

    python src/convert_prered.py \
        --src release/EdgeDepth-8/EdgeDepth-8.pt \
        --out-dir release/EdgeDepth-8-prered

Steps:
  a. load the original checkpoint, build the prered model via
     convert_from_original, both .eval().
  b. Equivalence: random input 1x3x368x1232 (seed 0) AND the 40 KITTI val
     pairs (same loader/crop as src/eval_kitti.py). Reports max
     |disp_orig - disp_prered| per check; FAILS (exit 1) if > 1e-3 px.
  c. Score both on the 40 pairs with the same metric as src/eval_kitti.py
     (masked + official EPE, D1). Prered must match to 3 decimals.
  d. Save EdgeDepth-8-prered.pt (state_dict + arch + cost_layout + source).
  e. Export EdgeDepth-8-prered.onnx mirroring src/export_onnx.py exactly,
     plus an onnxruntime check on 2 pairs vs torch (< 1e-3 px).
  f. Op-count table (Conv, Sub, Mul, Concat, Slice, total) for the original
     ONNX vs the grouped ONNX vs the new ONNX, ONNX channel widths of the
     cost-volume tensors, and MAC count via model.count_ops.
  g. Write release/EdgeDepth-8-prered/MODEL_CARD.md.

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
from model_prered import convert_from_original, load_original  # noqa: E402
from data import Kitti2015  # noqa: E402
from eval_kitti import score  # noqa: E402  (same metric as eval_kitti.py)
from export_onnx import strip_identity  # noqa: E402  (same post-step as export)

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

HEIGHT, WIDTH = 368, 1232
MAX_EQUIV_PX = 1e-3
MAX_ORT_PX = 1e-3
# Matching resolution for the fixed HEIGHTxWIDTH export.
MATCH_H, MATCH_W = HEIGHT // 8, WIDTH // 8  # 46 x 154


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


def onnx_cost_volume_widths(path: pathlib.Path) -> dict:
    """Channel widths of cost-volume tensors, measured from the ONNX graph.

    Runs shape inference and collects, over every tensor at matching
    resolution (46x154 for the fixed 368x1232 export), the widest channel
    dimension seen on the inputs/outputs of Sub, Mul and Concat nodes, plus
    the widest matching-resolution Conv input (the tensor feeding the
    aggregation stem region). Returns a dict with keys
    sub_in, mul_in, concat_out, conv_in (0 when unresolvable).
    """
    import onnx
    from onnx import shape_inference
    try:
        model = shape_inference.infer_shapes(onnx.load(str(path)))
    except Exception as exc:
        print(f"  (shape inference failed for {path.name}: {exc})")
        return {"sub_in": 0, "mul_in": 0, "concat_out": 0, "conv_in": 0}
    graph = model.graph
    dims: dict[str, list[int]] = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        tt = vi.type.tensor_type
        if tt.HasField("shape"):
            shape = [d.dim_value for d in tt.shape.dim]
            dims[vi.name] = shape
    # Initializers (weights) also carry shapes but are not matching-res tensors.
    out = {"sub_in": 0, "mul_in": 0, "concat_out": 0, "conv_in": 0}

    def ch_of(name: str) -> int:
        s = dims.get(name, [])
        if len(s) == 4 and s[2] == MATCH_H and s[3] == MATCH_W and s[1] > 0:
            return int(s[1])
        return 0

    for node in graph.node:
        if node.op_type in ("Sub", "Mul"):
            for inp in node.input:
                c = ch_of(inp)
                key = "sub_in" if node.op_type == "Sub" else "mul_in"
                out[key] = max(out[key], c)
            for o in node.output:
                c = ch_of(o)
                key = "sub_in" if node.op_type == "Sub" else "mul_in"
                out[key] = max(out[key], c)
        elif node.op_type == "Concat":
            for o in node.output:
                out["concat_out"] = max(out["concat_out"], ch_of(o))
            for inp in node.input:
                out["concat_out"] = max(out["concat_out"], ch_of(inp))
        elif node.op_type == "Conv":
            for inp in node.input[:1]:
                out["conv_in"] = max(out["conv_in"], ch_of(inp))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="source checkpoint, e.g. release/EdgeDepth-8/EdgeDepth-8.pt")
    ap.add_argument("--out-dir", required=True,
                    help="output dir, e.g. release/EdgeDepth-8-prered")
    ap.add_argument("--kitti-root", default=str(REPO / "data" / "kitti2015" / "training"),
                    help="KITTI 2015 training root holding image_2/image_3/disp_occ_0")
    ap.add_argument("--orig-onnx", default=None,
                    help="original ONNX for the op table "
                         "(default: <src-stem>.onnx next to --src)")
    ap.add_argument("--grouped-onnx", default=None,
                    help="grouped ONNX for the op table "
                         "(default: release/EdgeDepth-8-grouped/EdgeDepth-8-grouped.onnx)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"),
                    help="evaluation device for the equivalence/EPE gate "
                         "(default: cpu -- deterministic; cuda grouped-conv "
                         "kernels reorder floating-point summation across "
                         "layouts and amplify through softmax/refinement, so "
                         "the exactness gate runs on CPU; pass --device cuda "
                         "or auto for the src/convert_grouped.py behaviour)")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("cuda requested but unavailable -- using cpu")
            device = "cpu"
    print(f"device: {device}")
    wait_for_cool_gpu(device)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    # ---- (a) load + convert -------------------------------------------
    print("=" * 68)
    print("(a) load original, build prered")
    print("=" * 68)
    orig = load_original(args.src).to(device).eval()
    print(f"  source arch: {orig.arch}")
    prered = convert_from_original(orig).to(device).eval()
    print("  prered model built by weight remapping (no training)")

    # ---- (b) equivalence ------------------------------------------------
    print("=" * 68)
    print("(b) equivalence: max |disp_orig - disp_prered|")
    print("=" * 68)
    torch.manual_seed(0)
    left = torch.randn(1, 3, HEIGHT, WIDTH, device=device)
    right = torch.randn(1, 3, HEIGHT, WIDTH, device=device)
    with torch.no_grad():
        rand_diff = float((orig(left, right) - prered(left, right)).abs().max())
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
            d = float((orig(l, r) - prered(l, r)).abs().max())
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
    res_prered = score_split(prered, ds, device)
    print(f"  {'protocol':10s} {'model':>10s} {'EPE':>9s} {'D1-all':>9s}")
    for name in ("masked", "official"):
        eo, do = res_orig[name]
        eg, dg = res_prered[name]
        print(f"  {name:10s} {'original':>10s} {eo:8.3f}p {do:8.2f}%")
        print(f"  {name:10s} {'prered':>10s} {eg:8.3f}p {dg:8.2f}%")
        if round(eo, 3) != round(eg, 3):
            failures.append(
                f"{name} EPE mismatch: {eo:.4f} vs {eg:.4f} (must match to 3 decimals)")
        if round(do, 3) != round(dg, 3):
            failures.append(
                f"{name} D1 mismatch: {do:.4f} vs {dg:.4f} (must match to 3 decimals)")
    print("  expected original: 0.987 masked / 1.067 official")

    # ---- (d) save checkpoint ---------------------------------------------
    print("=" * 68)
    print("(d) save checkpoint")
    print("=" * 68)
    ckpt_path = out_dir / "EdgeDepth-8-prered.pt"
    torch.save({
        "model": {k: v.cpu() for k, v in prered.state_dict().items()},
        "arch": dict(prered.arch),
        "cost_layout": "prered",
        "source_ckpt": str(args.src),
        "source_arch": dict(orig.arch),
        "note": ("EdgeDepth-8 with the 24-branch cost='both' cost volume "
                 "pre-reduced (PreReducedCostVolume): sub path reduces once "
                 "per view then shifts in G-channel space; corr path uses one "
                 "grouped 1x1 over stacked products; plus a stem "
                 "input-channel permutation. Same math, no retraining. "
                 "See MODEL_CARD.md."),
    }, ckpt_path)
    print(f"  wrote {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.2f} MB)")

    # ---- (e) ONNX export (mirrors export_onnx.py) -------------------------
    print("=" * 68)
    print("(e) ONNX export + onnxruntime check")
    print("=" * 68)
    onnx_path = out_dir / "EdgeDepth-8-prered.onnx"
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
    worst_ort = float("nan")
    if ort is not None:
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
        worst_ort = 0.0
        with torch.no_grad():
            for i in range(2):  # 2 val pairs, torch vs onnx
                s = ds[i]
                l = s["left"][None].to(device)
                r = s["right"][None].to(device)
                expected = prered(l, r)[0, 0].cpu().numpy()
                got = sess.run(["disparity"],
                               {"left": l.cpu().numpy(),
                                "right": r.cpu().numpy()})[0][0, 0]
                worst_ort = max(worst_ort, float(np.abs(got - expected).max()))
        print(f"  onnxruntime vs torch (2 pairs): worst {worst_ort:.3e} px")
        if worst_ort > MAX_ORT_PX:
            failures.append(f"onnx parity {worst_ort:.3e} > {MAX_ORT_PX:.0e}")

    # ---- (f) op-count + MAC + channel-width tables ------------------------
    print("=" * 68)
    print("(f) op counts, cost-volume widths, and MACs")
    print("=" * 68)
    orig_onnx = (pathlib.Path(args.orig_onnx) if args.orig_onnx else
                 pathlib.Path(args.src).parent / (pathlib.Path(args.src).stem + ".onnx"))
    assert orig_onnx.exists(), f"original ONNX not found: {orig_onnx}"
    grouped_default = REPO / "release" / "EdgeDepth-8-grouped" / "EdgeDepth-8-grouped.onnx"
    grouped_onnx = (pathlib.Path(args.grouped_onnx) if args.grouped_onnx
                    else grouped_default)
    h0, h2 = onnx_op_counts(orig_onnx), onnx_op_counts(onnx_path)
    h1 = onnx_op_counts(grouped_onnx) if grouped_onnx.exists() else None
    if h1 is None:
        print(f"  (grouped ONNX not found at {grouped_onnx}; op table is orig vs prered)")
    keys = ["Conv", "Sub", "Mul", "Concat", "Slice"]
    op_rows = [(k, h0.get(k, 0),
                (h1.get(k, 0) if h1 is not None else None),
                h2.get(k, 0)) for k in keys]
    op_rows.append(("TOTAL nodes", h0["_total"],
                    (h1["_total"] if h1 is not None else None), h2["_total"]))
    if h1 is not None:
        print(f"  {'op':12s} {'orig':>8s} {'grouped':>8s} {'prered':>8s}")
        for k, a, b, c in op_rows:
            print(f"  {k:12s} {a:8d} {b:8d} {c:8d}")
    else:
        print(f"  {'op':12s} {'orig':>8s} {'prered':>8s}")
        for k, a, b, c in op_rows:
            print(f"  {k:12s} {a:8d} {c:8d}")

    w0 = onnx_cost_volume_widths(orig_onnx)
    w1 = onnx_cost_volume_widths(grouped_onnx) if grouped_onnx.exists() else None
    w2 = onnx_cost_volume_widths(onnx_path)
    print(f"  cost-volume channel widths at 1/8 res ({MATCH_H}x{MATCH_W}), from ONNX shapes:")
    if w1 is not None:
        print(f"  {'tensor':22s} {'orig':>8s} {'grouped':>8s} {'prered':>8s}")
        for key in ("sub_in", "mul_in", "concat_out", "conv_in"):
            print(f"  {key:22s} {w0[key]:8d} {w1[key]:8d} {w2[key]:8d}")
    else:
        print(f"  {'tensor':22s} {'orig':>8s} {'prered':>8s}")
        for key in ("sub_in", "mul_in", "concat_out", "conv_in"):
            print(f"  {key:22s} {w0[key]:8d} {w2[key]:8d}")
    print("  (sub_in/mul_in = widest Sub/Mul operand; concat_out = widest "
          "Concat tensor; conv_in = widest matching-res Conv input)")

    mac_rows = []
    for name, m in (("original", orig), ("prered", prered)):
        totals, total = count_ops(m, HEIGHT, WIDTH, device=device)
        mac_rows.append((name, totals, total))
        print(f"  {name}: {total / 1e9:.2f} GMAC")
    order = sorted(mac_rows[0][1].items(), key=lambda kv: -kv[1])
    print(f"  {'stage':18s} {'orig GMAC':>10s} {'prered GMAC':>12s}")
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
    eg_m, dg_m = res_prered["masked"]
    eg_o, dg_o = res_prered["official"]
    if h1 is not None:
        op_table = "\n".join(
            f"| {k} | {a} | {b} | {c} |" for k, a, b, c in op_rows)
        op_header = "| op | original | grouped | prered |"
        op_title = f"`{orig_onnx.name}` vs `{grouped_onnx.name}` vs `{onnx_path.name}`:"
    else:
        op_table = "\n".join(
            f"| {k} | {a} | {c} |" for k, a, b, c in op_rows)
        op_header = "| op | original | prered |"
        op_title = f"`{orig_onnx.name}` vs `{onnx_path.name}`:"
    if w1 is not None:
        width_table = "\n".join(
            f"| {key} | {w0[key]} | {w1[key]} | {w2[key]} |"
            for key in ("sub_in", "mul_in", "concat_out", "conv_in"))
        width_header = "| cost-volume tensor | original ch | grouped ch | prered ch |"
    else:
        width_table = "\n".join(
            f"| {key} | {w0[key]} | {w2[key]} |"
            for key in ("sub_in", "mul_in", "concat_out", "conv_in"))
        width_header = "| cost-volume tensor | original ch | prered ch |"
    mac_table = "\n".join(
        f"| {stage} | {mac_rows[0][1].get(stage, 0) / 1e9:.3f} "
        f"| {mac_rows[1][1].get(stage, 0) / 1e9:.3f} |"
        for stage, _ in order)
    card = f"""# EdgeDepth-8-prered (v1.0)

Same math as EdgeDepth-8, repackaged for the Hailo-8 dataflow compiler.
No retraining: weights are remapped, not relearned.

**Source:** `{args.src}` (arch: {orig.arch})

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
| random 1x3x368x1232 (seed 0) | {rand_diff:.3e} px |
| KITTI 2015 val, worst of 40 pairs | {worst:.3e} px (pair {worst_idx}) |
| onnxruntime vs torch, worst of 2 pairs | {worst_ort:.3e} px |

## Accuracy: KITTI 2015 val (40 pairs), float

| Protocol | original EPE | prered EPE | original D1-all | prered D1-all |
|----------|--------------|-------------|-----------------|----------------|
| official | {eo_o:.3f} px | {eg_o:.3f} px | {do_o:.2f}% | {dg_o:.2f}% |
| masked   | {eo_m:.3f} px | {eg_m:.3f} px | {do_m:.2f}% | {dg_m:.2f}% |

Prered matches original to 3 decimals on both protocols.

## Op counts (ONNX node types)

{op_title}

{op_header}
|---|---|---|{"---|" if h1 is not None else ""}
{op_table}

## ONNX channel widths of the cost-volume tensors

Measured by ONNX shape inference at matching resolution
({MATCH_H}x{MATCH_W} for the fixed {HEIGHT}x{WIDTH} export).
`sub_in`/`mul_in` = widest Sub/Mul operand channels,
`concat_out` = widest Concat tensor channels,
`conv_in` = widest matching-resolution Conv input channels.

{width_header}
|---|---|---|{"---|" if w1 is not None else ""}
{width_table}

Statically, the prered graph replaces the grouped 1536-channel
`cat[sub 768 | corr 768]` with a 192-channel sub path
(`cat([rl]*D)` / shifted `rr`, 8 -> 192 ch) alongside the 768-channel corr
path (`big_left`/`big_right`/product, 768 ch), concatenated once to the same
384-channel `[sub 192 | corr 192]` cost volume the stem consumes.

## MACs (`model.count_ops`, {HEIGHT}x{WIDTH})

| stage | original GMAC | prered GMAC |
|---|---|---|
{mac_table}

| TOTAL | {mac_rows[0][2] / 1e9:.3f} | {mac_rows[1][2] / 1e9:.3f} |

## Files

- `EdgeDepth-8-prered.pt` (state_dict + arch + `"cost_layout": "prered"` + source note)
- `EdgeDepth-8-prered.onnx` (opset 14, inputs `left`/`right`, output
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
    print("PASSED -- prered model is equivalent, scores match, artifacts written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
