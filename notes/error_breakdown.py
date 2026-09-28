"""Error-source breakdown for a stereo checkpoint on KITTI 2015 val.

Follows src/eval_kitti.py exactly for loading, dataset, inference,
the masked protocol (mask > 0.5) and the official protocol (disp > 0),
and D1 (err > 3 AND err > 5% of GT). Read-only: no training, no edits.

Usage:
    python notes/error_breakdown.py --ckpt runs/kft_both_s60/best.pt \
        --root data/kitti2015/training --tag s60ep58 [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from model import build_model, MAX_DISP  # noqa: E402
from data import Kitti2015  # noqa: E402

try:
    from scipy.ndimage import maximum_filter, minimum_filter
except ImportError:  # pragma: no cover
    maximum_filter = minimum_filter = None

ERR_BINS = [
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 5.0),
    (5.0, 10.0),
    (10.0, float("inf")),
]
DISP_BINS = [
    (0.0, 10.0),
    (10.0, 20.0),
    (20.0, 40.0),
    (40.0, 80.0),
    (80.0, 120.0),
    (120.0, float(MAX_DISP)),
    (float(MAX_DISP), float("inf")),
]

# Expected headlines for runs/kft_both_s60/best.pt from src/eval_kitti.py.
EXPECTED = {"masked_epe": 0.987, "official_epe": 1.067, "official_d1": 5.06}


def fmt_bin(lo, hi):
    if np.isinf(hi):
        return f">={lo:g}"
    return f"[{lo:g},{hi:g})"


def disp_bin_name(lo, hi):
    if np.isinf(hi):
        return f"[{lo:g},inf)"
    if lo == float(MAX_DISP):
        return f"[{int(lo)},inf)"
    if hi == float(MAX_DISP):
        return f"[{lo:g},{int(hi)})"
    return f"[{lo:g},{hi:g})"


def edge_proximity(disp, valid, size=7):
    """Bool map: True where the size x size neighbourhood contains a valid GT
    pixel differing by > 3 px. Only valid GT pixels participate (invalid
    filled with +-inf equivalents so they never trigger). numpy/scipy only."""
    assert maximum_filter is not None, "scipy is required for edge proximity"
    big = 1e9
    hi_f = np.where(valid, disp, -big)
    lo_f = np.where(valid, disp, big)
    mx = maximum_filter(hi_f, size=size, mode="constant", cval=-big)
    mn = minimum_filter(lo_f, size=size, mode="constant", cval=big)
    near = ((mx - disp > 3.0) | (disp - mn > 3.0)) & valid
    return near


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = args.device
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(blob)
    model.load_state_dict(blob.get("model", blob), strict=True)
    model.to(device).eval()
    ds = Kitti2015(args.root, split="val")
    assert len(ds) == 40, f"expected 40 val pairs, got {len(ds)}"

    # Accumulators per protocol: sums so aggregation is by SUMS, not means.
    acc = {}
    for name in ("masked", "official"):
        acc[name] = {
            "n": 0,
            "abs": 0.0,
            "signed": 0.0,
            "d1_n": 0,
            "region": {"border": [0, 0.0], "matched": [0, 0.0]},
            "grange": {"in_range": [0, 0.0], "out_of_range": [0, 0.0]},
            "errbins": [[0, 0.0] for _ in ERR_BINS],
            "d1_abs": 0.0,
            "dispbins": [[0, 0.0, 0.0] for _ in DISP_BINS],  # n, abs, signed
            "edge": {"near": [0, 0.0], "far": [0, 0.0]},
        }
    frames = []  # per-frame official stats

    for i in range(len(ds)):
        s = ds[i]
        fname = ds.samples[i][0].name
        with torch.no_grad():
            pred = model(s["left"][None].to(device),
                         s["right"][None].to(device))[0, 0].cpu().numpy()
        disp = s["disp"].numpy()[0]
        masked = s["mask"].numpy()[0] > 0.5
        official = disp > 0.0
        H, W = disp.shape
        assert pred.shape == disp.shape, (pred.shape, disp.shape)
        err = np.abs(pred - disp)
        signed = pred - disp
        d1 = (err > 3.0) & (err > 0.05 * np.abs(disp))
        col = np.broadcast_to(np.arange(W), disp.shape)
        is_border = col < MAX_DISP
        oor = disp > float(MAX_DISP)
        near = edge_proximity(disp, official, size=7)

        # error-magnitude bin index per pixel
        aerr = err
        ebi = np.full(disp.shape, len(ERR_BINS) - 1, dtype=np.int64)
        for b, (lo, hi) in enumerate(ERR_BINS):
            if b == len(ERR_BINS) - 1:
                continue
            if b == 0:
                ebi[(aerr >= lo) & (aerr < hi)] = b
            else:
                ebi[(aerr >= lo) & (aerr < hi)] = b
        ebi[aerr >= 10.0] = len(ERR_BINS) - 1
        # disp bin index per pixel
        dbi = np.full(disp.shape, len(DISP_BINS) - 1, dtype=np.int64)
        for b, (lo, hi) in enumerate(DISP_BINS):
            if np.isinf(hi):
                dbi[disp >= lo] = b
            elif lo == 0.0:
                dbi[(disp >= lo) & (disp < hi) & (disp > 0.0)] = b
            else:
                dbi[(disp >= lo) & (disp < hi)] = b

        for name, m in (("masked", masked), ("official", official)):
            A = acc[name]
            n = int(m.sum())
            if n == 0:
                continue
            A["n"] += n
            A["abs"] += float(err[m].sum())
            A["signed"] += float(signed[m].sum())
            d1m = d1[m]
            A["d1_n"] += int(d1m.sum())
            A["d1_abs"] += float(err[m][d1m].sum())
            b = is_border[m]
            A["region"]["border"][0] += int(b.sum())
            A["region"]["border"][1] += float(err[m][b].sum())
            A["region"]["matched"][0] += int((~b).sum())
            A["region"]["matched"][1] += float(err[m][~b].sum())
            o = oor[m]
            A["grange"]["out_of_range"][0] += int(o.sum())
            A["grange"]["out_of_range"][1] += float(err[m][o].sum())
            A["grange"]["in_range"][0] += int((~o).sum())
            A["grange"]["in_range"][1] += float(err[m][~o].sum())
            eb = ebi[m]
            ev = err[m]
            for bidx in range(len(ERR_BINS)):
                sel = eb == bidx
                A["errbins"][bidx][0] += int(sel.sum())
                A["errbins"][bidx][1] += float(ev[sel].sum())
            db = dbi[m]
            sv = signed[m]
            for bidx in range(len(DISP_BINS)):
                sel = db == bidx
                A["dispbins"][bidx][0] += int(sel.sum())
                A["dispbins"][bidx][1] += float(ev[sel].sum())
                A["dispbins"][bidx][2] += float(sv[sel].sum())
            ne = near[m]
            A["edge"]["near"][0] += int(ne.sum())
            A["edge"]["near"][1] += float(ev[ne].sum())
            A["edge"]["far"][0] += int((~ne).sum())
            A["edge"]["far"][1] += float(ev[~ne].sum())

        no = int(official.sum())
        eo = float(err[official].sum()) / no if no else float("nan")
        nm = int(masked.sum())
        em = float(err[masked].sum()) / nm if nm else float("nan")
        frames.append({"name": fname, "n_official": no, "epe_official": eo,
                       "abs_official": float(err[official].sum()),
                       "n_masked": nm, "epe_masked": em})

    out = {}
    for name in ("masked", "official"):
        A = acc[name]
        out[name] = {"epe": A["abs"] / A["n"], "d1": 100.0 * A["d1_n"] / A["n"],
                     "pixels": A["n"]}

    # Headline check against eval_kitti.py values.
    ok = (abs(out["masked"]["epe"] - EXPECTED["masked_epe"]) < 0.005
          and abs(out["official"]["epe"] - EXPECTED["official_epe"]) < 0.005
          and abs(out["official"]["d1"] - EXPECTED["official_d1"]) < 0.05)
    print(f"ckpt: {args.ckpt}  tag: {args.tag}  device: {device}")
    print(f"masked:   EPE {out['masked']['epe']:.3f} px  "
          f"D1 {out['masked']['d1']:.2f}%  n={out['masked']['pixels']:,d}")
    print(f"official: EPE {out['official']['epe']:.3f} px  "
          f"D1 {out['official']['d1']:.2f}%  n={out['official']['pixels']:,d}")
    print(f"(expected ~{EXPECTED['masked_epe']:.3f} masked / "
          f"~{EXPECTED['official_epe']:.3f} official, "
          f"D1 official ~{EXPECTED['official_d1']:.2f}%)")
    if not ok:
        print("HEADLINE MISMATCH vs eval_kitti.py -- stopping.")
        sys.exit(1)

    A = acc["official"]
    N, S = A["n"], A["abs"]

    def row(label, n, a):
        return (label, n, 100.0 * n / N, a / n if n else float("nan"),
                100.0 * a / S if S else float("nan"))

    print("\n[1] region (official)")
    print(f"{'group':>14s} {'pixels':>10s} {'%pix':>7s} "
          f"{'meanEPE':>8s} {'%err':>7s}")
    t1 = [row("border x<192", *A["region"]["border"]),
          row("matched x>=192", *A["region"]["matched"])]
    for lab, n, pp, me, pe in t1:
        print(f"{lab:>14s} {n:10,d} {pp:6.2f}% {me:7.3f}p {pe:6.2f}%")
    t1b = [row("in-range d<=192", *A["grange"]["in_range"]),
           row("out-of-range d>192", *A["grange"]["out_of_range"])]
    for lab, n, pp, me, pe in t1b:
        print(f"{lab:>14s} {n:10,d} {pp:6.2f}% {me:7.3f}p {pe:6.2f}%")

    print("\n[2] error magnitude |err| (official)")
    print(f"{'bin px':>12s} {'pixels':>10s} {'%pix':>7s} "
          f"{'meanEPE':>8s} {'%err':>7s}")
    t2 = [row(fmt_bin(*b), *v) for b, v in zip(ERR_BINS, A["errbins"])]
    for lab, n, pp, me, pe in t2:
        print(f"{lab:>12s} {n:10,d} {pp:6.2f}% {me:7.3f}p {pe:6.2f}%")
    d1row = row("D1 outliers", A["d1_n"], A["d1_abs"])
    print(f"{d1row[0]:>12s} {d1row[1]:10,d} {d1row[2]:6.2f}% "
          f"{d1row[3]:7.3f}p {d1row[4]:6.2f}%")
    inl_n = N - A["d1_n"]
    inl_abs = S - A["d1_abs"]
    epe_no_d1 = (S - A["d1_abs"]) / N
    epe_only_d1 = A["d1_abs"] / N
    print(f"EPE if every D1 outlier had error 0: {epe_no_d1:.3f} px "
          f"(inliers carry {100.0 * inl_abs / S:.2f}% of error mass)")
    print(f"EPE if every inlier had error 0:     {epe_only_d1:.3f} px "
          f"(outliers carry {100.0 * A['d1_abs'] / S:.2f}% of error mass)")

    print("\n[3] GT disparity bins (official)")
    print(f"{'GT disp':>12s} {'pixels':>10s} {'%pix':>7s} "
          f"{'meanEPE':>8s} {'%err':>7s} {'bias':>8s}")
    t3 = []
    for (lo, hi), (n, a, sg) in zip(DISP_BINS, A["dispbins"]):
        lab, nn, pp, me, pe = row(disp_bin_name(lo, hi), n, a)
        bias = sg / n if n else float("nan")
        t3.append((lab, nn, pp, me, pe, bias))
        print(f"{lab:>12s} {nn:10,d} {pp:6.2f}% {me:7.3f}p "
              f"{pe:6.2f}% {bias:+7.3f}")

    print("\n[4] depth-edge proximity, 7x7 GT neighbourhood differs by >3px "
          "(official)")
    print(f"{'group':>10s} {'pixels':>10s} {'%pix':>7s} "
          f"{'meanEPE':>8s} {'%err':>7s}")
    t4 = [row("near edge", *A["edge"]["near"]),
          row("rest", *A["edge"]["far"])]
    for lab, n, pp, me, pe in t4:
        print(f"{lab:>10s} {n:10,d} {pp:6.2f}% {me:7.3f}p {pe:6.2f}%")

    print("\n[5] signed bias mean(pred-GT) (official)")
    print(f"overall: {A['signed'] / N:+.3f} px")
    print(f"{'GT disp':>12s} {'bias':>8s}")
    for (lab, nn, pp, me, pe, bias) in t3:
        print(f"{lab:>12s} {bias:+8.3f}")

    print("\n[6] sub-pixel vs coarse (official inliers = not D1)")
    print(f"inliers: n={inl_n:,d} ({100.0 * inl_n / N:.2f}%), "
          f"mean|err|={inl_abs / inl_n:.3f} px, "
          f"error share={100.0 * inl_abs / S:.2f}%")
    print(f"outliers(D1): n={A['d1_n']:,d} ({100.0 * A['d1_n'] / N:.2f}%), "
          f"mean|err|={A['d1_abs'] / A['d1_n']:.3f} px, "
          f"error share={100.0 * A['d1_abs'] / S:.2f}%")

    fs = sorted(frames, key=lambda f: f["epe_official"], reverse=True)
    top5 = fs[:5]
    top5_abs = sum(f["abs_official"] for f in top5)
    print("\n[7] per-frame official EPE, worst 5 of 40")
    print(f"{'frame':>14s} {'EPE':>8s} {'pixels':>10s}")
    for f in top5:
        print(f"{f['name']:>14s} {f['epe_official']:7.3f}p {f['n_official']:10,d}")
    print(f"top-5 frames carry {100.0 * top5_abs / S:.2f}% of total error sum")
    print("\nall 40 frames (desc):")
    for f in fs:
        print(f"  {f['name']:>14s} {f['epe_official']:7.3f}p")

    # Error-share closure checks.
    checks = {
        "region border/matched": sum(r[4] for r in t1),
        "range in/out": sum(r[4] for r in t1b),
        "err bins": sum(r[4] for r in t2),
        "disp bins": sum(r[4] for r in t3),
        "edge near/rest": sum(r[4] for r in t4),
    }
    print("\nerror-share closure (each must be 100 +- 0.1):")
    allok = True
    for k, v in checks.items():
        flag = "OK" if abs(v - 100.0) <= 0.1 else "FAIL"
        allok &= flag == "OK"
        print(f"  {k:22s} {v:7.3f}% {flag}")
    if not allok:
        print("CLOSURE CHECK FAILED -- stopping.")
        sys.exit(1)

    # JSON with both protocols.
    j = {"meta": {"ckpt": args.ckpt, "tag": args.tag, "device": device,
                  "MAX_DISP": MAX_DISP, "n_frames": len(ds)},
         "expected": EXPECTED,
         "protocols": {},
         "frames_desc": [{"name": f["name"], "n_official": f["n_official"],
                          "epe_official": f["epe_official"],
                          "n_masked": f["n_masked"],
                          "epe_masked": f["epe_masked"]} for f in fs],
         "counterfactuals_official": {
             "epe_if_d1_zero": epe_no_d1, "epe_if_inliers_zero": epe_only_d1},
         "closure_checks": checks}
    for name in ("masked", "official"):
        B = acc[name]
        Bn, Bs = B["n"], B["abs"]
        def jr(lab, n, a, extra=None):
            d = {"group": lab, "pixels": n, "pct_pixels": 100.0 * n / Bn,
                 "mean_epe": a / n if n else None,
                 "pct_error": 100.0 * a / Bs if Bs else None}
            if extra:
                d.update(extra)
            return d
        j["protocols"][name] = {
            "pixels": Bn, "epe": Bs / Bn, "d1": 100.0 * B["d1_n"] / Bn,
            "bias": B["signed"] / Bn,
            "region": [jr("border", *B["region"]["border"]),
                       jr("matched", *B["region"]["matched"])],
            "range": [jr("in_range", *B["grange"]["in_range"]),
                      jr("out_of_range", *B["grange"]["out_of_range"])],
            "err_bins": [jr(fmt_bin(*b), *v)
                         for b, v in zip(ERR_BINS, B["errbins"])],
            "d1": jr("D1_outliers", B["d1_n"], B["d1_abs"]),
            "disp_bins": [jr(disp_bin_name(*b), n, a, {"bias": sg / n if n else None})
                          for b, (n, a, sg) in zip(DISP_BINS, B["dispbins"])],
            "edge": [jr("near", *B["edge"]["near"]), jr("far", *B["edge"]["far"])],
            "counterfactuals": {
                "epe_if_d1_zero": (Bs - B["d1_abs"]) / Bn,
                "epe_if_inliers_zero": B["d1_abs"] / Bn},
        }
    outp = pathlib.Path("runs/final_candidates") / f"error_breakdown_{args.tag}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(j, indent=1))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
