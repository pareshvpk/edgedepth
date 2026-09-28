"""Score Pi HEF predictions against packed GT (HOST side).

Reuses deploy/dfc_flow.py score() and the pixel-weighted aggregation from
step_emulate() (acc[k][0] += e*n ... result = sum/n). Non-finite prediction
pixels are EXCLUDED from the masks (counted and reported, never silently
dropped); with finite predictions this reduces exactly to score().

Usage:
    python deploy/pi/score_preds.py --pairs <pairs.npz> \\
        --pred ours=pred_ours.npz --pred zoo=pred_zoo.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[2]  # .../hailo-stereo
sys.path.insert(0, str(REPO / "deploy"))

from dfc_flow import score  # noqa: E402  (authoritative metric)


def _parse_pred(spec: str) -> tuple[str, pathlib.Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--pred must be LABEL=path.npz, got {spec!r}")
    label, path = spec.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(
            f"--pred must be LABEL=path.npz, got {spec!r}")
    p = pathlib.Path(path)
    return label, p if p.is_absolute() else REPO / p


def _score_one(pred: np.ndarray, disp: np.ndarray):
    """Pixel-weighted aggregate over images, like step_emulate().

    Returns (agg, per_image_masked_epe, nonfinite_px) where agg maps
    'masked'/'official' -> (EPE, D1, n_pixels).
    """
    acc = {"masked": [0.0, 0.0, 0], "official": [0.0, 0.0, 0]}
    per_image_epe: list[float] = []
    nonfinite_px = 0
    for i in range(disp.shape[0]):
        p, g = pred[i], disp[i]
        bad = ~np.isfinite(p)
        n_bad = int(bad.sum())
        nonfinite_px += n_bad
        if n_bad:
            # Exclude non-finite pred pixels via the GT: disp == 0 drops out
            # of both the official (disp > 0) and masked masks in score().
            g_eff = np.where(bad, 0.0, g).astype(np.float32, copy=False)
            p_eff = np.where(bad, 0.0, p).astype(np.float32, copy=False)
            s = score(p_eff, g_eff)
        else:
            s = score(p, g)
        per_image_epe.append(float(s["masked"][0]))
        for k in acc:
            e, d, nn = s[k]
            if nn:
                acc[k][0] += e * nn
                acc[k][1] += d * nn
                acc[k][2] += nn
    agg = {k: (v[0] / v[2], v[1] / v[2], v[2]) for k, v in acc.items() if v[2]}
    return agg, per_image_epe, nonfinite_px


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="pairs npz from pack_val_pairs.py")
    ap.add_argument("--pred", action="append", required=True, metavar="LABEL=path.npz",
                    help="one or more pred npz files from run_hef.py (repeatable)")
    ap.add_argument("--out", default="deploy/pi/smoke_results.json",
                    help="results json (default: deploy/pi/smoke_results.json)")
    args = ap.parse_args()

    pairs_path = pathlib.Path(args.pairs)
    pairs_path = pairs_path if pairs_path.is_absolute() else REPO / pairs_path
    pd = np.load(pairs_path, allow_pickle=True)
    pair_names = [str(x) for x in pd["names"]]
    disp = np.asarray(pd["disp"], dtype=np.float32)
    n = len(pair_names)

    results: dict = {}
    rows = []
    for spec in args.pred:
        label, path = _parse_pred(spec)
        d = np.load(path, allow_pickle=True)
        pred_names = [str(x) for x in d["names"]]
        assert pred_names == pair_names, \
            f"{label}: pred names do not match pairs file in order " \
            f"(first mismatch: {next(((a, b) for a, b in zip(pred_names, pair_names) if a != b), None)})"
        pred = np.asarray(d["pred"], dtype=np.float32)
        assert pred.shape == disp.shape, \
            f"{label}: pred {pred.shape} != disp {disp.shape}"
        lat = np.asarray(d["latency_s"], dtype=np.float64) if "latency_s" in d else None

        agg, per_image_epe, nonfinite_px = _score_one(pred, disp)
        mean_ms = float(lat.mean() * 1000.0) if lat is not None and len(lat) else float("nan")
        fps = float(1.0 / lat.mean()) if lat is not None and len(lat) and lat.mean() > 0 else float("nan")

        entry = {
            "N": n,
            "epe_masked": float(agg["masked"][0]),
            "d1_masked": float(agg["masked"][1]),
            "epe_official": float(agg["official"][0]),
            "d1_official": float(agg["official"][1]),
            "mean_ms": mean_ms,
            "fps": fps,
            "nonfinite_px": int(nonfinite_px),
            "per_image_masked_epe": [float(x) for x in per_image_epe],
            "pred_file": str(path),
        }
        results[label] = entry
        rows.append((label, entry))

    hdr = ("label", "N", "EPE masked", "D1 masked", "EPE official",
           "D1 official", "mean ms", "FPS", "nonfinite px")
    print(f"{hdr[0]:12s} | {hdr[1]:>3s} | {hdr[2]:>10s} | {hdr[3]:>9s} | "
          f"{hdr[4]:>12s} | {hdr[5]:>11s} | {hdr[6]:>8s} | {hdr[7]:>7s} | {hdr[8]}")
    for label, e in rows:
        print(f"{label:12s} | {e['N']:3d} | {e['epe_masked']:10.3f} | "
              f"{e['d1_masked']:9.2f} | {e['epe_official']:12.3f} | "
              f"{e['d1_official']:11.2f} | {e['mean_ms']:8.1f} | {e['fps']:7.2f} | "
              f"{e['nonfinite_px']}")

    out = pathlib.Path(args.out)
    out = out if out.is_absolute() else REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
