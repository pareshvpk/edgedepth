"""Full disparity + depth metrics for Pi HEF predictions (HOST side).

Masks are identical to deploy/dfc_flow.py score(); EPE and D1 therefore
match score_preds.py. Every metric is POOLED over all pixels of all images
(sum over images, divide once), for both the masked and official protocol.
Non-finite prediction pixels are excluded from every mask and counted.

Usage:
    python deploy/pi/full_metrics.py --pairs deploy/pi/kitti_val40.npz \\
        --pred edgedepth=deploy/pi/pred_edgedepth.npz [--pred zoo=...] \\
        [--out deploy/pi/full_metrics.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[2]  # .../hailo-stereo

# No per-image calibration ships in the repo; use the standard KITTI stereo
# focal length (1242-px-wide images) and baseline.
FOCAL_PX = 721.5377
BASELINE_M = 0.54
FB = FOCAL_PX * BASELINE_M
MAX_DEPTH_M = 80.0
BANDS = {"near": (0.0, 20.0), "medium": (20.0, 40.0), "far": (40.0, 80.0)}


def _resolve(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else REPO / p


def _parse_pred(spec: str) -> tuple[str, pathlib.Path]:
    label, sep, path = spec.partition("=")
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError(f"--pred must be LABEL=path.npz, got {spec!r}")
    return label, _resolve(path)


def _masks(gt: np.ndarray, finite_pred: np.ndarray) -> dict[str, np.ndarray]:
    official = gt > 0.0
    masked = official & (gt < 192.0) & np.isfinite(gt)
    masked[:, :192] = False
    return {"masked": masked & finite_pred, "official": official & finite_pred}


class _Acc:
    """float64 sums for one protocol (or one distance band)."""

    def __init__(self) -> None:
        self.n = 0
        self.err = 0.0
        self.err2 = 0.0
        self.d1 = 0
        self.n_z = 0
        self.zerr = 0.0
        self.zerr2 = 0.0
        self.absrel = 0.0
        self.delta = [0, 0, 0]

    def add_disp(self, err: np.ndarray, gt: np.ndarray) -> None:
        e = err.astype(np.float64)
        self.n += e.size
        self.err += e.sum()
        self.err2 += (e * e).sum()
        self.d1 += int(((e > 3.0) & (e > 0.05 * np.abs(gt))).sum())

    def add_depth(self, z_pred: np.ndarray, z_gt: np.ndarray) -> None:
        zp, zg = z_pred.astype(np.float64), z_gt.astype(np.float64)
        ze = np.abs(zp - zg)
        self.n_z += ze.size
        self.zerr += ze.sum()
        self.zerr2 += (ze * ze).sum()
        self.absrel += (ze / zg).sum()
        ratio = np.maximum(zp / zg, zg / zp)
        for k in range(3):
            self.delta[k] += int((ratio < 1.25 ** (k + 1)).sum())

    def disp_summary(self) -> dict:
        n = max(self.n, 1)
        return {"epe_px": self.err / n, "d1_pct": 100.0 * self.d1 / n,
                "disp_mae_px": self.err / n, "disp_rmse_px": float(np.sqrt(self.err2 / n))}

    def depth_summary(self) -> dict:
        n = max(self.n_z, 1)
        out = {"depth_mae_m": self.zerr / n, "depth_rmse_m": float(np.sqrt(self.zerr2 / n)),
               "abs_rel": self.absrel / n}
        for k in range(3):
            out[f"delta{k + 1}_pct"] = 100.0 * self.delta[k] / n
        return out


def _score(pred: np.ndarray, disp: np.ndarray) -> tuple[dict, int]:
    n_img, h, w = disp.shape
    total_px = n_img * h * w
    acc = {k: _Acc() for k in ("masked", "official")}
    bands = {k: {b: _Acc() for b in BANDS} for k in acc}
    cover = {k: [] for k in acc}
    beyond = {k: 0 for k in acc}
    nonfinite_px = 0
    for i in range(n_img):
        p, g = pred[i], disp[i]
        finite = np.isfinite(p)
        nonfinite_px += int((~finite).sum())
        z_gt = np.where(g > 0.0, FB / np.where(g > 0.0, g, 1.0), np.inf)
        z_pred = FB / np.maximum(np.where(finite, p, 0.0), FB / MAX_DEPTH_M)
        for k, m in _masks(g, finite).items():
            cover[k].append(100.0 * m.sum() / (h * w))
            err = np.abs(p - g)
            acc[k].add_disp(err[m], g[m])
            md = m & (z_gt <= MAX_DEPTH_M)
            beyond[k] += int((m & (z_gt > MAX_DEPTH_M)).sum())
            acc[k].add_depth(z_pred[md], z_gt[md])
            for b, (lo, hi) in BANDS.items():
                mb = md & (z_gt > lo) & (z_gt <= hi)
                bands[k][b].add_disp(err[mb], g[mb])
                bands[k][b].add_depth(z_pred[mb], z_gt[mb])

    out = {}
    for k, a in acc.items():
        out[k] = {**a.disp_summary(), **a.depth_summary(),
                  "pixels": a.n,
                  "coverage_pct": 100.0 * a.n / total_px,
                  "depth_coverage_pct": 100.0 * a.n_z / total_px,
                  "coverage_img_min_pct": float(min(cover[k])),
                  "coverage_img_max_pct": float(max(cover[k])),
                  "bands": {b: {"range_m": list(BANDS[b]), "pixels": ba.n,
                                "epe_px": ba.disp_summary()["epe_px"],
                                "d1_pct": ba.disp_summary()["d1_pct"],
                                "depth_mae_m": ba.depth_summary()["depth_mae_m"]}
                            for b, ba in bands[k].items()},
                  "beyond_80m_px": beyond[k]}
    return out, nonfinite_px


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="pairs npz from pack_val_pairs.py")
    ap.add_argument("--pred", action="append", required=True, metavar="LABEL=path.npz",
                    help="one or more pred npz files from run_hef.py (repeatable)")
    ap.add_argument("--out", default="deploy/pi/full_metrics.json",
                    help="results json (default: deploy/pi/full_metrics.json)")
    args = ap.parse_args()

    pd = np.load(_resolve(args.pairs), allow_pickle=True)
    pair_names = [str(x) for x in pd["names"]]
    disp = np.asarray(pd["disp"], dtype=np.float32)
    n = len(pair_names)

    results: dict = {}
    for spec in args.pred:
        label, path = _parse_pred(spec)
        d = np.load(path, allow_pickle=True)
        pred_names = [str(x) for x in d["names"]]
        assert pred_names == pair_names, f"{label}: pred names do not match pairs file in order"
        pred = np.asarray(d["pred"], dtype=np.float32)
        assert pred.shape == disp.shape, f"{label}: pred {pred.shape} != disp {disp.shape}"
        lat = np.asarray(d["latency_s"], dtype=np.float64) if "latency_s" in d else None

        scores, nonfinite_px = _score(pred, disp)
        results[label] = {
            "N": n,
            "mean_ms": float(lat.mean() * 1000.0) if lat is not None else float("nan"),
            "fps": float(1.0 / lat.mean()) if lat is not None else float("nan"),
            "nonfinite_px": nonfinite_px,
            "focal_px": FOCAL_PX,
            "baseline_m": BASELINE_M,
            **scores,
            "pred_file": str(path),
        }

    for proto in ("masked", "official"):
        print(f"\n== {proto} ==")
        print(f"{'label':14s} {'EPE':>6s} {'D1%':>6s} {'dMAE':>6s} {'dRMSE':>6s} "
              f"{'zMAE':>6s} {'zRMSE':>6s} {'AbsRel':>7s} {'d1%':>6s} {'d2%':>6s} {'d3%':>6s} "
              f"{'cov%':>6s} {'EPE near/med/far':>18s} {'FPS':>6s}")
        for label, r in results.items():
            s, b = r[proto], r[proto]["bands"]
            print(f"{label:14s} {s['epe_px']:6.3f} {s['d1_pct']:6.2f} {s['disp_mae_px']:6.3f} "
                  f"{s['disp_rmse_px']:6.3f} {s['depth_mae_m']:6.3f} {s['depth_rmse_m']:6.3f} "
                  f"{s['abs_rel']:7.4f} {s['delta1_pct']:6.2f} {s['delta2_pct']:6.2f} "
                  f"{s['delta3_pct']:6.2f} {s['coverage_pct']:6.2f} "
                  f"{b['near']['epe_px']:5.3f}/{b['medium']['epe_px']:5.3f}/{b['far']['epe_px']:5.3f} "
                  f"{r['fps']:6.2f}")

    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
