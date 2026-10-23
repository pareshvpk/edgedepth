"""Live demo on the Pi: run one KITTI stereo pair on the Hailo-8 and print disparity, depth and accuracy.

Standalone (numpy + hailo_platform only); copy next to the HEF and kitti_val40.npz on the Pi.

    python3 demo_kitti.py --scene 000173_10                  # 3 auto-picked points (near / mid / far)
    python3 demo_kitti.py --scene 000173_10 --point 800 300  # your own pixel(s), repeatable
    python3 demo_kitti.py --scene all                        # all 40 scenes + summary
    python3 demo_kitti.py --list                             # scene names
"""
import argparse
import time

import numpy as np

H, W = 368, 1232
FOCAL, BASELINE = 721.5377, 0.54   # nominal KITTI calibration (no per-image calib packed)
FB = FOCAL * BASELINE
C = {"b": "\033[1m", "dim": "\033[2m", "g": "\033[32m", "y": "\033[33m", "c": "\033[36m", "r": "\033[31m", "0": "\033[0m"}


def depth(d):
    return FB / d if d >= 0.5 else float("inf")


def masks(gt):
    official = gt > 0
    masked = official & (gt < 192)
    masked[:, :192] = False
    return {"official": official, "masked": masked}


def image_metrics(pred, gt):
    out = {}
    for name, m in masks(gt).items():
        e = np.abs(pred - gt)[m]
        zg, zp = FB / gt[m], FB / np.maximum(pred[m], FB / 80.0)
        dm = zg <= 80
        ze = np.abs(zp - zg)[dm]
        ratio = np.maximum(zp / zg, zg / zp)[dm]
        out[name] = {"px": int(m.sum()), "epe": float(e.mean()),
                     "d1": 100.0 * float(((e > 3) & (e > 0.05 * gt[m])).mean()),
                     "rmse": float(np.sqrt((e ** 2).mean())), "zmae": float(ze.mean()),
                     "absrel": float((ze / zg[dm]).mean()), "delta1": 100.0 * float((ratio < 1.25).mean())}
    return out


def auto_points(gt):
    """One valid-GT pixel each near (~8 m), mid (~20 m) and far (~40 m), outside the unmatched border."""
    pts = []
    v, u = np.nonzero((gt > 0) & (np.arange(W)[None, :] >= 192))
    z = FB / gt[v, u]
    for target in (8.0, 20.0, 40.0):
        k = int(np.argmin(np.abs(z - target)))
        pts.append((int(u[k]), int(v[k])))
    return pts


class Hailo:
    def __init__(self, hef_path):
        from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
                                    InputVStreamParams, OutputVStreamParams, VDevice)
        self.hef = HEF(hef_path)
        names = [i.name for i in self.hef.get_input_vstream_infos()]
        self.ln = next(n for n in names if n.endswith("input_layer1"))
        self.rn = next(n for n in names if n.endswith("input_layer2"))
        self.on = self.hef.get_output_vstream_infos()[0].name
        self.vd = VDevice()
        ng = self.vd.configure(self.hef, ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe))[0]
        self.infer = InferVStreams(ng, InputVStreamParams.make(ng, format_type=FormatType.UINT8),
                                   OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32))
        self.infer.__enter__()
        self.act = ng.activate(ng.create_params())
        self.act.__enter__()

    def run(self, left, right):
        t0 = time.perf_counter()
        res = self.infer.infer({self.ln: np.ascontiguousarray(left[None]), self.rn: np.ascontiguousarray(right[None])})
        ms = (time.perf_counter() - t0) * 1000.0
        return np.squeeze(np.asarray(res[self.on], dtype=np.float32)), ms

    def close(self):
        self.act.__exit__(None, None, None)
        self.infer.__exit__(None, None, None)
        self.vd.release()


def show_scene(name, pred, gt, ms, points):
    b, d0, c, g, y, dim = C["b"], C["0"], C["c"], C["g"], C["y"], C["dim"]
    print(f"\n{b}{c}=== KITTI 2015 {name} | EdgeDepth-8 on Hailo-8 ==={d0}")
    print(f"  Hailo-8 inference      {b}{ms:7.1f} ms{d0}  ({1000.0 / ms:.2f} FPS)")
    print(f"  Disparity map          {W}x{H} px, range {pred.min():.2f} - {pred.max():.2f} px, mean {pred.mean():.2f} px")
    valid = pred[:, 192:] >= 0.5
    zz = FB / pred[:, 192:][valid]
    print(f"  Depth map              median {np.median(zz):.1f} m, 5-95% range {np.percentile(zz, 5):.1f} - {np.percentile(zz, 95):.1f} m")

    m = image_metrics(pred, gt)
    o, k = m["official"], m["masked"]
    print(f"\n{b}  Accuracy vs LiDAR ground truth ({o['px']:,} GT pixels){d0}")
    print(f"  {dim}{'':22s}{'official':>10s}{'masked':>10s}{d0}")
    for lbl, key, fmt, unit in (("EPE", "epe", "{:10.3f}", "px"), ("D1 bad pixels", "d1", "{:10.2f}", "%"),
                                ("Disparity RMSE", "rmse", "{:10.3f}", "px"), ("Depth MAE", "zmae", "{:10.3f}", "m"),
                                ("AbsRel", "absrel", "{:10.4f}", ""), ("delta < 1.25", "delta1", "{:10.2f}", "%")):
        print(f"  {lbl:22s}{fmt.format(o[key])}{fmt.format(k[key])}  {unit}")

    print(f"\n{b}  Point readout{d0}")
    print(f"  {dim}{'pixel (u,v)':>13s} {'disp px':>9s} {'depth m':>9s} {'GT disp':>9s} {'GT depth':>9s} {'disp err':>9s} {'depth err':>10s}  verdict{d0}")
    snapped = []
    for u, v in points:
        snapped.append((u, v))
        if gt[v, u] <= 0:  # LiDAR is sparse: also report the nearest pixel that has ground truth
            vv, uu = np.nonzero(gt[max(0, v - 15):v + 16, max(0, u - 15):u + 16] > 0)
            if len(uu):
                k = int(np.argmin((uu + max(0, u - 15) - u) ** 2 + (vv + max(0, v - 15) - v) ** 2))
                snapped.append((int(uu[k] + max(0, u - 15)), int(vv[k] + max(0, v - 15))))
    for u, v in snapped:
        dp, zp = float(pred[v, u]), depth(float(pred[v, u]))
        dg = float(gt[v, u])
        zps = f"{zp:9.2f}" if np.isfinite(zp) else f"{'inf':>9s}"
        if dg > 0:
            zg, de = depth(dg), abs(dp - dg)
            ze = abs(zp - zg) if np.isfinite(zp) else float("inf")
            ok = not (de > 3 and de > 0.05 * dg)
            verdict = f"{g}good{d0}" if ok else f"{C['r']}bad (D1){d0}"
            print(f"  {f'({u},{v})':>13s} {dp:9.2f} {zps} {dg:9.2f} {zg:9.2f} {de:9.2f} {ze:9.2f} m  {verdict}")
        else:
            print(f"  {f'({u},{v})':>13s} {dp:9.2f} {zps} {'-':>9s} {'-':>9s} {'-':>9s} {'-':>10s}  {y}no LiDAR here{d0}")
        if u < 192:
            print(f"  {dim}    note: u < 192 has no match in the right image (extrapolated){d0}")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hef", default="EdgeDepth-8_hailo8.hef")
    ap.add_argument("--pairs", default="kitti_val40.npz")
    ap.add_argument("--scene", default="000173_10", help="scene name, index 0-39, or 'all'")
    ap.add_argument("--point", nargs=2, type=int, action="append", metavar=("U", "V"),
                    help="pixel column/row in the 1232x368 image (repeatable)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    print(f"{C['dim']}loading {a.pairs} ...{C['0']}", flush=True)
    d = np.load(a.pairs)
    names = [str(n).replace(".png", "") for n in d["names"]]
    if a.list:
        print("  ".join(f"{i}:{n}" for i, n in enumerate(names)))
        return
    if a.scene == "all":
        idx = list(range(len(names)))
    elif a.scene in names:
        idx = [names.index(a.scene)]
    else:
        idx = [int(a.scene)]

    print(f"{C['dim']}configuring Hailo-8 with {a.hef} ...{C['0']}", flush=True)
    hailo = Hailo(a.hef)
    hailo.run(d["left"][0], d["right"][0])  # warm-up
    rows = []
    try:
        for i in idx:
            left, right, gt = d["left"][i], d["right"][i], d["disp"][i]
            pred, ms = hailo.run(left, right)
            if len(idx) == 1:
                show_scene(names[i], pred, gt, ms, [tuple(p) for p in a.point] if a.point else auto_points(gt))
            else:
                m = image_metrics(pred, gt)["official"]
                rows.append((ms, m))
                print(f"  [{i + 1:2d}/{len(idx)}] {names[i]}  {ms:6.1f} ms   EPE {m['epe']:5.3f} px   "
                      f"D1 {m['d1']:5.2f} %   depth MAE {m['zmae']:5.2f} m")
    finally:
        hailo.close()
    if rows:
        n = np.array([r[1]["px"] for r in rows], dtype=np.float64)
        avg = lambda k: float(np.sum([r[1][k] for r in rows] * n) / n.sum())  # noqa: E731
        ms = np.array([r[0] for r in rows])
        print(f"\n{C['b']}{C['c']}=== {len(rows)} KITTI scenes on Hailo-8 (official, pixel-weighted) ==={C['0']}")
        print(f"  mean latency {ms.mean():.1f} ms ({1000 / ms.mean():.2f} FPS)   EPE {avg('epe'):.3f} px   "
              f"D1 {avg('d1'):.2f} %   depth MAE {avg('zmae'):.3f} m   delta<1.25 {avg('delta1'):.2f} %")


if __name__ == "__main__":
    main()
