"""Run a stereo HEF on the Pi over packed KITTI val pairs (PI side).

Only numpy + hailo_platform are used -- deliberately no imports from this
repo, so the file can be copied to the Pi standalone.

Both HEFs take two UINT8 NHWC inputs (368x1232x3, RAW uint8 RGB --
normalization is inside the HEF) and produce one UINT8 disparity map
(368x1232x1). HailoRT dequantizes the output to float32 here.

Usage (on the Pi):
    python run_hef.py --hef <model.hef> --pairs kitti_val40.npz \\
        --out pred_<model>.npz [--warmup 2]
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

H, W, C = 368, 1232, 3


def _pick_streams(hef):
    """Map input vstreams to LEFT/RIGHT by the `input_layer{1,2}` suffix."""
    infos = hef.get_input_vstream_infos()
    names = [i.name for i in infos]
    left = next((n for n in names if n.endswith("input_layer1")), None)
    right = next((n for n in names if n.endswith("input_layer2")), None)
    if left is None or right is None:
        raise RuntimeError(
            f"need one input vstream ending in 'input_layer1' (LEFT) and one "
            f"ending in 'input_layer2' (RIGHT); HEF has: {names}")
    # Verify the HEF-side frame shape is (368, 1232, 3).
    for info in infos:
        shape = tuple(info.shape)
        if shape != (H, W, C):
            raise RuntimeError(
                f"input vstream {info.name!r} has shape {shape}, "
                f"expected ({H}, {W}, {C})")
    out_infos = hef.get_output_vstream_infos()
    if len(out_infos) != 1:
        raise RuntimeError(
            f"expected exactly 1 output vstream, HEF has: "
            f"{[i.name for i in out_infos]}")
    return left, right, out_infos[0].name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hef", required=True, help="path to the .hef file")
    ap.add_argument("--pairs", required=True, help="input npz from pack_val_pairs.py")
    ap.add_argument("--out", required=True, help="output pred npz")
    ap.add_argument("--warmup", type=int, default=2,
                    help="untimed warmup frames (default: 2)")
    args = ap.parse_args()

    # Imported here, after parsing, so --help works even where
    # hailo_platform is not installed, e.g. on the host.
    from hailo_platform import (
        HEF,
        ConfigureParams,
        FormatType,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )

    d = np.load(args.pairs, allow_pickle=True)
    names, left, right = d["names"], d["left"], d["right"]
    n = len(names)
    if left.shape != (n, H, W, C) or left.dtype != np.uint8:
        raise RuntimeError(f"left: {left.shape} {left.dtype}, expected ({n}, {H}, {W}, {C}) uint8")
    if right.shape != (n, H, W, C) or right.dtype != np.uint8:
        raise RuntimeError(f"right: {right.shape} {right.dtype}, expected ({n}, {H}, {W}, {C}) uint8")

    hef = HEF(args.hef)
    left_name, right_name, out_name = _pick_streams(hef)

    preds = np.empty((n, H, W), dtype=np.float32)
    latency = np.empty((n,), dtype=np.float64)

    with VDevice() as vdevice:
        cfg = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe)
        ng = vdevice.configure(hef, cfg)[0]
        in_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(
            ng, format_type=FormatType.FLOAT32)
        with InferVStreams(ng, in_params, out_params) as infer:
            with ng.activate(ng.create_params()):
                for _ in range(args.warmup):
                    infer.infer({left_name: np.ascontiguousarray(left[0][None]),
                                 right_name: np.ascontiguousarray(right[0][None])})
                for i in range(n):
                    frame = {left_name: np.ascontiguousarray(left[i][None]),
                             right_name: np.ascontiguousarray(right[i][None])}
                    t0 = time.perf_counter()
                    res = infer.infer(frame)
                    latency[i] = time.perf_counter() - t0
                    out = res[out_name]
                    preds[i] = np.squeeze(np.asarray(out, dtype=np.float32))
                    print(f"{names[i]}: {latency[i]*1000:.1f} ms")

    if preds.shape != (n, H, W):
        raise RuntimeError(f"pred shape {preds.shape}, expected ({n}, {H}, {W})")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, names=names, pred=preds,
             latency_s=latency.astype(np.float64))

    mean_s = float(latency.mean())
    finite = preds[np.isfinite(preds)]
    print(f"mean {mean_s*1000:.1f} ms  FPS {1.0/mean_s:.2f}  "
          f"pred min/max/mean {float(finite.min()):.3f}/{float(finite.max()):.3f}/"
          f"{float(finite.mean()):.3f}  "
          f"nonfinite px {int((~np.isfinite(preds)).sum())}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
