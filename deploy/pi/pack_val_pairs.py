"""Pack KITTI val pairs for Pi on-device smoke test (HOST side).

Uses deploy/dfc_flow.py kitti_val_pairs() as the authoritative source of the
40-pair split and crop. Saves raw uint8 stereo pairs + float32 GT disparity
so the Pi just runs the HEF and the host does all scoring.

Usage:
    python deploy/pi/pack_val_pairs.py [--root ...] [--out ...] [--limit N]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[2]  # .../hailo-stereo
sys.path.insert(0, str(REPO / "deploy"))

from dfc_flow import kitti_val_pairs  # noqa: E402  (authoritative split/crop)


def _resolve(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else REPO / p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/kitti2015/training",
                    help="KITTI 2015 training dir (default: data/kitti2015/training)")
    ap.add_argument("--out", default="deploy/pi/kitti_val40.npz",
                    help="output npz (default: deploy/pi/kitti_val40.npz)")
    ap.add_argument("--limit", type=int, default=None,
                    help="pack only the first N val pairs (default: None = all 40)")
    args = ap.parse_args()

    root = _resolve(args.root)
    out = _resolve(args.out)

    names, lefts, rights, disps = [], [], [], []
    for name, left, right, disp in kitti_val_pairs(root, limit=args.limit):
        assert left.shape == (368, 1232, 3) and left.dtype == np.uint8, \
            f"{name}: left {left.shape} {left.dtype}"
        assert right.shape == (368, 1232, 3) and right.dtype == np.uint8, \
            f"{name}: right {right.shape} {right.dtype}"
        assert disp.shape == (368, 1232), f"{name}: disp {disp.shape}"
        names.append(name)
        lefts.append(left)
        rights.append(right)
        disps.append(disp.astype(np.float32, copy=False))

    left = np.stack(lefts).astype(np.uint8, copy=False)
    right = np.stack(rights).astype(np.uint8, copy=False)
    disp = np.stack(disps).astype(np.float32, copy=False)
    names_arr = np.array(names)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=names_arr, left=left, right=right, disp=disp)
    print(f"N={len(names)}  names={names_arr.shape} {names_arr.dtype}  "
          f"left={left.shape} {left.dtype}  right={right.shape} {right.dtype}  "
          f"disp={disp.shape} {disp.dtype}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
