"""
Turn predicted disparity into a coloured 3D point cloud (.ply).

    # run the PyTorch model on a stereo pair
    python src/pointcloud.py --ckpt runs/kitti_mixed_fixed768/best.pt \
        --left samples/kitti_000160_10_LEFT.png --right samples/kitti_000160_10_RIGHT.png

    # or consume a disparity map produced elsewhere (HEF / emulator), [368,1232] .npy
    python src/pointcloud.py --disp pred.npy --left samples/kitti_000160_10_LEFT.png

The network predicts disparity -- how far each left pixel moved in the right
image -- never metres. Three lines of pinhole geometry turn that into 3D:

    Z = f x B / d            depth along the optical axis
    X = (u - cx) x Z / f     right of the optical centre
    Y = (v - cy) x Z / f     below the optical centre

so every pixel (u, v) becomes a point (X, Y, Z) in metres, in the left
camera's frame: X right, Y down, Z forward.

This runs on the HOST, after inference, deliberately. On a Hailo-15H it belongs
on the ARM core: ~450K divisions per frame is negligible there, division is not
a native NPU op, and int8 is a poor fit for 1/d (it needs centimetres near and
tolerates metres far). Keeping it out of the HEF also keeps the HEF
camera-agnostic -- a new camera is four numbers here, not a recompile.

Calibration is the whole game. f, B, cx, cy default to NOMINAL KITTI values,
the same f x B = 386.5 px.m src/eval_depth.py uses -- KITTI's calib files are
not in this checkout, and KITTI 2015 frames come from several drives with
slightly different intrinsics. On any other camera, pass its own values: a
wrong f or B scales every depth by the same factor, a wrong cx/cy shears the
cloud sideways.

The input crop moves the optical centre. The graph takes exactly 368x1232;
a larger image is cropped anchored bottom-right (as src/eval_kitti.py and
deploy/dfc_flow.py do), which drops columns from the left and rows from the
top, so cx and cy shift by those amounts. An image of any other size is first
resized to 1232 wide (as src/demo_server.py does), which scales f, cx, cy.
Camera.for_input() carries both through, so the 3D is right on the pixels the
model actually saw.

Known limits, inherited from the model rather than added here:
  - Sky / above the horizon is untrained and comes out as a spurious near
    surface. --max-entropy (PyTorch path only) drops pixels whose matching was
    uncertain. The HEF does not export that confidence, so on-device output
    cannot be filtered this way yet.
  - The left MAX_DISP columns have no match in the right image and are
    extrapolated (weak border supervision). --drop-border removes them.
  - Near range (0-5 m, disparity >= ~80 px) is under-read: near objects come
    out too far.

Only numpy, torch and Pillow, matching the rest of src/.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

HEIGHT, WIDTH = 368, 1232      # the graph's fixed input, same as the HEF
MAX_DISP = 192                 # model.MAX_DISP; duplicated so --disp needs no torch

# Nominal KITTI: the same f x B = 386.5 px.m as src/eval_depth.py. cx, cy are the
# common KITTI raw-drive values for the full 1242x375 frame.
KITTI_FOCAL = 715.7            # px
KITTI_BASELINE = 0.54          # m
KITTI_CX = 609.56              # px, full frame
KITTI_CY = 172.85              # px, full frame

# Z is only resolvable down to about half a pixel of disparity; below that a
# pixel is at "infinity" as far as this rig is concerned.
MIN_DISP = 0.5


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics plus stereo baseline, in the image's own pixels."""
    f: float
    cx: float
    cy: float
    baseline: float

    def for_input(self, orig_h: int, orig_w: int) -> tuple["Camera", str]:
        """The same camera, re-expressed in the 368x1232 frame the model sees.

        Mirrors prepare() exactly: bottom-right crop if the image is big enough,
        else resize to WIDTH then bottom crop.
        """
        if orig_h >= HEIGHT and orig_w >= WIDTH:
            dy, dx = orig_h - HEIGHT, orig_w - WIDTH
            return (dataclasses.replace(self, cx=self.cx - dx, cy=self.cy - dy),
                    f"bottom-right crop, cx -{dx} px, cy -{dy} px")
        s = WIDTH / orig_w
        new_h = max(HEIGHT, round(orig_h * s))
        dy = new_h - HEIGHT
        return (Camera(f=self.f * s, cx=self.cx * s, cy=self.cy * s - dy,
                       baseline=self.baseline),
                f"resized x{s:.4f} then bottom crop, cy -{dy} px")


def disparity_to_depth(disp: np.ndarray, cam: Camera) -> np.ndarray:
    """Z = f x B / d. Pixels below MIN_DISP come back as +inf."""
    z = np.full(disp.shape, np.inf, dtype=np.float32)
    ok = disp >= MIN_DISP
    z[ok] = cam.f * cam.baseline / disp[ok]
    return z


def depth_to_points(z: np.ndarray, cam: Camera) -> np.ndarray:
    """Back-project every pixel: [H,W] depth -> [H,W,3] XYZ in metres."""
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (u - cam.cx) * z / cam.f
    y = (v - cam.cy) * z / cam.f
    return np.stack([x, y, z], axis=-1)


def build_cloud(disp, rgb_u8, cam, max_depth=80.0, drop_border=False,
                entropy=None, max_entropy=None):
    """Disparity + colour -> (N,3) float32 XYZ, (N,3) uint8 RGB, stats dict."""
    z = disparity_to_depth(disp, cam)
    keep = np.isfinite(z) & (z <= max_depth)
    stats = {"pixels": int(disp.size),
             "beyond_range": int((~keep).sum())}
    if drop_border:
        border = np.zeros_like(keep)
        border[:, :MAX_DISP] = True
        stats["border_dropped"] = int((keep & border).sum())
        keep &= ~border
    if entropy is not None and max_entropy is not None:
        unsure = entropy > max_entropy
        stats["uncertain_dropped"] = int((keep & unsure).sum())
        keep &= ~unsure
    xyz = depth_to_points(z, cam)[keep]
    stats["points"] = int(keep.sum())
    return xyz.astype(np.float32), rgb_u8[keep], stats


def write_ply(path: pathlib.Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Binary little-endian PLY; opens in MeshLab, CloudCompare, Open3D."""
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(rec)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(rec.tobytes())


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

def prepare(img: Image.Image) -> np.ndarray:
    """Any image -> the graph's 368x1232 uint8 RGB. Pair with Camera.for_input."""
    img = img.convert("RGB")
    if img.height >= HEIGHT and img.width >= WIDTH:
        return np.asarray(img.crop((img.width - WIDTH, img.height - HEIGHT,
                                    img.width, img.height)), dtype=np.uint8)
    s = WIDTH / img.width
    img = img.resize((WIDTH, max(HEIGHT, round(img.height * s))), Image.BILINEAR)
    return np.asarray(img.crop((0, img.height - HEIGHT, WIDTH, img.height)),
                      dtype=np.uint8)


def run_model(ckpt, left_u8, right_u8, want_entropy):
    """PyTorch inference; optionally the matcher's normalised softmax entropy.

    Entropy is read off the aggregation output with a hook -- the same
    softmax SoftArgmin applies -- and divided by ln(D), so 0 is a single sharp
    match and 1 is no preference at all. Upsampled nearest x8 to full res.
    """
    import torch
    import torch.nn.functional as F
    from model import HailoStereo
    from data import MEAN, STD

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = HailoStereo()
    model.load_state_dict(blob.get("model", blob), strict=True)   # EXP-1
    model.to(device).eval()

    captured = {}
    if want_entropy:
        model.aggregation.register_forward_hook(
            lambda m, i, o: captured.__setitem__("cost", o))

    def tensor(u8):
        x = (u8.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)

    with torch.no_grad():
        disp = model(tensor(left_u8), tensor(right_u8))[0, 0].cpu().numpy()
        entropy = None
        if want_entropy:
            cost = captured["cost"]
            prob = F.softmax(-cost * model.soft_argmin.temperature(), dim=1)
            ent = -(prob * prob.clamp_min(1e-12).log()).sum(1, keepdim=True)
            ent = F.interpolate(ent / math.log(prob.shape[1]),
                                scale_factor=8, mode="nearest")
            entropy = ent[0, 0].cpu().numpy()
    return disp, entropy


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="PyTorch checkpoint; needs --right")
    src.add_argument("--disp", help="precomputed [368,1232] disparity .npy (HEF/emulator)")
    p.add_argument("--left", required=True, help="left image (colours the cloud)")
    p.add_argument("--right", help="right image, with --ckpt")
    p.add_argument("--out", default=None, help="default: artifacts/<left stem>.ply")
    p.add_argument("--focal", type=float, default=KITTI_FOCAL, help="px, full frame")
    p.add_argument("--baseline", type=float, default=KITTI_BASELINE, help="m")
    p.add_argument("--cx", type=float, default=KITTI_CX, help="px, full frame")
    p.add_argument("--cy", type=float, default=KITTI_CY, help="px, full frame")
    p.add_argument("--max-depth", type=float, default=80.0,
                   help="m; drop farther points (KITTI GT stops at ~80 m)")
    p.add_argument("--drop-border", action="store_true",
                   help=f"drop the left {MAX_DISP} columns (no stereo match exists)")
    p.add_argument("--max-entropy", type=float, default=None,
                   help="0..1; drop uncertain matches, e.g. 0.6 (--ckpt only)")
    p.add_argument("--save-depth", action="store_true",
                   help="also write the [368,1232] depth map as .npy")
    args = p.parse_args()

    if args.ckpt and not args.right:
        p.error("--ckpt needs --right")
    if args.max_entropy is not None and not args.ckpt:
        p.error("--max-entropy needs --ckpt: the HEF does not export match confidence")

    left_img = Image.open(args.left)
    left_u8 = prepare(left_img)
    cam, how = Camera(args.focal, args.cx, args.cy, args.baseline) \
        .for_input(left_img.height, left_img.width)

    entropy = None
    if args.ckpt:
        right_u8 = prepare(Image.open(args.right))
        disp, entropy = run_model(args.ckpt, left_u8, right_u8,
                                  want_entropy=args.max_entropy is not None)
    else:
        disp = np.load(args.disp).astype(np.float32).squeeze()
        if disp.shape != (HEIGHT, WIDTH):
            p.error(f"--disp must be {HEIGHT}x{WIDTH}, got {disp.shape}")

    xyz, rgb, stats = build_cloud(disp, left_u8, cam, args.max_depth,
                                  args.drop_border, entropy, args.max_entropy)

    out = pathlib.Path(args.out or f"artifacts/{pathlib.Path(args.left).stem}.ply")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out, xyz, rgb)
    if args.save_depth:
        np.save(out.with_suffix(".depth.npy"), disparity_to_depth(disp, cam))

    print(f"input      {left_img.width}x{left_img.height} -> {WIDTH}x{HEIGHT} ({how})")
    print(f"camera     f={cam.f:.2f} px  cx={cam.cx:.2f}  cy={cam.cy:.2f}  "
          f"B={cam.baseline:.3f} m  (f x B = {cam.f * cam.baseline:.1f} px.m)")
    print(f"disparity  {disp.min():.2f} .. {disp.max():.2f} px")
    for k, v in stats.items():
        print(f"{k:<18} {v:>9,}  ({100 * v / stats['pixels']:.1f}%)")
    if len(xyz):
        lo, hi = xyz.min(0), xyz.max(0)
        print(f"extent     X {lo[0]:.1f}..{hi[0]:.1f}  Y {lo[1]:.1f}..{hi[1]:.1f}  "
              f"Z {lo[2]:.1f}..{hi[2]:.1f} m")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
