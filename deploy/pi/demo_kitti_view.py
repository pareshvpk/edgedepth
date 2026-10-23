"""Pi desktop demo: run a KITTI stereo pair on the Hailo-8, print the details, show images + 3D point cloud.

Run it from a terminal ON THE PI DESKTOP (not over ssh), in the folder holding demo_kitti.py,
EdgeDepth_hailo8.hef and kitti_val40.npz:

    python3 demo_kitti_view.py --scene 000173_10
    python3 demo_kitti_view.py --scene 000173_10 --point 800 300 --point 400 250
    python3 demo_kitti_view.py --scene 5 --save          # also writes PNGs of both windows

Window 1: LEFT image, RIGHT image, Hailo-8 disparity (the readout points are numbered on them).
Window 2: 3D point cloud (drag to rotate, scroll to zoom) + bird's-eye map against LiDAR ground truth.
Close both windows to finish.
"""
import argparse
import contextlib
import io
import re

import numpy as np

import demo_kitti as dk

# 368x1232 bottom-right crop of the 375x1242 KITTI frame: cx, cy shift by 10 and 7 px.
FX, CX, CY = dk.FOCAL, 609.5593 - 10, 172.854 - 7
SURF, INK, INK2, GRID = "#f7f7f4", "#1a1a19", "#52514e", "#dedcd6"
MARK = ["#e34948", "#2a78d6", "#eda100", "#6250d6", "#1baf7a", "#eb6834"]


def cloud(disp, rgb, max_depth=60.0):
    """Pixels -> (N,3) camera-frame XYZ (X right, Y down, Z forward) + colours; drops the unmatched left border."""
    v, u = np.mgrid[0:dk.H, 0:dk.W]
    keep = (disp >= dk.FB / max_depth) & (u >= 192)
    z = dk.FB / disp[keep]
    x = (u[keep] - CX) * z / FX
    y = (v[keep] - CY) * z / FX
    return np.stack([x, y, z], 1), rgb[keep]


def to_xyz(u, v, d):
    z = dk.FB / d
    return (u - CX) * z / FX, (v - CY) * z / FX, z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hef", default="EdgeDepth_hailo8.hef")
    ap.add_argument("--pairs", default="kitti_val40.npz")
    ap.add_argument("--scene", default="000173_10", help="scene name or index 0-39 (--list shows them)")
    ap.add_argument("--point", nargs=2, type=int, action="append", metavar=("U", "V"),
                    help="pixel column/row in the 1232x368 image (repeatable)")
    ap.add_argument("--points3d", type=int, default=40000, help="points drawn in the 3D window (default 40000)")
    ap.add_argument("--save", action="store_true", help="also save both windows as PNG")
    ap.add_argument("--pred-npz", help="use saved predictions instead of the Hailo-8 (testing without the chip)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is missing on this Pi. Install it with:\n    sudo apt install -y python3-matplotlib")

    print(f"{dk.C['dim']}loading {a.pairs} ...{dk.C['0']}", flush=True)
    d = np.load(a.pairs)
    names = [str(n).replace(".png", "") for n in d["names"]]
    if a.list:
        print("  ".join(f"{i}:{n}" for i, n in enumerate(names)))
        return
    i = names.index(a.scene) if a.scene in names else int(a.scene)
    left, right, gt = d["left"][i], d["right"][i], d["disp"][i]

    if a.pred_npz:
        pred, ms = np.load(a.pred_npz)["pred"][i].astype(np.float32), float("nan")
    else:
        print(f"{dk.C['dim']}configuring Hailo-8 with {a.hef} ...{dk.C['0']}", flush=True)
        hailo = dk.Hailo(a.hef)
        try:
            hailo.run(left, right)  # warm-up
            pred, ms = hailo.run(left, right)
        finally:
            hailo.close()

    points = [tuple(p) for p in a.point] if a.point else dk.auto_points(gt)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dk.show_scene(names[i], pred, gt, ms, points)
    print(buf.getvalue(), end="")
    report = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue()).strip("\n")
    print(f"\n{dk.C['dim']}opening the image and point-cloud windows (close them to finish) ...{dk.C['0']}", flush=True)

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "font.size": 9, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "axes.titlesize": 10})
    title = f"KITTI 2015 {names[i]}  |  EdgeDepth on Hailo-8" + (f"  |  {ms:.1f} ms" if ms == ms else "")

    # ---- Window 1: stereo pair + disparity
    f1, ax = plt.subplots(3, 1, figsize=(10, 9.2), num="Window 1 - Stereo input + Hailo-8 disparity")
    f1.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=12)
    ax[0].imshow(left); ax[0].set_title("LEFT camera image")
    ax[1].imshow(right); ax[1].set_title("RIGHT camera image")
    im = ax[2].imshow(pred, cmap="magma", vmin=0, vmax=150)
    ax[2].set_title("Disparity from Hailo-8 (px): bright = near, dark = far")
    f1.colorbar(im, ax=ax[2], orientation="horizontal", fraction=0.06, pad=0.08, label="disparity (px)")
    for k, (u, v) in enumerate(points):
        for a_ in (ax[0], ax[2]):
            a_.scatter([u], [v], s=70, facecolor="none", edgecolor=MARK[k % 6], linewidth=2)
            a_.annotate(str(k + 1), (u, v), xytext=(6, -6), textcoords="offset points", color=MARK[k % 6],
                        fontweight="bold", fontsize=11)
    ax[2].axvline(192, color="white", lw=0.8, ls="--")
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    f1.tight_layout(rect=(0, 0, 1, 0.96))

    # ---- Window 2: 3D point cloud + bird's-eye map
    xyz, rgb = cloud(pred, left)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(xyz), size=min(a.points3d, len(xyz)), replace=False)
    x, y, z, c = xyz[sel, 0], xyz[sel, 1], xyz[sel, 2], rgb[sel] / 255.0
    f2 = plt.figure(figsize=(15, 10.5), num="Window 2 - 3D depth point cloud + results (Hailo-8)")
    g2 = f2.add_gridspec(2, 2, height_ratios=[2.0, 1.0], hspace=0.12, wspace=0.12)
    f2.suptitle(f"3D point cloud from stereo depth  |  {title}", x=0.01, ha="left", fontweight="bold", fontsize=12)
    a3 = f2.add_subplot(g2[0, 0], projection="3d")
    a3.scatter(x, z, -y, c=c, s=2.0, linewidths=0, depthshade=False)
    a3.set_xlim(-15, 15); a3.set_ylim(0, 45); a3.set_zlim(-2.5, 5)
    a3.set_box_aspect((30, 45, 7.5))
    a3.view_init(elev=20, azim=-65)
    a3.set_xlabel("X right (m)"); a3.set_ylabel("forward (m)"); a3.set_zlabel("up (m)")
    a3.set_title("3D view (drag to rotate, scroll to zoom), 0-45 m")
    for p_ in (a3.xaxis, a3.yaxis, a3.zaxis):
        p_.pane.set_facecolor(SURF); p_.pane.set_edgecolor(GRID)

    bev = f2.add_subplot(g2[0, 1])
    gv, gu = np.nonzero((gt > 0) & (np.arange(dk.W)[None, :] >= 192))
    gz = dk.FB / gt[gv, gu]
    gx = (gu - CX) * gz / FX
    bev.scatter(gx, gz, s=0.3, c="#9a9890", linewidths=0, label="LiDAR ground truth")
    hs = bev.scatter(x, z, c=-y, cmap="viridis", vmin=-2, vmax=3, s=0.6, linewidths=0, label="Hailo-8 point cloud")
    bev.scatter([0], [0], marker="^", s=90, c=INK, zorder=4)
    for k, (u, v) in enumerate(points):
        dd = float(pred[v, u])
        if dd >= 0.5:
            px, py, pz = to_xyz(u, v, dd)
            a3.scatter([px], [pz], [-py], s=60, c=MARK[k % 6], depthshade=False)
            bev.scatter([px], [pz], s=70, facecolor="none", edgecolor=MARK[k % 6], linewidth=2, zorder=5)
            bev.annotate(f"{k + 1}: {pz:.1f} m", (px, pz), xytext=(7, 4), textcoords="offset points",
                         color=MARK[k % 6], fontweight="bold")
    bev.set_xlim(-25, 25); bev.set_ylim(0, 60); bev.set_aspect("equal")
    bev.set_xlabel("X right (m)"); bev.set_ylabel("forward (m)")
    bev.set_title("Bird's-eye map: Hailo-8 cloud (coloured by height) vs LiDAR (grey)")
    f2.colorbar(hs, ax=bev, fraction=0.035, pad=0.02, label="height above camera (m)")
    bev.grid(color=GRID, lw=0.6)
    bev.legend(loc="upper left", fontsize=8, markerscale=12, frameon=False)
    tx = f2.add_subplot(g2[1, :])
    tx.set_position([0.03, 0.03, 0.94, 0.27])
    tx.axis("off")
    tx.text(0.0, 1.0, "Results printed by the Hailo-8 run (same as the terminal)", transform=tx.transAxes,
            fontweight="bold", fontsize=10, va="top")
    blocks = report.split("\n\n")  # [header+maps, accuracy table, point readout]
    left_txt = "\n\n".join(blocks[:2]).replace("=== ", "").replace(" ===", "")
    right_txt = "\n\n".join(blocks[2:])
    box = dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=GRID)
    tx.text(0.0, 0.86, left_txt, transform=tx.transAxes, family="monospace", fontsize=8.0, va="top", bbox=box)
    tx.text(0.50, 0.86, right_txt, transform=tx.transAxes, family="monospace", fontsize=8.0, va="top", bbox=box)

    if a.save:
        for f, tag in ((f1, "images"), (f2, "pointcloud")):
            out = f"demo_{names[i]}_{tag}.png"
            f.savefig(out, dpi=130)
            print(f"  saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
