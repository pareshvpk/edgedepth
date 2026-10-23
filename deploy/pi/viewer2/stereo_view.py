"""Two-window stereo-depth viewer for the Hailo-8 results on KITTI.

Window 1: LEFT image, RIGHT image and Hailo-8 disparity (hover = pixel, disparity, depth; click = mark in 3D).
Window 2: the 3D depth point cloud of the same scene (light theme, rotate / zoom / pan).

    python stereo_view.py [--scene 000160_10 | --scene 0]
"""
import argparse
import pathlib
import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

REPO = pathlib.Path(__file__).resolve().parents[3]  # .../hailo-stereo
sys.path.insert(0, str(REPO / "src"))
from pointcloud import Camera, build_cloud  # noqa: E402

PAIRS = REPO / "deploy" / "pi" / "kitti_val40.npz"
PRED = REPO / "deploy" / "pi" / "pred_dense24.npz"
# Packed pairs are the 368x1232 bottom-right crop of 375x1242 KITTI frames; nominal KITTI calibration.
CAM = Camera(f=721.5377, cx=609.5593, cy=172.854, baseline=0.54).for_input(375, 1242)[0]
FB = CAM.f * CAM.baseline
DISP_MAX = 150.0
LUT = (pg.colormap.get("magma").getLookupTable(0.0, 1.0, 256) if pg.colormap.get("magma") is not None
       else np.stack([np.arange(256)] * 3, 1)).astype(np.uint8)[:, :3]


def to_pixmap(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy())


def colourize(disp):
    idx = np.clip(disp / DISP_MAX * 255.0, 0, 255).astype(np.uint8)
    return LUT[idx]


def ramp(t):
    a = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]) / 255.0
    t = np.clip(t, 0, 1) * (len(a) - 1)
    i = np.minimum(t.astype(int), len(a) - 2)
    f = (t - i)[:, None]
    return a[i] * (1 - f) + a[i + 1] * f


class ImagePane(QtWidgets.QLabel):
    """Scaled image with a caption; reports hovered / clicked pixel in source coordinates."""
    hovered = QtCore.pyqtSignal(int, int)
    clicked = QtCore.pyqtSignal(int, int)

    def __init__(self, caption):
        super().__init__()
        self.caption, self.src = caption, None
        self.setMouseTracking(True)
        self.setMinimumSize(400, 130)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def set_image(self, pix):
        self.src = pix
        self._rescale()

    def resizeEvent(self, ev):
        self._rescale()
        super().resizeEvent(ev)

    def _rescale(self):
        if self.src is None:
            return
        pix = self.src.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        f = p.font(); f.setBold(True); f.setPointSize(10); p.setFont(f)
        r = p.fontMetrics().boundingRect(self.caption).adjusted(-6, -3, 6, 3)
        r.moveTo(6, 6)
        p.setBrush(QtGui.QColor(255, 255, 255, 225)); p.setPen(QtCore.Qt.NoPen)
        p.drawRoundedRect(r, 4, 4)
        p.setPen(QtGui.QColor("#1a1a19")); p.drawText(r, QtCore.Qt.AlignCenter, self.caption)
        p.end()
        self.setPixmap(pix)

    def _to_src(self, pos):
        pm = self.pixmap()
        if pm is None or self.src is None:
            return None
        ox, oy = (self.width() - pm.width()) / 2, (self.height() - pm.height()) / 2
        u = int((pos.x() - ox) * self.src.width() / pm.width())
        v = int((pos.y() - oy) * self.src.height() / pm.height())
        if 0 <= u < self.src.width() and 0 <= v < self.src.height():
            return u, v
        return None

    def mouseMoveEvent(self, ev):
        uv = self._to_src(ev.pos())
        if uv:
            self.hovered.emit(*uv)

    def mousePressEvent(self, ev):
        uv = self._to_src(ev.pos())
        if uv:
            self.clicked.emit(*uv)


class StereoWindow(QtWidgets.QWidget):
    """Window 1: stereo pair + disparity."""

    def __init__(self, names):
        super().__init__()
        self.setWindowTitle("Window 1 — Stereo input + Hailo-8 disparity")
        self.setStyleSheet("background: #f4f4f1; color: #1a1a19;")
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("<b>Scene:</b>"))
        self.scene = QtWidgets.QComboBox()
        self.scene.addItems(names)
        self.scene.setStyleSheet("background: white;")
        top.addWidget(self.scene)
        top.addStretch(1)
        lay.addLayout(top)
        self.left = ImagePane("LEFT camera image")
        self.right = ImagePane("RIGHT camera image")
        self.disp = ImagePane("Disparity from Hailo-8 (px)")
        for w in (self.left, self.right, self.disp):
            lay.addWidget(w, 1)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("0 px (far)"))
        grad = QtWidgets.QLabel()
        grad.setFixedHeight(14)
        grad.setScaledContents(True)
        grad.setPixmap(to_pixmap(np.repeat(LUT[None], 14, axis=0)))
        bar.addWidget(grad, 1)
        bar.addWidget(QtWidgets.QLabel(f"{DISP_MAX:.0f} px (near)"))
        lay.addLayout(bar)
        self.readout = QtWidgets.QLabel("Hover over an image to read disparity and depth; click to mark the point in 3D.")
        self.readout.setStyleSheet("font-family: Consolas, monospace; font-size: 12px; padding: 4px;"
                                   "background: white; border: 1px solid #c9c8c2; border-radius: 4px;")
        lay.addWidget(self.readout)
        self.resize(900, 820)


class CloudWindow(QtWidgets.QWidget):
    """Window 2: 3D point cloud."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Window 2 — 3D depth point cloud (Hailo-8)")
        self.setStyleSheet("QWidget { background: #f4f4f1; color: #1a1a19; } "
                           "QComboBox, QPushButton { background: white; }")
        self.xyz = self.rgb = None
        lay = QtWidgets.QVBoxLayout(self)
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("Colour by"))
        self.colour = QtWidgets.QComboBox()
        self.colour.addItems(["Camera colours (RGB)", "Height", "Distance"])
        ctl.addWidget(self.colour)
        self.size_lbl = QtWidgets.QLabel()
        ctl.addWidget(self.size_lbl)
        self.size = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.size.setRange(1, 8); self.size.setValue(2)
        self.size.setFixedWidth(90); ctl.addWidget(self.size)
        self.range_lbl = QtWidgets.QLabel()
        ctl.addWidget(self.range_lbl)
        self.range = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.range.setRange(5, 80); self.range.setValue(60)
        self.range.setFixedWidth(110); ctl.addWidget(self.range)
        self.hide_near = QtWidgets.QCheckBox("Hide < 4 m (sky artefacts)")
        ctl.addWidget(self.hide_near)
        reset = QtWidgets.QPushButton("Reset view"); reset.clicked.connect(self.reset_camera); ctl.addWidget(reset)
        bev = QtWidgets.QPushButton("Bird's-eye")
        bev.clicked.connect(lambda: self.view.setCameraPosition(pos=pg.Vector(0, 30, 0), distance=75,
                                                                elevation=90, azimuth=-90))
        ctl.addWidget(bev)
        ctl.addStretch(1)
        lay.addLayout(ctl)

        self.title = QtWidgets.QLabel()
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self.title)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("#f4f4f1")
        grid = gl.GLGridItem(); grid.setSize(60, 80); grid.setSpacing(5, 5)
        grid.translate(0, 40, -1.7); grid.setColor((0, 0, 0, 55))
        self.view.addItem(grid)
        self.view.addItem(gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(0.16, 0.47, 0.84, 1), size=12))
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=2, pxMode=True)
        self.scatter.setGLOptions("opaque")  # additive blending washes overlapping points to white
        self.view.addItem(self.scatter)
        self.marker = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(0.89, 0.29, 0.28, 1), size=16)
        self.marker.setGLOptions("opaque")
        self.marker.setVisible(False)
        self.view.addItem(self.marker)
        lay.addWidget(self.view, 1)
        self.info = QtWidgets.QLabel()
        lay.addWidget(self.info)
        help_ = QtWidgets.QLabel("<small>Left-drag rotate · wheel zoom · middle-drag / Ctrl+left-drag pan · "
                                 "blue dot = camera · red dot = pixel clicked in Window 1 · axes in metres</small>")
        lay.addWidget(help_)
        for w in (self.colour,):
            w.currentIndexChanged.connect(self.redraw)
        for w in (self.size, self.range):
            w.valueChanged.connect(self.redraw)
        self.hide_near.toggled.connect(self.redraw)
        self.reset_camera()
        self.resize(1100, 820)

    def set_cloud(self, xyz, rgb, name):
        self.xyz = np.stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]], axis=1)  # camera frame -> (X, forward, up)
        self.rgb = rgb
        self.title.setText(f"<b style='font-size:14px'>3D point cloud from stereo depth</b> &nbsp; "
                           f"Hailo-8 (EdgeDepth-8) · KITTI {name} · left + right → disparity → 3D")
        self.marker.setVisible(False)
        self.redraw()

    def mark(self, p):
        self.marker.setData(pos=np.array([[p[0], p[2], -p[1]]], dtype=np.float32))
        self.marker.setVisible(True)

    def redraw(self, *_):
        self.size_lbl.setText(f"Size {self.size.value()}")
        self.range_lbl.setText(f"Max {self.range.value()} m")
        if self.xyz is None:
            return
        d = self.xyz[:, 1]
        keep = d <= self.range.value()
        if self.hide_near.isChecked():
            keep &= d >= 4.0
        pos = self.xyz[keep]
        mode = self.colour.currentIndex()
        if mode == 0:
            col = self.rgb[keep] / 255.0
        elif mode == 2:
            col = ramp(1 - pos[:, 1] / self.range.value())
        else:
            col = ramp((pos[:, 2] + 2.0) / 5.0)
        col = np.concatenate([col, np.ones((len(col), 1))], axis=1).astype(np.float32)
        self.scatter.setData(pos=pos, color=col, size=self.size.value())
        self.info.setText(f"{len(pos):,} of {len(self.xyz):,} points shown "
                          f"(left 192 columns have no stereo match and are dropped)")

    def reset_camera(self):
        self.view.setCameraPosition(pos=pg.Vector(0, 20, 1), distance=44, elevation=16, azimuth=-90)


class App:
    def __init__(self, start):
        print("loading KITTI pairs and Hailo-8 predictions ...", flush=True)
        pairs = np.load(PAIRS)
        self.names = [str(n).replace(".png", "") for n in pairs["names"]]
        self.left, self.right = pairs["left"], pairs["right"]
        self.pred = np.load(PRED)["pred"].astype(np.float32)
        self.w1 = StereoWindow(self.names)
        self.w2 = CloudWindow()
        self.w1.scene.currentIndexChanged.connect(self.load)
        self.w1.left.hovered.connect(self.hover); self.w1.right.hovered.connect(self.hover)
        self.w1.disp.hovered.connect(self.hover)
        self.w1.left.clicked.connect(self.click); self.w1.disp.clicked.connect(self.click)
        self.w1.right.clicked.connect(self.click)
        scr = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.w1.setGeometry(scr.x(), scr.y() + 30, int(scr.width() * 0.42), scr.height() - 40)
        self.w2.setGeometry(scr.x() + int(scr.width() * 0.42), scr.y() + 30, int(scr.width() * 0.58), scr.height() - 40)
        self.w1.show(); self.w2.show()
        i = self.names.index(start) if start in self.names else (int(start) if start and start.isdigit() else 0)
        self.w1.scene.setCurrentIndex(i)
        if i == 0:
            self.load(0)

    def load(self, i):
        self.i = i
        d = self.pred[i]
        self.w1.left.set_image(to_pixmap(self.left[i]))
        self.w1.right.set_image(to_pixmap(self.right[i]))
        self.w1.disp.set_image(to_pixmap(colourize(d)))
        xyz, rgb, _ = build_cloud(d, self.left[i], CAM, max_depth=80.0, drop_border=True)
        self.w2.set_cloud(xyz, rgb, self.names[i])

    def _point(self, u, v):
        d = float(self.pred[self.i][v, u])
        z = FB / d if d >= 0.5 else float("inf")
        return d, z

    def hover(self, u, v):
        d, z = self._point(u, v)
        zs = f"{z:6.2f} m" if np.isfinite(z) else "     ∞"
        note = "  (no stereo match: left 192 columns)" if u < 192 else ""
        self.w1.readout.setText(f"u={u:4d}  v={v:3d}  |  disparity {d:6.2f} px  |  depth Z {zs}{note}")

    def click(self, u, v):
        d, z = self._point(u, v)
        if not np.isfinite(z):
            return
        x = (u - CAM.cx) * z / CAM.f
        y = (v - CAM.cy) * z / CAM.f
        self.w2.mark((x, y, z))
        self.hover(u, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="000160_10", help="scene name (e.g. 000173_10) or index 0-39")
    args = ap.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    _ = App(args.scene)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
