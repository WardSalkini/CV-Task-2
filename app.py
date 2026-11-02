"""
Computer Vision Workbench - frontend (Streamlit).      Run:  py -m streamlit run app.py
Backend: engine.cpp (compile it first, it produces engine.dll / engine.so next to this file).

WHERE IS MY PART?  (search for the tag; engine.cpp uses the same tags)
  SETUP                library loading and small helpers used by everyone
  TASK 1 - PERSON 1    Canny edge detector      : Canny controls, "Canny steps" tab
  TASK 1 - PERSON 2    Hough lines/circles/ellipses : Hough controls, "Hough accumulators" and "Data" tabs
  TASK 2 - PERSON 3    Snake (greedy active contour) : initial contour, snake parameters, "Snake" tab
  TASK 2 - PERSON 4    Chain code, perimeter, area   : "Chain code" and "Measurements" tabs
  COMPARE              pin one result, change a setting, see both side by side
  MAIN                 page layout: one shared image, a part selector, settings on the left, results on the right
"""
import base64
import csv
import ctypes
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import zipfile

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

st.set_page_config(layout="wide", page_title="Computer Vision Workbench")

# =====================================================================
# SETUP: load the C++ engine
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.name == "nt":
    _mingw = os.environ.get("MINGW64_BIN", r"C:\msys64\mingw64\bin")
    if os.path.isdir(_mingw):
        os.add_dll_directory(_mingw)
    LIB_NAME = "engine.dll"
elif sys.platform == "darwin":
    LIB_NAME = "engine.dylib"
else:
    LIB_NAME = "engine.so"
LIB_PATH = os.path.join(BASE_DIR, LIB_NAME)

if not os.path.isfile(LIB_PATH):
    st.error(f"{LIB_NAME} not found next to app.py. Build engine.cpp first.")
    st.code("cmake --build .        (or, from the x64 Native Tools Command Prompt)\n"
            "cl /O2 /EHsc /std:c++17 /LD engine.cpp /Fe:engine.dll", language="text")
    st.stop()
try:
    lib = ctypes.CDLL(LIB_PATH)
except OSError as err:
    st.error(f"Could not load {LIB_NAME}: {err}")
    st.stop()

U8P = ctypes.POINTER(ctypes.c_uint8)
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int)
F64P = ctypes.POINTER(ctypes.c_double)
c_int, c_float = ctypes.c_int, ctypes.c_float


def u8(array):
    return array.ctypes.data_as(U8P)


def f32(array):
    return array.ctypes.data_as(F32P)


# =====================================================================
# SETUP: image helpers (Pillow for reading files, NumPy for everything else)
# Images are uint8, gray (H, W) or BGR (H, W, 3), the layout the engine expects.
# =====================================================================
def load_image(data):
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    if img.mode in ("L", "1", "I;16", "I", "F"):
        return np.ascontiguousarray(np.asarray(img.convert("L"), np.uint8))
    return np.ascontiguousarray(np.asarray(img.convert("RGB"), np.uint8)[..., ::-1])


def encode_png(image):
    arr = np.asarray(image, np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr[..., ::-1] if arr.ndim == 3 else arr).save(buf, format="PNG")
    return buf.getvalue()


def to_bgr(image):
    return np.ascontiguousarray(np.stack([image] * 3, axis=-1)) if image.ndim == 2 else image.copy()


def stamp(img, xs, ys, color, radius=0):
    """Paint a small square at every (x, y)."""
    h, w = img.shape[:2]
    xs, ys = np.asarray(xs, np.int64), np.asarray(ys, np.int64)
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            x, y = xs + ox, ys + oy
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            img[y[ok], x[ok]] = color


def draw_contour(image, contour, color=(0, 255, 0), thickness=2, points=False):
    """BGR copy of the image with a closed (N, 2) contour drawn on it."""
    vis = to_bgr(image)
    pts = np.asarray(contour, np.float64).reshape(-1, 2)
    nxt = np.roll(pts, -1, axis=0)
    xs, ys = [], []
    for a, b in zip(pts, nxt):
        steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        t = np.linspace(0.0, 1.0, steps)
        xs.append(np.rint(a[0] + t * (b[0] - a[0])))
        ys.append(np.rint(a[1] + t * (b[1] - a[1])))
    stamp(vis, np.concatenate(xs), np.concatenate(ys), color, max(0, (thickness - 1) // 2))
    if points:
        p = np.rint(pts).astype(np.int64)
        stamp(vis, p[:, 0], p[:, 1], (0, 0, 255), max(1, thickness // 2))
    return vis


def colormap(gray):
    """uint8 (H, W) -> BGR heat map (black, purple, orange, yellow)."""
    t = np.asarray(gray, np.float64) / 255.0
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    rgb = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], np.float64)
    out = np.stack([np.interp(t, stops, rgb[:, c]) for c in range(3)], axis=-1)
    return np.ascontiguousarray(np.rint(out[..., ::-1]).astype(np.uint8))


def show(image, caption=None):
    kwargs = dict(caption=caption, channels="GRAY" if image.ndim == 2 else "BGR", clamp=True)
    try:
        st.image(image, width="stretch", **kwargs)
    except TypeError:  # older Streamlit
        st.image(image, use_container_width=True, **kwargs)


def table(rows):
    st.dataframe(rows, hide_index=True)


# ---------------------------------------------------------------------
# Settings widgets. Each value lives in st.session_state under a "k_..." key (seeded once, no default is
# handed to Streamlit). keep_settings() re-assigns them on every run, so a setting survives when you switch
# to the other part and come back.
# ---------------------------------------------------------------------
def _key(label, suffix=""):
    return "k_" + re.sub(r"\W+", "_", label.lower()).strip("_") + suffix


def _seed(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def slider(label, lo, hi, value, step=None, key=None, **kw):
    key = key or _key(label)
    _seed(key, value)
    return st.slider(label, lo, hi, step=step, key=key, **kw)


def select_slider(label, options, value, key=None, **kw):
    key = key or _key(label)
    _seed(key, value)
    return st.select_slider(label, options, key=key, **kw)


def number_input(label, lo, hi, value, key=None, **kw):
    key = key or _key(label)
    _seed(key, value)
    return st.number_input(label, lo, hi, key=key, **kw)


def checkbox(label, value=False, key=None, **kw):
    key = key or _key(label)
    _seed(key, value)
    return st.checkbox(label, key=key, **kw)


def selectbox(label, options, key=None, **kw):
    return st.selectbox(label, options, key=key or _key(label), **kw)


def radio(label, options, key=None, **kw):
    return st.radio(label, options, key=key or _key(label), **kw)


def keep_settings():
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("k_"):
            st.session_state[key] = st.session_state[key]


# =====================================================================
# TASK 1 - PERSON 1 : CANNY EDGE DETECTOR
# Backend: engine.cpp, section "TASK 1 - PERSON 1" (canny_run, edge_similarity)
# =====================================================================
lib.canny_run.restype = None
lib.canny_run.argtypes = [
    U8P, c_int, c_int, c_int, c_int,
    c_int, c_float, c_float, c_float, c_int,
    U8P, U8P, U8P, U8P, U8P, F32P,
]
lib.edge_similarity.restype = c_float
lib.edge_similarity.argtypes = [U8P, U8P, c_int]


def canny_stages(image, c):
    """The four Canny stages as images: smoothing, gradient magnitude, non-maximum suppression, final edges."""
    h, w = image.shape[:2]
    smooth, magnitude, nms, edges = (np.empty((h, w), np.uint8) for _ in range(4))
    overlay = np.empty((h, w, 3), np.uint8)
    info = np.zeros(4, np.float32)
    lib.canny_run(u8(image), h, w, 1 if image.ndim == 2 else 3, 0, int(c["ksize"]), float(c["sigma"]),
                  float(c["low"]), float(c["high"]), int(c["auto"]),
                  u8(smooth), u8(magnitude), u8(nms), u8(edges), u8(overlay), f32(info))
    return {"smooth": smooth, "magnitude": magnitude, "nms": nms, "edges": edges}


def canny_controls():
    st.subheader("Canny")
    ksize = select_slider("Gaussian kernel", [3, 5, 7, 9, 11], value=5)
    sigma = slider("Sigma", 0.5, 5.0, 1.4, 0.1)
    auto = checkbox("Automatic thresholds", False)
    low = slider("Low threshold", 0, 255, 20, disabled=auto)
    high = slider("High threshold", 0, 255, 60, disabled=auto)
    return dict(ksize=ksize, sigma=sigma, auto=auto, low=low, high=high)


def canny_steps_tab(stages):
    cols = st.columns(4)
    for col, key, title in zip(cols, ("smooth", "magnitude", "nms", "edges"),
                               ("Gaussian smoothing", "Sobel magnitude", "Non-maximum suppression", "Hysteresis (final edges)")):
        with col:
            show(stages[key], title)


# =====================================================================
# TASK 1 - PERSON 2 : HOUGH TRANSFORM (lines, circles, ellipses)
# Backend: engine.cpp, section "TASK 1 - PERSON 2" (canny_detect_shapes)
# =====================================================================
lib.canny_detect_shapes.restype = None
lib.canny_detect_shapes.argtypes = [
    U8P, c_int, c_int, c_int,
    c_int, c_float, c_float, c_float, c_int, c_int,
    c_float, c_float, c_int, c_int, c_int,
    c_int, c_int, c_int, c_int, c_int, c_int,
    c_int, c_int, c_int, c_int, c_float, c_int, c_int, c_int, c_int,
    U8P, U8P, U8P, c_int, I32P, I32P, U8P, F32P, I32P,
    F32P, I32P, F32P, I32P, F32P,
]


def hough_controls():
    st.subheader("Hough detectors")
    use_lines = checkbox("Detect lines")
    use_circles = checkbox("Detect circles")
    use_ellipses = checkbox("Detect ellipses")
    thickness = slider("Overlay line/shape thickness", 1, 10, 4)
    with st.expander("Line parameters", expanded=use_lines):
        rho_step = number_input("Rho step", 1.0, 10.0, 1.0)
        theta_step = number_input("Theta step (degrees)", 0.5, 10.0, 1.0)
        line_threshold = number_input("Line vote threshold", 1, 1000, 80)
        line_nms = slider("Line NMS window", 3, 15, 5, 2)
        max_lines = slider("Maximum lines", 1, 64, 16)
    with st.expander("Circle parameters", expanded=use_circles):
        rmin = number_input("Minimum radius", 2, 1000, 10)
        rmax = number_input("Maximum radius", 3, 1500, 150)
        rstep = number_input("Radius step", 1, 20, 2)
        circle_threshold = number_input("Circle vote threshold", 1, 72, 12)
        min_dist = number_input("Minimum center distance", 1, 1000, 20)
        max_circles = slider("Maximum circles", 1, 32, 8)
        gradient_mode = checkbox("Use gradient-guided voting", True)
    with st.expander("Ellipse parameters", expanded=use_ellipses):
        min_major = number_input("Minimum major axis (diameter)", 4, 2000, 30)
        max_major = number_input("Maximum major axis (diameter)", 5, 3000, 300)
        min_minor = number_input("Minimum minor axis (diameter)", 2, 1500, 10)
        st.caption("Detection floor (support): 0.15")
        min_support = slider("Minimum support (fraction of outline)", 0.05, 1.00, 0.35, 0.05)
        max_points = number_input("Maximum sampled edge points", 50, 4000, 300)
        max_working_size = number_input("Max working size (px, 0 = off)", 0, 10000, 600)
        st.caption("Ellipse voting cost grows roughly with N^3.")
        max_ellipses = slider("Maximum ellipses", 1, 64, 16)
    return dict(
        lines=use_lines, circles=use_circles, ellipses=use_ellipses, thickness=thickness,
        rho_step=rho_step, theta_step=theta_step, line_threshold=line_threshold, line_nms=line_nms,
        max_lines=max_lines, rmin=rmin, rmax=rmax, rstep=rstep, circle_threshold=circle_threshold,
        min_dist=min_dist, max_circles=max_circles, gradient_mode=gradient_mode, min_major=min_major,
        max_major=max_major, min_minor=min_minor, min_support=min_support,
        max_points=max_points, max_working_size=max_working_size, max_ellipses=max_ellipses)


def hough_detect(image, c, h):
    """Canny edges plus the Hough detectors selected in `h`. Returns edges, overlay, shapes and accumulators."""
    rows, cols = image.shape[:2]
    flags = int(h["lines"]) | (int(h["circles"]) << 1) | (int(h["ellipses"]) << 2)
    if h["ellipses"]:
        flags |= 8
    edges = np.empty((rows, cols), np.uint8)
    overlay = np.empty((rows, cols, 3), np.uint8)
    line_capacity = 4096 * 360
    line_acc = np.zeros(line_capacity, np.uint8)
    circle_acc = np.zeros((rows, cols), np.uint8)
    lines = np.zeros((h["max_lines"], 7), np.float32)
    circles = np.zeros((h["max_circles"], 4), np.float32)
    ellipse_buffer = 64
    ellipses = np.zeros((ellipse_buffer, 7), np.float32)
    line_w, line_h = c_int(), c_int()
    line_n, circle_n, ellipse_n = c_int(), c_int(), c_int()
    info = np.zeros(10, np.float32)
    lib.canny_detect_shapes(
        u8(image), rows, cols, 1 if image.ndim == 2 else 3,
        int(c["ksize"]), float(c["sigma"]), float(c["low"]), float(c["high"]), int(c["auto"]), flags,
        float(h["rho_step"]), float(h["theta_step"]), int(h["line_threshold"]), int(h["line_nms"]),
        int(h["max_lines"]), int(h["rmin"]), int(h["rmax"]), int(h["rstep"]), int(h["circle_threshold"]),
        int(h["min_dist"]), int(h["max_circles"]), int(h["gradient_mode"]),
        int(h["min_major"]), int(h["max_major"]), int(h["min_minor"]), 0.15,
        int(h["max_points"]), int(h["max_working_size"]), ellipse_buffer, int(h["thickness"]),
        u8(edges), u8(overlay), u8(line_acc), line_capacity, ctypes.byref(line_w), ctypes.byref(line_h),
        u8(circle_acc), f32(lines), ctypes.byref(line_n), f32(circles), ctypes.byref(circle_n),
        f32(ellipses), ctypes.byref(ellipse_n), f32(info))
    acc = line_acc[:line_w.value * line_h.value].reshape((line_h.value, line_w.value)) \
        if line_w.value and line_h.value else np.zeros((1, 1), np.uint8)
    return {"edges": edges, "overlay": overlay, "lines": lines[:line_n.value].copy(),
            "circles": circles[:circle_n.value].copy(), "ellipses": ellipses[:ellipse_n.value].copy(),
            "line_acc": acc.copy(), "circle_acc": circle_acc, "thresholds": (float(info[0]), float(info[1])),
            "line_rho_max": float(np.ceil(np.hypot(cols, rows))),
            "line_rho_step": float(h["rho_step"]), "line_theta_step": float(h["theta_step"]),
            "sample_ratio": float(info[4]), "raw_edge_count": int(info[5]),
            "ellipse_scale": float(info[6]), "ellipse_work_width": int(info[7]),
            "ellipse_work_height": int(info[8]), "ellipse_work_edge_count": int(info[9])}


def draw_ellipse_overlay(overlay, ellipses, thickness):
    """Draw filtered engine ellipses in Python using the engine's coordinate convention."""
    rgb = Image.fromarray(np.ascontiguousarray(overlay[..., ::-1])).convert("RGB")
    draw = ImageDraw.Draw(rgb)
    color = (220, 0, 255)
    width = max(1, int(thickness))
    cross = 3 * width
    for x, y, a, b, angle, _votes, _support in ellipses:
        angle_rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        q = np.linspace(0.0, 2.0 * np.pi, 361)
        points = np.column_stack((
            x + a * np.cos(q) * cos_a - b * np.sin(q) * sin_a,
            y + a * np.cos(q) * sin_a + b * np.sin(q) * cos_a,
        ))
        polyline = [tuple(np.rint(point).astype(int)) for point in points]
        draw.line(polyline, fill=color, width=width, joint="curve")
        center = (int(round(x)), int(round(y)))
        draw.line((center[0] - cross, center[1], center[0] + cross, center[1]),
                  fill=color, width=max(2, width // 2))
        draw.line((center[0], center[1] - cross, center[0], center[1] + cross),
                  fill=color, width=max(2, width // 2))
    return np.ascontiguousarray(np.asarray(rgb)[..., ::-1])


def ellipse_csv(ellipses):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    headers = ["center_x", "center_y", "semi_major", "semi_minor", "angle", "votes", "support"]
    writer.writerow(headers)
    writer.writerows(ellipses.tolist())
    return output.getvalue()


def hough_accumulators_tab(result, h):
    if not (h["lines"] or h["circles"]):
        st.info("Enable line or circle detection to see the accumulators.")
    if h["lines"]:
        resized = Image.fromarray(result["line_acc"]).resize((600, 400), Image.BILINEAR)
        show(colormap(np.asarray(resized)), "Line accumulator (rho x theta)")
    if h["circles"]:
        show(colormap(result["circle_acc"]), "Circle center accumulator")


def hough_data_tab(result, h):
    if not (h["lines"] or h["circles"] or h["ellipses"]):
        st.info("Enable a detector to list what it finds.")
    if h["lines"]:
        st.subheader("Lines: rho, theta, votes, endpoints")
        table([dict(zip(["rho", "theta", "votes", "x1", "y1", "x2", "y2"], r)) for r in result["lines"].tolist()])
    if h["circles"]:
        st.subheader("Circles: center_x, center_y, radius, votes")
        table([dict(zip(["center_x", "center_y", "radius", "votes"], r)) for r in result["circles"].tolist()])
    if h["ellipses"]:
        headers = ["center_x", "center_y", "semi_major", "semi_minor", "angle", "votes", "support"]
        st.subheader("Ellipses: center_x, center_y, semi_major, semi_minor, angle, votes, support")
        table([dict(zip(headers, r)) for r in result["ellipses"].tolist()])
        st.download_button("Download detections CSV", ellipse_csv(result["ellipses"]),
                           "ellipse_detections.csv", "text/csv")


# =====================================================================
# TASK 2 - PERSON 3 : SNAKE (greedy active contour)
# Backend: engine.cpp, section "TASK 2 - PERSON 3" (snake_greedy, snake_gradient_map)
# =====================================================================
lib.snake_greedy.restype = None
lib.snake_greedy.argtypes = [
    U8P, c_int, c_int, c_int,
    F32P, c_int,
    c_float, c_float, c_float, c_float,
    c_int, c_int, c_int,
    c_float, c_float,
    F32P, F32P,
    F32P, c_int, c_int, I32P,
]
lib.snake_gradient_map.restype = None
lib.snake_gradient_map.argtypes = [U8P, c_int, c_int, c_int, c_float, U8P]
SNAP_EVERY = 5


def init_circle(cx, cy, r, n=100):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1).astype(np.float32)


def init_ellipse(cx, cy, rx, ry, n=100):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1).astype(np.float32)


def _resample_closed(p, n):
    p = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    out = []
    for v in np.linspace(0, cum[-1], n, endpoint=False):
        i = min(np.searchsorted(cum, v, side="right") - 1, len(seg) - 1)
        f = (v - cum[i]) / max(seg[i], 1e-9)
        out.append(p[i] + f * (p[i + 1] - p[i]))
    return np.array(out, np.float32)


def init_rect(x0, y0, x1, y1, n=100):
    """Rectangle with rounded corners (sharp corners make a greedy snake leave spikes)."""
    dense = _resample_closed(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), 4 * n)
    for _ in range(40):
        dense = 0.5 * dense + 0.25 * (np.roll(dense, 1, axis=0) + np.roll(dense, -1, axis=0))
    return _resample_closed(dense, n)


CANVAS_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; background: transparent; overflow: hidden; }
  canvas { display: block; touch-action: none; }
</style></head>
<body>
<canvas id="c"></canvas>
<script>
(function () {
  var cv = document.getElementById("c"), ctx = cv.getContext("2d");
  var args = null, img = new Image(), imgSrc = "";
  var box = null, prev = null, mode = null, anchor = null, grab = null, k = 1, W = 0, H = 0;

  function send(type, data) {
    window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, data || {}), "*");
  }
  function norm(b) { return [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])]; }
  function corners(b) { return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]; }
  function clampPoint(p) { return [Math.min(Math.max(p[0], 0), args.orig_w), Math.min(Math.max(p[1], 0), args.orig_h)]; }
  function pos(e) {
    var r = cv.getBoundingClientRect();
    return clampPoint([(e.clientX - r.left) / k, (e.clientY - r.top) / k]);
  }
  function handleAt(p, b) {
    var c = corners(b);
    for (var i = 0; i < 4; i++) {
      if (Math.hypot((p[0] - c[i][0]) * k, (p[1] - c[i][1]) * k) < 12) return i;
    }
    return -1;
  }
  function inside(p, b) { return p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3]; }

  function layout() {
    if (!args) return;
    W = Math.min(Math.max(240, document.documentElement.clientWidth), 1100);
    k = W / args.orig_w;
    H = Math.round(args.orig_h * k);
    cv.width = W; cv.height = H;
    send("streamlit:setFrameHeight", {height: H + 2});
    draw();
  }

  function draw() {
    if (!args) return;
    ctx.clearRect(0, 0, W, H);
    if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, W, H);
    if (!box) return;
    var b = norm(box), x0 = b[0] * k, y0 = b[1] * k, x1 = b[2] * k, y1 = b[3] * k;
    var cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, w = x1 - x0, h = y1 - y0;
    ctx.lineWidth = 2; ctx.strokeStyle = "#ffa000";
    ctx.beginPath();
    if (args.shape === "Circle") ctx.arc(cx, cy, Math.max(w, h) / 2, 0, 2 * Math.PI);
    else if (args.shape === "Ellipse") ctx.ellipse(cx, cy, w / 2, h / 2, 0, 0, 2 * Math.PI);
    else ctx.rect(x0, y0, w, h);
    ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.setLineDash([5, 4]);
    ctx.strokeRect(x0, y0, w, h); ctx.setLineDash([]);
    var c = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    for (var i = 0; i < 4; i++) {
      ctx.fillStyle = "#ffffff"; ctx.strokeStyle = "#333333";
      ctx.fillRect(c[i][0] - 5, c[i][1] - 5, 10, 10); ctx.strokeRect(c[i][0] - 5, c[i][1] - 5, 10, 10);
    }
  }

  cv.addEventListener("pointerdown", function (e) {
    if (!args) return;
    var p = pos(e), b = box ? norm(box) : null, hit = b ? handleAt(p, b) : -1;
    prev = box ? box.slice() : null;
    if (hit >= 0) { mode = "resize"; anchor = corners(b)[(hit + 2) % 4]; }
    else if (b && inside(p, b)) { mode = "move"; grab = [p[0] - b[0], p[1] - b[1], b[2] - b[0], b[3] - b[1]]; }
    else { mode = "draw"; anchor = p; box = [p[0], p[1], p[0], p[1]]; }
    cv.setPointerCapture(e.pointerId); e.preventDefault(); draw();
  });
  cv.addEventListener("pointermove", function (e) {
    if (!args) return;
    var p = pos(e);
    if (mode === "draw" || mode === "resize") { box = [anchor[0], anchor[1], p[0], p[1]]; draw(); }
    else if (mode === "move") {
      var nx = Math.min(Math.max(p[0] - grab[0], 0), args.orig_w - grab[2]);
      var ny = Math.min(Math.max(p[1] - grab[1], 0), args.orig_h - grab[3]);
      box = [nx, ny, nx + grab[2], ny + grab[3]]; draw();
    } else {
      var b = box ? norm(box) : null;
      cv.style.cursor = b && handleAt(p, b) >= 0 ? "nwse-resize" : (b && inside(p, b) ? "move" : "crosshair");
    }
  });
  function sendBox(b) {
    send("streamlit:setComponentValue", {value: {x0: b[0], y0: b[1], x1: b[2], y1: b[3]}, dataType: "json"});
  }
  function same(a, b) {
    if (!a || !b) return false;
    var x = norm(a), y = norm(b);
    return Math.abs(x[0] - y[0]) + Math.abs(x[1] - y[1]) + Math.abs(x[2] - y[2]) + Math.abs(x[3] - y[3]) < 0.5;
  }
  function finish(e) {
    if (!mode) return;
    var b = norm(box), tiny = (b[2] - b[0] < 6 || b[3] - b[1] < 6);
    if (mode === "draw" && tiny && prev) {
      // a plain click on the picture: center the existing box there, keep its size
      var pb = norm(prev), w = pb[2] - pb[0], h = pb[3] - pb[1];
      var nx = Math.min(Math.max(anchor[0] - w / 2, 0), args.orig_w - w);
      var ny = Math.min(Math.max(anchor[1] - h / 2, 0), args.orig_h - h);
      box = [nx, ny, nx + w, ny + h]; sendBox(box);
    } else if (tiny) { box = prev; }
    else { box = b; if (!same(box, prev)) sendBox(box); }
    mode = null; draw();
  }
  cv.addEventListener("pointerup", finish);
  cv.addEventListener("pointercancel", finish);

  window.addEventListener("message", function (ev) {
    if (!ev.data || ev.data.type !== "streamlit:render") return;
    args = ev.data.args;
    if (mode === null && args.box) box = args.box.slice();
    if (args.image_url !== imgSrc) { imgSrc = args.image_url; img.onload = layout; img.src = imgSrc; }
    else layout();
  });
  window.addEventListener("resize", layout);
  send("streamlit:componentReady", {apiVersion: 1});
})();
</script>
</body></html>
"""


@st.cache_resource
def box_canvas_component():
    """Small drag-a-box component (plain HTML + JS, written next to the temp files once)."""
    folder = os.path.join(tempfile.gettempdir(), "cv_task2_box_canvas")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "index.html")
    if not os.path.isfile(path) or open(path, encoding="utf8").read() != CANVAS_HTML:
        with open(path, "w", encoding="utf8") as f:
            f.write(CANVAS_HTML)
    return components.declare_component("box_canvas", path=folder)


@st.cache_data(show_spinner=False, max_entries=4)
def canvas_picture(image):
    """The image as a (smaller) JPEG data URL for the drawing area."""
    h, w = image.shape[:2]
    rgb = np.ascontiguousarray(image[..., ::-1] if image.ndim == 3 else np.stack([image] * 3, axis=-1))
    pil = Image.fromarray(rgb)
    if w > 1000:
        pil = pil.resize((1000, max(1, round(h * 1000 / w))), Image.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def default_box(shape, w, h):
    cx, cy = w / 2, h / 2
    if shape == "Circle":
        r = min(h, w) / 3
        return [cx - r, cy - r, cx + r, cy + r]
    if shape == "Ellipse":
        return [cx - w / 3, cy - h / 3, cx + w / 3, cy + h / 3]
    return [w / 4, h / 4, 3 * w / 4, 3 * h / 4]


def initial_contour(shape, box, n_points, w, h):
    """Initial closed contour that fits a box [x0, y0, x1, y1] (image pixels). Returns (points, readable geometry)."""
    x0, x1 = sorted((min(max(box[0], 0), w - 1), min(max(box[2], 0), w - 1)))
    y0, y1 = sorted((min(max(box[1], 0), h - 1), min(max(box[3], 0), h - 1)))
    x1, y1 = max(x1, x0 + 6), max(y1, y0 + 6)
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if shape == "Circle":
        r = max(bw, bh) / 2
        return init_circle(cx, cy, r, n_points), {"Center x": round(cx), "Center y": round(cy), "Radius": round(r)}
    if shape == "Ellipse":
        return init_ellipse(cx, cy, bw / 2, bh / 2, n_points), {"Center x": round(cx), "Center y": round(cy),
                                                              "Radius x": round(bw / 2), "Radius y": round(bh / 2)}
    return init_rect(x0, y0, x1, y1, n_points), {"Left": round(x0), "Top": round(y0), "Right": round(x1), "Bottom": round(y1)}


def snake_controls(image):
    """Left-panel controls. Returns the parameters; 'slider_box' is set only when the contour is placed with sliders."""
    h, w = image.shape[:2]
    size = f"_{w}x{h}"  # slider ranges depend on the image, so each image size gets its own slider state
    st.subheader("Initial contour")
    placement = radio("Place the contour by", ["Dragging on the image", "Sliders"])
    shape = selectbox("Shape", ["Circle", "Ellipse", "Rectangle"])
    n_points = slider("Number of points", 20, 400, 100, 10)
    slider_box = None
    if placement == "Sliders":
        if shape == "Rectangle":
            left = slider("Left", 0, w - 1, w // 4, key=_key("Left", size))
            right = slider("Right", 0, w - 1, 3 * w // 4, key=_key("Right", size))
            top = slider("Top", 0, h - 1, h // 4, key=_key("Top", size))
            bottom = slider("Bottom", 0, h - 1, 3 * h // 4, key=_key("Bottom", size))
            slider_box = [left, top, right, bottom]
        else:
            cx = slider("Center x", 0, w - 1, w // 2, key=_key("Center x", size))
            cy = slider("Center y", 0, h - 1, h // 2, key=_key("Center y", size))
            if shape == "Circle":
                r = slider("Radius", 5, max(6, min(h, w) // 2), max(6, min(h, w) // 3), key=_key("Radius", size))
                slider_box = [cx - r, cy - r, cx + r, cy + r]
            else:
                rx = slider("Radius x", 5, max(6, w // 2), max(6, w // 3), key=_key("Radius x", size))
                ry = slider("Radius y", 5, max(6, h // 2), max(6, h // 3), key=_key("Radius y", size))
                slider_box = [cx - rx, cy - ry, cx + rx, cy + ry]
    st.subheader("Snake parameters")
    alpha = slider("Alpha (continuity)", 0.0, 5.0, 1.0, 0.1)
    beta = slider("Beta (curvature)", 0.0, 5.0, 1.0, 0.1)
    gamma = slider("Gamma (edge)", 0.0, 5.0, 1.5, 0.1)
    balloon = slider("Balloon (+ inflate / - deflate)", -3.0, 3.0, -0.5, 0.1)
    max_iter = slider("Max iterations", 10, 1000, 300, 10)
    window = select_slider("Search half-window", [1, 2, 3], 2)
    sigma = slider("Gaussian sigma", 0.0, 6.0, 2.0, 0.5)
    corner = slider("Corner threshold (0 = off)", 0.0, 2.0, 0.0, 0.05)
    return dict(placement=placement, shape=shape, n_points=n_points, slider_box=slider_box, alpha=alpha, beta=beta,
                gamma=gamma, balloon=balloon, max_iter=max_iter, window=window, sigma=sigma, corner=corner)


def run_snake(image, p):
    """Evolve the closed contour p['init'] with the greedy algorithm."""
    image = np.ascontiguousarray(image, np.uint8)
    h, w = image.shape[:2]
    init = np.ascontiguousarray(p["init"], np.float32)
    n = len(init)
    max_snaps = 400
    out = np.zeros((n, 2), np.float32)
    info = np.zeros(10, np.float32)
    hist = np.zeros((max_snaps, n, 2), np.float32)
    count = c_int(0)
    lib.snake_greedy(u8(image), h, w, 1 if image.ndim == 2 else 3, f32(init), n,
                     float(p["alpha"]), float(p["beta"]), float(p["gamma"]), float(p["balloon"]),
                     int(p["window"]), int(p["max_iter"]), 0, float(p["sigma"]), float(p["corner"]),
                     f32(out), f32(info), f32(hist), max_snaps, SNAP_EVERY, ctypes.byref(count))
    return {"init": init.copy(), "contour": out, "iterations": int(info[0]), "converged": bool(info[4]),
            "mean_gradient": float(info[3]), "energy": float(info[2]),
            "history": [hist[i].copy() for i in range(count.value)]}


def energy_map(image, sigma):
    image = np.ascontiguousarray(image, np.uint8)
    h, w = image.shape[:2]
    out = np.zeros((h, w), np.uint8)
    lib.snake_gradient_map(u8(image), h, w, 1 if image.ndim == 2 else 3, float(sigma), u8(out))
    return out


def snake_tab(image, snake, final):
    a, b = st.columns(2)
    with a:
        show(energy_map(image, snake["settings"]["Gaussian sigma"]), "Image energy (bright = strong edge)")
        st.download_button("Download final image (PNG)", encode_png(final), "snake_final.png", "image/png")
    frames = snake["history"]
    with b:
        if len(frames) > 1:
            i = st.slider("Evolution (snapshot)", 0, len(frames) - 1, len(frames) - 1,
                          help=f"One snapshot every {SNAP_EVERY} iterations.")
            show(draw_contour(image, frames[i], (0, 255, 0), 2), f"Iteration {i * SNAP_EVERY}")
        else:
            st.info("The contour did not move.")


# =====================================================================
# TASK 2 - PERSON 4 : CHAIN CODE, PERIMETER, AREA
# Backend: engine.cpp, section "TASK 2 - PERSON 4" (contour_to_chain, chain_*, polygon_measure)
# Chain code: Freeman 8 directions, 0 right, 1 up-right, 2 up, 3 up-left, 4 left, 5 down-left, 6 down, 7 down-right.
# =====================================================================
lib.contour_to_chain.restype = c_int
lib.contour_to_chain.argtypes = [F32P, c_int, I32P, U8P, c_int]
lib.chain_differential.restype = None
lib.chain_differential.argtypes = [U8P, c_int, U8P]
lib.chain_decode.restype = None
lib.chain_decode.argtypes = [I32P, U8P, c_int, I32P]
lib.chain_measure.restype = None
lib.chain_measure.argtypes = [U8P, c_int, F64P]
lib.chain_fill.restype = ctypes.c_longlong
lib.chain_fill.argtypes = [I32P, U8P, c_int, c_int, c_int, U8P]
lib.polygon_measure.restype = None
lib.polygon_measure.argtypes = [F32P, c_int, F64P]
DIRECTIONS = ["0 right", "1 up-right", "2 up", "3 up-left", "4 left", "5 down-left", "6 down", "7 down-right"]


def analyze_contour(contour, shape, pixel_size=1.0):
    """Chain code of a closed (N, 2) contour, with perimeter and area. Lengths scale by pixel_size, areas by its square."""
    h, w = shape[:2]
    contour = np.ascontiguousarray(contour, np.float32)
    n = len(contour)
    start = np.zeros(2, np.int32)
    capacity = max(1024, 4 * n)
    while True:  # grow the buffer until the whole chain fits
        codes = np.zeros(capacity, np.uint8)
        length = lib.contour_to_chain(f32(contour), n, start.ctypes.data_as(I32P), u8(codes), capacity)
        if length <= capacity:
            break
        capacity = length
    if length == 0:
        raise ValueError("The contour is degenerate (fewer than 3 distinct points).")
    codes = codes[:length].copy()
    diff = np.zeros_like(codes)
    lib.chain_differential(u8(codes), length, u8(diff))
    boundary = np.zeros((length, 2), np.int32)
    lib.chain_decode(start.ctypes.data_as(I32P), u8(codes), length, boundary.ctypes.data_as(I32P))
    m = np.zeros(5, np.float64)
    lib.chain_measure(u8(codes), length, m.ctypes.data_as(F64P))
    mask = np.zeros((h, w), np.uint8)
    pixels = lib.chain_fill(start.ctypes.data_as(I32P), u8(codes), length, h, w, u8(mask))
    poly = np.zeros(2, np.float64)
    lib.polygon_measure(f32(contour), n, poly.ctypes.data_as(F64P))
    s = float(pixel_size)
    return {"start": (int(start[0]), int(start[1])), "chain": codes, "differential": diff, "boundary": boundary,
            "mask": mask, "length": length, "closed": bool(m[4]), "n_even": int(m[2]), "n_odd": int(m[3]),
            "perimeter": m[0] * s, "area": m[1] * s * s, "area_pixels": pixels * s * s,
            "polygon_perimeter": poly[0] * s, "polygon_area": poly[1] * s * s,
            "histogram": np.bincount(codes, minlength=8), "pixel_size": s,
            "string": "".join(map(str, codes.tolist())), "diff_string": "".join(map(str, diff.tolist()))}


def verify(res):
    """Recompute the engine's numbers a second way in NumPy. Returns [(name, engine, independent, difference)]."""
    s = res["pixel_size"]
    p = res["boundary"].astype(np.float64)
    q = np.roll(p, -1, axis=0)
    perimeter = float(np.hypot(*(q - p).T).sum())
    area = 0.5 * abs(float(np.dot(p[:, 0], q[:, 1]) - np.dot(q[:, 0], p[:, 1])))
    rows = [("Perimeter: chain-code formula vs summed step lengths", res["perimeter"], perimeter * s),
            ("Area: C++ shoelace vs NumPy shoelace", res["area"], area * s * s)]
    if len(np.unique(res["boundary"], axis=0)) == len(res["boundary"]):  # Pick's theorem needs a simple boundary
        rows.append(("Pixel count: flood fill vs Pick's theorem", res["area_pixels"], (area + len(p) / 2 + 1) * s * s))
    return [(n, a, b, abs(a - b)) for n, a, b in rows]


def checks_ok(checks):
    return all(d <= max(1e-6, 1e-3 * abs(ref)) for _, _, ref, d in checks)


def draw_analysis(image, res, contour):
    """Image with the interior tinted, the chain-code boundary in green and the start pixel in red."""
    vis = to_bgr(image).astype(np.float32)
    m = res["mask"] > 0
    vis[m] = 0.7 * vis[m] + 0.3 * np.array([0, 200, 255], np.float32)
    vis = np.clip(vis + 0.5, 0, 255).astype(np.uint8)
    b = res["boundary"]
    stamp(vis, b[:, 0], b[:, 1], (0, 255, 0), 0)
    stamp(vis, [res["start"][0]], [res["start"][1]], (0, 0, 255), 3)
    return vis


def chain_report(res):
    return (f"start (x, y): {res['start'][0]}, {res['start'][1]}\nlength: {res['length']}\n"
            f"perimeter: {res['perimeter']:.4f}\narea (shoelace): {res['area']:.4f}\n"
            f"area (pixels): {res['area_pixels']:.0f}\nchain code:\n{res['string']}\n"
            f"differential code:\n{res['diff_string']}\n")


def measurement_controls():
    st.subheader("Measurement")
    return number_input("Pixel size (1 = report in pixels)", 0.0001, 1000.0, 1.0, format="%.4f")


def person4_metrics(res, checks):
    unit = "px" if res["pixel_size"] == 1.0 else "units"
    cols = st.columns(5)
    cols[0].metric(f"Perimeter ({unit})", f"{res['perimeter']:.2f}")
    cols[1].metric(f"Area, shoelace ({unit}²)", f"{res['area']:.1f}")
    cols[2].metric(f"Area, pixel count ({unit}²)", f"{res['area_pixels']:.0f}")
    cols[3].metric("Chain length", res["length"])
    cols[4].metric("Checks", "All passed" if checks_ok(checks) else "Mismatch")


def chain_tab(res):
    st.write(f"Start pixel (x, y): **{res['start'][0]}, {res['start'][1]}**. "
             f"{res['n_even']} straight steps and {res['n_odd']} diagonal steps. "
             "Directions: 0 right, 1 up-right, 2 up, 3 up-left, 4 left, 5 down-left, 6 down, 7 down-right.")
    st.text_area("Chain code", res["string"], height=130)
    st.text_area("First difference (rotation invariant)", res["diff_string"], height=130)
    st.bar_chart({"steps": {name: int(v) for name, v in zip(DIRECTIONS, res["histogram"])}})
    st.download_button("Download chain code (TXT)", chain_report(res), "chain_code.txt", "text/plain")


def measurements_tab(image, res, checks, contour):
    a, b = st.columns(2)
    analysis = draw_analysis(image, res, contour)
    with a:
        show(draw_contour(image, contour, (255, 160, 0), 2), "Final snake contour")
    with b:
        show(analysis, "Chain-code boundary (green), interior, start pixel (red)")
    st.subheader("Verification")
    st.caption("Each number from the C++ engine is recomputed independently with NumPy.")
    table([{"Check": n, "C++ engine": round(x, 4), "Independent": round(y, 4), "Difference": round(d, 6),
            "Result": "match" if d <= max(1e-6, 1e-3 * abs(y)) else "MISMATCH"} for n, x, y, d in checks])
    st.subheader("Snake polygon vs chain code")
    table([{"Measure": "Perimeter", "Snake polygon": round(res["polygon_perimeter"], 3), "Chain code": round(res["perimeter"], 3)},
           {"Measure": "Area", "Snake polygon": round(res["polygon_area"], 3), "Chain code": round(res["area"], 3)}])
    st.caption("Perimeter = even codes + sqrt(2) x odd codes. It is slightly longer than the smooth polygon "
               "(up to about 6.6% on round shapes). Area = shoelace formula on the chain-code boundary.")
    summary = {k: res[k] for k in ("length", "perimeter", "area", "area_pixels", "polygon_perimeter", "polygon_area",
                                   "n_even", "n_odd", "closed", "pixel_size")}
    summary.update(start=list(res["start"]), histogram=res["histogram"].tolist(),
                   verification=[{"check": n, "engine": x, "independent": y, "difference": d} for n, x, y, d in checks])
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("analysis.png", encode_png(analysis))
        z.writestr("mask.png", encode_png(res["mask"]))
        z.writestr("chain_code.txt", chain_report(res))
        z.writestr("results.json", json.dumps(summary, indent=2))
    st.download_button("Download results (ZIP)", archive.getvalue(), "contour_results.zip", "application/zip")


# =====================================================================
# COMPARE: pin a result, change a setting, look at both side by side
# =====================================================================
def compare_buttons(name):
    st.subheader("Compare")
    a, b = st.columns(2)
    pin = a.button("Pin result", key=f"pin_{name}", width="stretch",
                   help="Keep the current result as A. Change a setting, then open the Compare tab.")
    if b.button("Clear pin", key=f"clear_{name}", width="stretch"):
        st.session_state.pop(f"pinned_{name}", None)
    return pin


def get_pin(name, image_id):
    pin = st.session_state.get(f"pinned_{name}")
    return pin if pin is not None and pin["image_id"] == image_id else None


def settings_diff(a, b):
    """Rows for every setting that differs between two settings dicts."""
    rows = []
    for key in list(a) + [k for k in b if k not in a]:
        va, vb = a.get(key, "off"), b.get(key, "off")
        if va != vb:
            rows.append({"Setting": key, "A (pinned)": str(va), "B (current)": str(vb)})
    return rows


def show_settings_diff(a, b):
    st.subheader("Settings that differ")
    rows = settings_diff(a, b)
    if rows:
        table(rows)
    else:
        st.info("A and B use the same settings.")


def no_pin_message():
    st.info("Click **Pin result** in the left panel, change a setting, then come back to this tab "
            "to see the pinned result (A) next to the current one (B).")


def edge_difference(a, b):
    """White = edge in both, red = only in A, green = only in B."""
    out = np.zeros(a.shape + (3,), np.uint8)
    ea, eb = a > 0, b > 0
    out[ea & eb] = (255, 255, 255)
    out[ea & ~eb] = (0, 0, 255)
    out[~ea & eb] = (0, 255, 0)
    return out


def compare_canny_tab(image_id, now):
    pin = get_pin("canny", image_id)
    if pin is None:
        return no_pin_message()
    a, b = st.columns(2)
    with a:
        show(pin["overlay"], "A (pinned)")
    with b:
        show(now["overlay"], "B (current)")
    show(edge_difference(pin["edges"], now["edges"]), "Edge difference: white = both, red = only A, green = only B")
    ea, eb = np.ascontiguousarray(pin["edges"]), np.ascontiguousarray(now["edges"])
    overlap = lib.edge_similarity(u8(ea), u8(eb), ea.size)
    st.subheader("Numbers")
    table([{"Measure": k, "A (pinned)": str(pin["numbers"][k]), "B (current)": str(now["numbers"][k])}
           for k in now["numbers"]]
          + [{"Measure": "Edge overlap (IoU, 1 = identical)", "A (pinned)": "", "B (current)": f"{float(overlap):.3f}"}])
    show_settings_diff(pin["settings"], now["settings"])


def compare_contour_tab(image_id, now):
    pin = get_pin("contour", image_id)
    if pin is None:
        return no_pin_message()
    a, b = st.columns(2)
    with a:
        show(pin["final"], "A (pinned)")
    with b:
        show(now["final"], "B (current)")
    st.subheader("Numbers")
    rows = []
    for k, vb in now["numbers"].items():
        va = pin["numbers"][k]
        both_numbers = isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
            and not isinstance(va, bool) and not isinstance(vb, bool)
        rows.append({"Measure": k, "A (pinned)": str(va), "B (current)": str(vb),
                     "B - A": f"{vb - va:+.3f}".rstrip("0").rstrip(".") if both_numbers else ""})
    table(rows)
    show_settings_diff(pin["settings"], now["settings"])


# =====================================================================
# PART 1 PAGE: Canny & Hough (Persons 1 and 2)
# =====================================================================
@st.cache_data(show_spinner=False, max_entries=16)
def run_canny_hough(image, canny_p, hough_p):
    result = hough_detect(image, canny_p, hough_p)
    result["stages"] = canny_stages(image, canny_p)
    return result


def engine_hough_settings(h):
    return {key: value for key, value in h.items() if key not in {"min_support", "max_ellipses"}}


def describe_canny(c, h):
    d = {"Gaussian kernel": c["ksize"], "Sigma": c["sigma"], "Automatic thresholds": c["auto"]}
    if not c["auto"]:
        d.update({"Low threshold": c["low"], "High threshold": c["high"]})
    d["Overlay thickness"] = h["thickness"]
    if h["lines"]:
        d.update({"Rho step": h["rho_step"], "Theta step": h["theta_step"], "Line vote threshold": h["line_threshold"],
                  "Line NMS window": h["line_nms"], "Maximum lines": h["max_lines"]})
    if h["circles"]:
        d.update({"Minimum radius": h["rmin"], "Maximum radius": h["rmax"], "Radius step": h["rstep"],
                  "Circle vote threshold": h["circle_threshold"], "Minimum center distance": h["min_dist"],
                  "Maximum circles": h["max_circles"], "Gradient-guided voting": h["gradient_mode"]})
    if h["ellipses"]:
        d.update({"Minimum major axis": h["min_major"], "Maximum major axis": h["max_major"],
                  "Minimum minor axis (diameter)": h["min_minor"],
                  "Detection floor (support)": 0.15,
                  "Minimum support (fraction of outline)": h["min_support"],
                  "Maximum sampled edge points": h["max_points"],
                  "Max working size (px)": h["max_working_size"],
                  "Maximum ellipses": h["max_ellipses"]})
    for name in ("lines", "circles", "ellipses"):
        d[f"Detect {name}"] = h[name]
    return d


def canny_part(image, image_id):
    with st.sidebar:
        pin = compare_buttons("canny")
        canny_p = canny_controls()        # PERSON 1
        hough_p = hough_controls()        # PERSON 2
    result = run_canny_hough(image, canny_p, engine_hough_settings(hough_p))
    if hough_p["ellipses"]:
        support = result["ellipses"][:, 6]
        keep = support >= hough_p["min_support"]
        result["ellipses"] = result["ellipses"][keep][:hough_p["max_ellipses"]].copy()
        result["overlay"] = draw_ellipse_overlay(result["overlay"], result["ellipses"], hough_p["thickness"])
        if result["sample_ratio"] < 0.10:
            st.warning("Sparse sampling: raise Maximum sampled edge points or thin the edges")
    else:
        result["ellipses"] = np.empty((0, 7), np.float32)

    now = {"settings": describe_canny(canny_p, hough_p), "edges": result["edges"], "overlay": result["overlay"],
           "numbers": {"Edge pixels": int(np.count_nonzero(result["edges"])),
                       "Lines": len(result["lines"]) if hough_p["lines"] else "off",
                       "Circles": len(result["circles"]) if hough_p["circles"] else "off",
                       "Ellipses": len(result["ellipses"]) if hough_p["ellipses"] else "off"}}
    if pin:
        st.session_state["pinned_canny"] = dict(now, image_id=image_id)

    m = st.columns(8)
    m[0].metric("Edge pixels", f"{now['numbers']['Edge pixels']:,}")
    m[1].metric("Thresholds used", f"{result['thresholds'][0]:.0f} / {result['thresholds'][1]:.0f}")
    m[2].metric("Lines", now["numbers"]["Lines"])
    m[3].metric("Circles", now["numbers"]["Circles"])
    m[4].metric("Ellipses", now["numbers"]["Ellipses"])
    m[5].metric("Sample ratio", f"{result['sample_ratio']:.1%}" if hough_p["ellipses"] else "off")
    m[6].metric(
        "Ellipse working size",
        f"{result['ellipse_work_width']} x {result['ellipse_work_height']} "
        f"({result['ellipse_scale']:.3f}x)" if hough_p["ellipses"] else "off",
    )
    m[7].metric(
        "Working edge pixels",
        f"{result['ellipse_work_edge_count']:,}" if hough_p["ellipses"] else "off",
    )
    st.caption(f"Image size: {image.shape[1]} x {image.shape[0]} px")
    tabs = st.tabs(["Superimposition", "Canny steps", "Hough accumulators", "Detected data", "Compare"])
    with tabs[0]:
        a, b, c = st.columns(3)
        with a:
            show(image, "Original")
        with b:
            show(result["edges"], "Canny edges")
        with c:
            show(result["overlay"], "Detected shapes")
        st.download_button("Download superimposed image", encode_png(result["overlay"]), "shapes.png", "image/png")
    with tabs[1]:
        canny_steps_tab(result["stages"])                  # PERSON 1
    with tabs[2]:
        hough_accumulators_tab(result, hough_p)            # PERSON 2
    with tabs[3]:
        hough_data_tab(result, hough_p)                    # PERSON 2
    with tabs[4]:
        compare_canny_tab(image_id, now)


# =====================================================================
# PART 2 PAGE: Snake & Contour (Persons 3 and 4)
# =====================================================================
def contour_numbers(res, snake):
    return {"Perimeter": round(res["perimeter"], 2), "Area (shoelace)": round(res["area"], 1),
            "Area (pixel count)": round(res["area_pixels"]), "Chain length": res["length"],
            "Snake iterations": snake["iterations"], "Converged": "yes" if snake["converged"] else "no",
            "Mean edge strength": round(snake["mean_gradient"])}


def contour_part(image, image_id):
    h, w = image.shape[:2]
    snake = st.session_state.get("snake")
    if snake is not None and snake["image_id"] != image_id:
        snake = None
    with st.sidebar:
        run = st.button("Run snake", type="primary", width="stretch")
        pin = compare_buttons("contour")
        p = snake_controls(image)                 # PERSON 3
        pixel_size = measurement_controls()       # PERSON 4

    # Initial contour: a box drawn with the mouse on the image, or the box given by the sliders
    drawing = p["placement"] != "Sliders"
    box_state, canvas_key = f"box_{image_id}", f"canvas_{image_id}"
    drawn = st.session_state.get(canvas_key)
    if drawing and isinstance(drawn, dict):
        st.session_state[box_state] = [drawn["x0"], drawn["y0"], drawn["x1"], drawn["y1"]]
    box = (st.session_state.get(box_state) or default_box(p["shape"], w, h)) if drawing else p["slider_box"]
    init, geometry = initial_contour(p["shape"], box, p["n_points"], w, h)

    names = {"shape": "Shape", "n_points": "Number of points", "alpha": "Alpha (continuity)",
             "beta": "Beta (curvature)", "gamma": "Gamma (edge)", "balloon": "Balloon", "max_iter": "Max iterations",
             "window": "Search half-window", "sigma": "Gaussian sigma", "corner": "Corner threshold"}
    settings = dict({names[k]: p[k] for k in names}, **geometry)
    if run:
        with st.spinner("Evolving the contour..."):
            snake = dict(run_snake(image, dict(p, init=init)), image_id=image_id, settings=settings)
        st.session_state["snake"] = snake

    left, right = st.columns(2)
    with left:
        if drawing:
            box_canvas_component()(image_url=canvas_picture(image), orig_w=w, orig_h=h, box=[float(v) for v in box],
                                   shape=p["shape"], key=canvas_key, default=None)
            st.caption("Initial contour. Click on the object to center the contour there, drag inside it to move it, "
                       "drag a corner to resize, or drag on empty space to draw a new one. Then press Run snake.")
        else:
            show(draw_contour(image, init, (255, 160, 0), 2, points=True), "Initial contour")
    with right:
        if snake is None:
            st.info("Place the initial contour around the object, then press **Run snake** in the left panel.")
            return
        final = draw_contour(image, snake["contour"], (0, 255, 0), 2, points=True)
        show(final, "Final contour")
        st.caption(f"Snake: {snake['iterations']} iterations, "
                   f"{'converged' if snake['converged'] else 'stopped at the iteration limit'}, "
                   f"mean edge strength {snake['mean_gradient']:.0f}/255.")
        if snake["settings"] != settings:
            st.warning("The settings changed since the last run. Press **Run snake** to update.")

    try:
        res = analyze_contour(snake["contour"], image.shape, pixel_size)           # PERSON 4
    except ValueError as err:
        st.error(str(err))
        return
    checks = verify(res)
    now = {"settings": dict(snake["settings"], **{"Pixel size": pixel_size}),
           "final": draw_analysis(image, res, snake["contour"]), "numbers": contour_numbers(res, snake)}
    if pin:
        st.session_state["pinned_contour"] = dict(now, image_id=image_id)

    person4_metrics(res, checks)
    if not snake["converged"]:
        st.warning("The snake stopped at the iteration limit. Increase the iterations or start closer to the object.")
    tabs = st.tabs(["Chain code", "Measurements", "Snake details", "Compare"])
    with tabs[0]:
        chain_tab(res)                                                             # PERSON 4
    with tabs[1]:
        measurements_tab(image, res, checks, snake["contour"])                     # PERSON 4
    with tabs[2]:
        snake_tab(image, snake, final)                                             # PERSON 3
    with tabs[3]:
        compare_contour_tab(image_id, now)


# =====================================================================
# MAIN: one shared image, two parts. Settings are on the left, results on the right.
# =====================================================================
st.title("Computer Vision Workbench")
keep_settings()
uploaded = st.sidebar.file_uploader("Upload grayscale or color image", type=["png", "jpg", "jpeg", "bmp"])
if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()
raw = uploaded.getvalue()
try:
    image = load_image(raw)
except Exception as err:
    st.error(f"Could not read this image: {err}")
    st.stop()

image_id = hashlib.md5(raw).hexdigest()
if st.session_state.get("image_id") != image_id:  # a new image: forget the old snake and pinned results
    st.session_state["image_id"] = image_id
    for key in ("snake", "pinned_canny", "pinned_contour"):
        st.session_state.pop(key, None)

part = st.sidebar.radio("Part", ["Canny & Hough", "Snake & Contour"], key="k_part")
st.sidebar.divider()
if part == "Canny & Hough":
    canny_part(image, image_id)
else:
    contour_part(image, image_id)