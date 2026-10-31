"""Person 3 - Active contour (greedy snake): Python bridge + command line runner.

Use as a module:
    from snake_wrapper import init_circle, run_snake, draw_contour
Or from the terminal:
    python snake_wrapper.py image.png --circle 200 150 90
    python snake_wrapper.py image.png --rect 50 40 350 260 --alpha 1 --beta 1 --gamma 1.5
"""
import argparse
import ctypes
import os
import sys

import cv2 as cv
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.name == "nt":
    _mingw = os.environ.get("MINGW64_BIN", r"C:\msys64\mingw64\bin")
    if os.path.isdir(_mingw):
        os.add_dll_directory(_mingw)
    LIB_NAME = "snake_engine.dll"
elif sys.platform == "darwin":
    LIB_NAME = "snake_engine.dylib"
else:
    LIB_NAME = "snake_engine.so"

LIB_PATH = os.path.join(BASE_DIR, LIB_NAME)
if not os.path.isfile(LIB_PATH):
    raise FileNotFoundError(f"{LIB_NAME} not found at {LIB_PATH}. Build it first (see SNAKE_README.md).")
_lib = ctypes.CDLL(LIB_PATH)

U8P = ctypes.POINTER(ctypes.c_uint8)
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int)

_lib.snake_greedy.restype = None
_lib.snake_greedy.argtypes = [
    U8P, ctypes.c_int, ctypes.c_int, ctypes.c_int,          # image, rows, cols, channels
    F32P, ctypes.c_int,                                      # init_xy, n
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,  # alpha beta gamma balloon
    ctypes.c_int, ctypes.c_int, ctypes.c_int,                # window, max_iter, move_tol
    ctypes.c_float, ctypes.c_float,                          # sigma, corner_thresh
    F32P, F32P,                                              # out_xy, out_info
    F32P, ctypes.c_int, ctypes.c_int, I32P,                  # hist_xy, hist_capacity, snap_every, hist_count
]
_lib.snake_gradient_map.restype = None
_lib.snake_gradient_map.argtypes = [U8P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, U8P]


# ---------------------------------------------------------------- initial contours
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


def _round_corners(p, iters=25):
    """Laplacian smoothing of a closed polygon. Sharp corners make a greedy snake leave spikes, so a
    rounded start is much more reliable. Spacing is restored by resampling afterwards."""
    p = p.astype(np.float32)
    for _ in range(iters):
        p = 0.5 * p + 0.25 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0))
    return p


def init_rect(x0, y0, x1, y1, n=100, round_corners=True):
    """n points spread evenly along the perimeter of a rectangle (corners rounded by default)."""
    dense = _resample_closed(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), 4 * n)
    if round_corners:
        dense = _round_corners(dense, 40)
    return _resample_closed(dense, n)


def init_polygon(points, n=100, round_corners=True):
    """Resample a user polygon (list of (x, y)) to n evenly spaced points (corners rounded by default)."""
    dense = _resample_closed(np.asarray(points, np.float32), 4 * n)
    if round_corners:
        dense = _round_corners(dense, 40)
    return _resample_closed(dense, n)


# ---------------------------------------------------------------- engine call
def _u8(a):
    return a.ctypes.data_as(U8P)


def run_snake(image, init, alpha=1.0, beta=1.0, gamma=1.5, balloon=0.0, window=2,
              max_iter=300, move_tol=0, sigma=2.0, corner_thresh=0.0, snap_every=5, max_snaps=400):
    """Evolve a closed snake with the greedy algorithm.

    image : uint8 gray (H,W) or BGR (H,W,3).  init : (N,2) float array of (x, y).
    Returns dict: contour (N,2), info, history (list of (N,2)).
    """
    image = np.ascontiguousarray(image, dtype=np.uint8)
    h, w = image.shape[:2]
    ch = 1 if image.ndim == 2 else 3
    init = np.ascontiguousarray(init, dtype=np.float32)
    n = len(init)
    out = np.zeros((n, 2), np.float32)
    info = np.zeros(6, np.float32)
    hist = np.zeros((max_snaps, n, 2), np.float32)
    count = ctypes.c_int(0)
    _lib.snake_greedy(
        _u8(image), h, w, ch, init.ctypes.data_as(F32P), n,
        float(alpha), float(beta), float(gamma), float(balloon),
        int(window), int(max_iter), int(move_tol), float(sigma), float(corner_thresh),
        out.ctypes.data_as(F32P), info.ctypes.data_as(F32P),
        hist.ctypes.data_as(F32P), int(max_snaps), int(snap_every), ctypes.byref(count),
    )
    return {
        "contour": out,
        "iterations": int(info[0]),
        "moved_last": int(info[1]),
        "energy": float(info[2]),
        "mean_gradient": float(info[3]),
        "converged": bool(info[4]),
        "history": [hist[i].copy() for i in range(count.value)],
    }


def gradient_map(image, sigma=2.0):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    h, w = image.shape[:2]
    out = np.zeros((h, w), np.uint8)
    _lib.snake_gradient_map(_u8(image), h, w, 1 if image.ndim == 2 else 3, float(sigma), _u8(out))
    return out


# ---------------------------------------------------------------- drawing
def draw_contour(image, contour, color=(0, 255, 0), thickness=2, points=True, point_color=(0, 0, 255)):
    """Return a BGR copy of the image with the closed contour drawn on it."""
    vis = cv.cvtColor(image, cv.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    pts = np.round(contour).astype(np.int32).reshape(-1, 1, 2)
    cv.polylines(vis, [pts], True, color, thickness, cv.LINE_AA)
    if points:
        for x, y in pts.reshape(-1, 2):
            cv.circle(vis, (int(x), int(y)), max(1, thickness), point_color, -1, cv.LINE_AA)
    return vis


def load_image(path):
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    return img if img.dtype == np.uint8 else cv.convertScaleAbs(img)


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Greedy active contour (snake)")
    ap.add_argument("image")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--circle", nargs=3, type=float, metavar=("CX", "CY", "R"))
    g.add_argument("--ellipse", nargs=4, type=float, metavar=("CX", "CY", "RX", "RY"))
    g.add_argument("--rect", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--points", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.0, help="continuity weight")
    ap.add_argument("--beta", type=float, default=1.0, help="curvature weight")
    ap.add_argument("--gamma", type=float, default=1.5, help="image (edge) weight")
    ap.add_argument("--balloon", type=float, default=0.0, help=">0 inflate, <0 deflate")
    ap.add_argument("--window", type=int, default=2, help="search half-window (2 -> 5x5)")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tol", type=int, default=0, help="stop when <= tol points moved")
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma before gradient")
    ap.add_argument("--corner", type=float, default=0.0, help="corner curvature threshold (0 = off)")
    ap.add_argument("--out", default="snake_out", help="output folder")
    a = ap.parse_args()

    img = load_image(a.image)
    if a.circle:
        init = init_circle(*a.circle, n=a.points)
    elif a.ellipse:
        init = init_ellipse(*a.ellipse, n=a.points)
    else:
        init = init_rect(*a.rect, n=a.points)

    res = run_snake(img, init, a.alpha, a.beta, a.gamma, a.balloon, a.window, a.iters, a.tol,
                    a.sigma, a.corner)
    os.makedirs(a.out, exist_ok=True)
    cv.imwrite(os.path.join(a.out, "initial.png"), draw_contour(img, init, (255, 160, 0), 2, False))
    cv.imwrite(os.path.join(a.out, "final.png"), draw_contour(img, res["contour"]))
    np.savetxt(os.path.join(a.out, "contour.csv"), res["contour"], delimiter=",", header="x,y", comments="")
    np.save(os.path.join(a.out, "contour.npy"), res["contour"])
    print(f"iterations={res['iterations']}  converged={res['converged']}  moved_last={res['moved_last']}")
    print(f"final energy={res['energy']:.3f}  mean edge strength on contour={res['mean_gradient']:.1f}/255")
    print(f"saved -> {a.out}/final.png, initial.png, contour.csv, contour.npy")


if __name__ == "__main__":
    main()
