"""Person 4 - Contour analysis (chain code, perimeter, area): Python bridge + command line runner.

Use as a module (the contour is any closed (N, 2) array of (x, y), e.g. the snake output):
    from contour_wrapper import analyze_contour, draw_analysis
    res = analyze_contour(snake_result["contour"], image_shape=img.shape)
Or from the terminal:
    python contour_wrapper.py --selftest
    python contour_wrapper.py image.png --contour snake_out/contour.npy
"""
import argparse
import ctypes
import json
import os
import sys

import cv2 as cv
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.name == "nt":
    _mingw = os.environ.get("MINGW64_BIN", r"C:\msys64\mingw64\bin")
    if os.path.isdir(_mingw):
        os.add_dll_directory(_mingw)
    LIB_NAME = "contour_engine.dll"
elif sys.platform == "darwin":
    LIB_NAME = "contour_engine.dylib"
else:
    LIB_NAME = "contour_engine.so"

LIB_PATH = os.path.join(BASE_DIR, LIB_NAME)
if not os.path.isfile(LIB_PATH):
    raise FileNotFoundError(f"{LIB_NAME} not found at {LIB_PATH}. Build the contour_engine CMake target first.")
_lib = ctypes.CDLL(LIB_PATH)

U8P = ctypes.POINTER(ctypes.c_uint8)
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int)
F64P = ctypes.POINTER(ctypes.c_double)

_lib.contour_to_chain.restype = ctypes.c_int
_lib.contour_to_chain.argtypes = [F32P, ctypes.c_int, I32P, U8P, ctypes.c_int]
_lib.chain_differential.restype = None
_lib.chain_differential.argtypes = [U8P, ctypes.c_int, U8P]
_lib.chain_decode.restype = None
_lib.chain_decode.argtypes = [I32P, U8P, ctypes.c_int, I32P]
_lib.chain_measure.restype = None
_lib.chain_measure.argtypes = [U8P, ctypes.c_int, F64P]
_lib.chain_fill.restype = ctypes.c_longlong
_lib.chain_fill.argtypes = [I32P, U8P, ctypes.c_int, ctypes.c_int, ctypes.c_int, U8P]
_lib.polygon_measure.restype = None
_lib.polygon_measure.argtypes = [F32P, ctypes.c_int, F64P]

DIRECTION_NAMES = ["0 right", "1 up-right", "2 up", "3 up-left", "4 left", "5 down-left", "6 down", "7 down-right"]


# ---------------------------------------------------------------- engine calls
def contour_to_chain(contour):
    """Closed float contour -> (start (x, y), chain code uint8 array). Empty chain if degenerate."""
    contour = np.ascontiguousarray(contour, dtype=np.float32)
    n = len(contour)
    start = np.zeros(2, np.int32)
    capacity = max(1024, 4 * n)
    while True:
        codes = np.zeros(capacity, np.uint8)
        length = _lib.contour_to_chain(contour.ctypes.data_as(F32P), n, start.ctypes.data_as(I32P),
                                       codes.ctypes.data_as(U8P), capacity)
        if length <= capacity:
            return start, codes[:length].copy()
        capacity = length


def chain_differential(codes):
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    out = np.zeros_like(codes)
    if len(codes):
        _lib.chain_differential(codes.ctypes.data_as(U8P), len(codes), out.ctypes.data_as(U8P))
    return out


def chain_decode(start, codes):
    """Boundary pixels (N, 2) of (x, y) rebuilt from the chain code."""
    start = np.ascontiguousarray(start, dtype=np.int32)
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    out = np.zeros((len(codes), 2), np.int32)
    if len(codes):
        _lib.chain_decode(start.ctypes.data_as(I32P), codes.ctypes.data_as(U8P), len(codes), out.ctypes.data_as(I32P))
    return out


def chain_measure(codes):
    """Perimeter and area computed from the chain code only."""
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    out = np.zeros(5, np.float64)
    if len(codes):
        _lib.chain_measure(codes.ctypes.data_as(U8P), len(codes), out.ctypes.data_as(F64P))
    return {"perimeter": float(out[0]), "area": float(out[1]), "n_even": int(out[2]),
            "n_odd": int(out[3]), "closed": bool(out[4])}


def chain_fill(start, codes, rows, cols):
    """(pixel count inside the boundary incl. boundary, uint8 mask rows x cols)."""
    start = np.ascontiguousarray(start, dtype=np.int32)
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    mask = np.zeros((rows, cols), np.uint8)
    if not len(codes):
        return 0, mask
    count = _lib.chain_fill(start.ctypes.data_as(I32P), codes.ctypes.data_as(U8P), len(codes), rows, cols,
                            mask.ctypes.data_as(U8P))
    return int(count), mask


def polygon_measure(contour):
    contour = np.ascontiguousarray(contour, dtype=np.float32)
    out = np.zeros(2, np.float64)
    _lib.polygon_measure(contour.ctypes.data_as(F32P), len(contour), out.ctypes.data_as(F64P))
    return {"perimeter": float(out[0]), "area": float(out[1])}


# ---------------------------------------------------------------- analysis
def analyze_contour(contour, image_shape, pixel_size=1.0):
    """Full Person 4 analysis of a closed contour.

    contour : (N, 2) array of (x, y).  image_shape : (H, W[, C]) of the image the contour lives on.
    pixel_size : physical size of one pixel (e.g. mm); lengths scale by it, areas by its square.
    Returns a dict with chain code, differential code, boundary pixels, mask and all measurements.
    """
    h, w = image_shape[:2]
    start, codes = contour_to_chain(contour)
    if not len(codes):
        raise ValueError("Contour is degenerate (fewer than 3 distinct points); nothing to encode.")
    m = chain_measure(codes)
    pixels, mask = chain_fill(start, codes, h, w)
    poly = polygon_measure(contour)
    boundary = chain_decode(start, codes)
    s = float(pixel_size)
    return {
        "start": (int(start[0]), int(start[1])),
        "chain": codes,
        "chain_string": "".join(map(str, codes.tolist())),
        "differential": chain_differential(codes),
        "boundary": boundary,
        "mask": mask,
        "length": len(codes),
        "perimeter": m["perimeter"] * s,            # chain code: n_even + sqrt(2) * n_odd
        "area": m["area"] * s * s,                  # shoelace on the chain-code boundary
        "area_pixels": pixels * s * s,              # filled pixel count (boundary included)
        "polygon_perimeter": poly["perimeter"] * s,  # raw float snake polygon
        "polygon_area": poly["area"] * s * s,
        "n_even": m["n_even"], "n_odd": m["n_odd"], "closed": m["closed"],
        "histogram": np.bincount(codes, minlength=8),
        "pixel_size": s,
    }


def verify_analysis(res, image_shape):
    """Cross-check the C++ numbers against OpenCV (independent implementation).
    Returns a list of (name, ours, reference, abs difference)."""
    h, w = image_shape[:2]
    s = res["pixel_size"]
    pts = res["boundary"].astype(np.int32).reshape(-1, 1, 2)
    filled = np.zeros((h, w), np.uint8)
    cv.fillPoly(filled, [pts], 255)  # fills the boundary pixels too, like chain_fill
    rows = [
        ("Perimeter (chain) vs cv.arcLength", res["perimeter"], cv.arcLength(pts, True) * s),
        ("Area (chain, shoelace) vs cv.contourArea", res["area"], cv.contourArea(pts) * s * s),
        ("Area (filled pixels) vs cv.fillPoly count", res["area_pixels"], float(np.count_nonzero(filled)) * s * s),
    ]
    return [(n, a, b, abs(a - b)) for n, a, b in rows]


# ---------------------------------------------------------------- drawing
def draw_analysis(image, res, contour=None, fill_alpha=0.30, show_start=True):
    """BGR copy of the image with the filled interior, chain-code boundary and start pixel drawn."""
    vis = cv.cvtColor(image, cv.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    mask = res["mask"] > 0
    tint = np.zeros_like(vis)
    tint[:] = (0, 200, 255)
    vis[mask] = cv.addWeighted(vis, 1 - fill_alpha, tint, fill_alpha, 0)[mask]
    pts = res["boundary"].reshape(-1, 1, 2).astype(np.int32)
    cv.polylines(vis, [pts], True, (0, 255, 0), 1, cv.LINE_8)
    if contour is not None:
        cv.polylines(vis, [np.round(contour).astype(np.int32).reshape(-1, 1, 2)], True, (255, 160, 0), 1, cv.LINE_AA)
    if show_start:
        cv.circle(vis, res["start"], 5, (0, 0, 255), -1, cv.LINE_AA)
    return vis


def load_contour(path):
    """Read a contour saved by snake_wrapper.py (contour.npy or contour.csv)."""
    return np.load(path) if path.lower().endswith(".npy") else np.loadtxt(path, delimiter=",", skiprows=1)


# ---------------------------------------------------------------- self test + CLI
def selftest():
    """Sanity checks on shapes with a known answer. Returns True if everything passes."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    # 1) tutorial example: square 0..4 -> chain code, perimeter 16, area 16
    sq = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], np.float32)
    r = analyze_contour(sq, (10, 10))
    check("square chain code", r["chain_string"] == "0000" "6666" "4444" "2222", r["chain_string"])
    check("square perimeter = 16", abs(r["perimeter"] - 16) < 1e-9, f"{r['perimeter']:.3f}")
    check("square area = 16", abs(r["area"] - 16) < 1e-9, f"{r['area']:.3f}")
    check("square filled pixels = 25 (5x5 incl. boundary)", r["area_pixels"] == 25, f"{r['area_pixels']:.0f}")
    # 2) diamond made only of diagonal steps: perimeter = 8*sqrt(2)
    dm = np.array([[4, 0], [8, 4], [4, 8], [0, 4]], np.float32)
    r = analyze_contour(dm, (12, 12))
    check("diamond uses only odd codes", r["n_even"] == 0 and r["n_odd"] == 16, f"odd={r['n_odd']}")
    check("diamond perimeter = 16*sqrt(2)", abs(r["perimeter"] - 16 * np.sqrt(2)) < 1e-9, f"{r['perimeter']:.3f}")
    check("diamond area = 32", abs(r["area"] - 32) < 1e-9, f"{r['area']:.3f}")
    # 3) circle r=50: area ~ pi r^2, perimeter ~ 2 pi r (chain-code length overestimates a little)
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circ = np.stack([100 + 50 * np.cos(t), 100 + 50 * np.sin(t)], 1).astype(np.float32)
    r = analyze_contour(circ, (200, 200))
    check("circle area within 2% of pi r^2", abs(r["area"] / (np.pi * 2500) - 1) < 0.02, f"{r['area']:.1f} vs {np.pi * 2500:.1f}")
    # 8-connected chain-code length is a known over-estimator (up to ~6.6% on circles): staircase steps
    check("circle perimeter within 7% of 2 pi r (chain-code bias)", 0 <= r["perimeter"] / (2 * np.pi * 50) - 1 < 0.07,
          f"{r['perimeter']:.1f} vs {2 * np.pi * 50:.1f}")
    # 4) independent check against OpenCV and closure
    check("chain code is closed", r["closed"])
    for name, a, b, d in verify_analysis(r, (200, 200)):
        check(name, d <= max(1e-6, 1e-3 * abs(b)), f"{a:.3f} vs {b:.3f}")
    # 5) rotation invariance of the differential code (rotate by 90 degrees = add 2 to every code)
    c = r["chain"]
    check("differential code is rotation invariant",
          np.array_equal(chain_differential(c), chain_differential((c + 2) % 8)))
    # 6) spur removal: a contour that goes out and back over the same pixel gets the same chain code
    spur = np.array([[0, 0], [4, 0], [8, 0], [4, 0], [4, 4], [0, 4]], np.float32)
    check("spurs are removed", analyze_contour(spur, (10, 10))["chain_string"] == "0000" "6666" "4444" "2222")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Chain code, perimeter and area of a closed contour")
    ap.add_argument("image", nargs="?")
    ap.add_argument("--contour", help="contour.npy / contour.csv written by snake_wrapper.py")
    ap.add_argument("--pixel-size", type=float, default=1.0, help="physical size of one pixel (default 1)")
    ap.add_argument("--out", default="contour_out", help="output folder")
    ap.add_argument("--selftest", action="store_true", help="run the built-in checks and exit")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.image or not a.contour:
        ap.error("image and --contour are required (or use --selftest)")

    from snake_wrapper import load_image  # same loader the snake CLI uses
    img = load_image(a.image)
    contour = load_contour(a.contour)
    res = analyze_contour(contour, img.shape, a.pixel_size)
    os.makedirs(a.out, exist_ok=True)
    cv.imwrite(os.path.join(a.out, "analysis.png"), draw_analysis(img, res, contour))
    cv.imwrite(os.path.join(a.out, "mask.png"), res["mask"])
    with open(os.path.join(a.out, "chain_code.txt"), "w") as f:
        f.write(f"start (x, y): {res['start'][0]}, {res['start'][1]}\nlength: {res['length']}\n")
        f.write(f"chain code:\n{res['chain_string']}\ndifferential code:\n{''.join(map(str, res['differential'].tolist()))}\n")
    summary = {k: res[k] for k in ("start", "length", "perimeter", "area", "area_pixels", "polygon_perimeter",
                                    "polygon_area", "n_even", "n_odd", "closed", "pixel_size")}
    summary["histogram"] = res["histogram"].tolist()
    summary["verification"] = [{"check": n, "ours": x, "opencv": y, "abs_diff": d} for n, x, y, d in verify_analysis(res, img.shape)]
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"start={res['start']}  chain length={res['length']}  closed={res['closed']}")
    print(f"perimeter (chain)={res['perimeter']:.2f}   area (shoelace)={res['area']:.1f}   area (pixels)={res['area_pixels']:.0f}")
    for n, x, y, d in verify_analysis(res, img.shape):
        print(f"  check: {n}: {x:.3f} vs {y:.3f} (diff {d:.3g})")
    print(f"saved -> {a.out}/analysis.png, mask.png, chain_code.txt, results.json")


if __name__ == "__main__":
    main()
