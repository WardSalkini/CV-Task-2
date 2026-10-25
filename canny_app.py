import streamlit as st
import cv2 as cv
import numpy as np
import ctypes
import os
import sys

# =====================================================================
# C++ ENGINE LINKING
# =====================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))

dll_directories = []  # keep DLL search directories alive
if os.name == "nt":
    opencv_bin = r"C:\opencv\opencv\build\x64\vc16\bin"
    if os.path.isdir(opencv_bin):
        dll_directories.append(os.add_dll_directory(opencv_bin))
    mingw_bin = os.environ.get("MINGW64_BIN", r"C:\msys64\mingw64\bin")
    if os.path.isdir(mingw_bin):
        dll_directories.append(os.add_dll_directory(mingw_bin))
    lib_name = "canny_engine.dll"
elif sys.platform == "darwin":
    lib_name = "canny_engine.dylib"
else:
    lib_name = "canny_engine.so"

lib_path = os.path.join(base_dir, lib_name)
if not os.path.isfile(lib_path):
    raise FileNotFoundError(f"{lib_name} not found at: {lib_path}")

cpp = ctypes.CDLL(lib_path)

U8P = ctypes.POINTER(ctypes.c_uint8)
F32P = ctypes.POINTER(ctypes.c_float)

cpp.canny_run.restype = None
cpp.canny_run.argtypes = [
    U8P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,   # image, rows, cols, channels, color mode
    ctypes.c_int, ctypes.c_float,                                  # Gaussian kernel size, sigma
    ctypes.c_float, ctypes.c_float, ctypes.c_int,                  # low, high, auto threshold (always 0 here)
    U8P, U8P, U8P, U8P, U8P, F32P                                  # smooth, mag, nms, edges, overlay, info
]


def _p(array):
    return array.ctypes.data_as(U8P)


# =====================================================================
# PYTHON <-> C++ WRAPPER (no image processing happens in Python)
# =====================================================================
@st.cache_data(show_spinner=False, max_entries=16)
def run_canny(img, ksize, sigma, low, high):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else 3
    smooth = np.empty((h, w), np.uint8)
    mag = np.empty((h, w), np.uint8)
    nms = np.empty((h, w), np.uint8)
    edges = np.empty((h, w), np.uint8)
    overlay = np.empty((h, w, 3), np.uint8)
    info = np.zeros(4, np.float32)
    # color_mode 0 = convert color images to gray first; auto threshold = 0 (manual)
    cpp.canny_run(_p(img), h, w, channels, 0,
                  int(ksize), float(sigma), float(low), float(high), 0,
                  _p(smooth), _p(mag), _p(nms), _p(edges), _p(overlay),
                  info.ctypes.data_as(F32P))
    return {"edges": edges, "overlay": overlay, "count": int(info[2])}


def show(col, title, img):
    with col:
        st.subheader(title)
        st.image(img, channels="BGR" if img.ndim == 3 else "GRAY", width="stretch")


# =====================================================================
# PAGE
# =====================================================================
st.set_page_config(layout="wide", page_title="Canny Studio")
st.title("Canny Edge Detection (C++ Engine)")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"])
    st.markdown("**Canny parameters**")
    ksize = st.select_slider("Gaussian kernel size", options=[3, 5, 7, 9, 11], value=5)
    sigma = st.slider("Gaussian sigma", 0.5, 5.0, 1.4, 0.1)
    low = st.slider("Low threshold", 0, 255, 20,
                    help="Scale 0-255, relative to the strongest gradient in the image.")
    high = st.slider("High threshold", 0, 255, 60)

if uploaded_file is None:
    st.info("Upload a gray or color image from the sidebar to start.")
    st.stop()

raw = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
img = cv.imdecode(raw, cv.IMREAD_UNCHANGED)
if img is None:
    st.error("Could not read the uploaded image.")
    st.stop()
if img.dtype != np.uint8:
    img = cv.convertScaleAbs(img, alpha=255.0 / max(float(img.max()), 1.0))
if img.ndim == 3 and img.shape[2] == 4:
    img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
img = np.ascontiguousarray(img)
st.caption(f"{'Color (BGR)' if img.ndim == 3 else 'Gray'} image, {img.shape[1]} x {img.shape[0]} px")

res = run_canny(img, ksize, sigma, low, high)

m1, m2 = st.columns(2)
m1.metric("Edge pixels", f"{res['count']:,}")
m2.metric("Edge density", f"{100.0 * res['count'] / (img.shape[0] * img.shape[1]):.2f}%")

a, b, c = st.columns(3)
show(a, "Original", img)
show(b, "Detected edges", res["edges"])
show(c, "Edges over original", res["overlay"])