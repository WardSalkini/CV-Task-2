import ctypes
import os
import sys
import time

import cv2 as cv
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.name == "nt":
    for path in (
        r"C:\opencv\opencv\build\x64\vc16\bin",
        os.environ.get("MINGW64_BIN", r"C:\msys64\mingw64\bin"),
    ):
        if os.path.isdir(path):
            os.add_dll_directory(path)
    LIB_NAME = "canny_engine.dll"
elif sys.platform == "darwin":
    LIB_NAME = "canny_engine.dylib"
else:
    LIB_NAME = "canny_engine.so"

LIB_PATH = os.path.join(BASE_DIR, LIB_NAME)
if not os.path.isfile(LIB_PATH):
    raise FileNotFoundError(f"{LIB_NAME} not found at {LIB_PATH}. Build the CMake target first.")
cpp = ctypes.CDLL(LIB_PATH)
U8P = ctypes.POINTER(ctypes.c_uint8)
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int)

cpp.canny_run.restype = None
cpp.canny_run.argtypes = [
    U8P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
    U8P, U8P, U8P, U8P, U8P, F32P,
]
cpp.canny_opencv.restype = None
cpp.canny_opencv.argtypes = [
    U8P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, U8P,
]
cpp.edge_similarity.restype = ctypes.c_float
cpp.edge_similarity.argtypes = [U8P, U8P, ctypes.c_int]
cpp.canny_detect_shapes.restype = None
cpp.canny_detect_shapes.argtypes = [
    U8P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    U8P, U8P, U8P, ctypes.c_int, I32P, I32P, U8P, F32P, I32P,
    F32P, I32P, F32P, I32P, F32P,
]


def ptr(array):
    return array.ctypes.data_as(U8P)


@st.cache_data(show_spinner=False, max_entries=16)
def run_pipeline(image, ksize, sigma, low, high, auto_threshold, flags,
                 rho_step, theta_step, line_threshold, line_nms, max_lines,
                 rmin, rmax, rstep, circle_threshold, min_dist, max_circles,
                 gradient_mode, min_major, max_major, min_minor, ellipse_threshold,
                 max_points, max_ellipses, thickness):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else 3
    edges = np.empty((h, w), np.uint8)
    overlay = np.empty((h, w, 3), np.uint8)
    line_capacity = 4096 * 360
    line_acc = np.zeros(line_capacity, np.uint8)
    circle_acc = np.zeros((h, w), np.uint8)
    lines = np.zeros((max_lines, 7), np.float32)
    circles = np.zeros((max_circles, 4), np.float32)
    ellipses = np.zeros((max_ellipses, 6), np.float32)
    line_w = ctypes.c_int()
    line_h = ctypes.c_int()
    line_count = ctypes.c_int()
    circle_count = ctypes.c_int()
    ellipse_count = ctypes.c_int()
    info = np.zeros(4, np.float32)
    cpp.canny_detect_shapes(
        ptr(image), h, w, channels, int(ksize), float(sigma), float(low), float(high),
        int(auto_threshold), int(flags), float(rho_step), float(theta_step),
        int(line_threshold), int(line_nms), int(max_lines), int(rmin), int(rmax),
        int(rstep), int(circle_threshold), int(min_dist), int(max_circles),
        int(gradient_mode),
        int(min_major), int(max_major), int(min_minor), int(ellipse_threshold),
        int(max_points), int(max_ellipses), int(thickness), ptr(edges), ptr(overlay), ptr(line_acc),
        line_capacity, ctypes.byref(line_w), ctypes.byref(line_h), ptr(circle_acc),
        lines.ctypes.data_as(F32P), ctypes.byref(line_count),
        circles.ctypes.data_as(F32P), ctypes.byref(circle_count),
        ellipses.ctypes.data_as(F32P), ctypes.byref(ellipse_count), info.ctypes.data_as(F32P),
    )
    return {
        "edges": edges,
        "overlay": overlay,
        "lines": lines[:line_count.value].copy(),
        "circles": circles[:circle_count.value].copy(),
        "ellipses": ellipses[:ellipse_count.value].copy(),
        "line_acc": line_acc[:line_w.value * line_h.value].reshape((line_h.value, line_w.value))
        if line_w.value and line_h.value else np.zeros((1, 1), np.uint8),
        "circle_acc": circle_acc,
        "info": info,
    }


st.set_page_config(layout="wide", page_title="Canny and Hough Workbench")
st.title("Computer Vision Workbench")
with st.sidebar:
    uploaded = st.file_uploader("Upload grayscale or color image", type=["png", "jpg", "jpeg", "bmp"])
    st.subheader("Canny")
    ksize = st.select_slider("Gaussian kernel", [3, 5, 7, 9, 11], value=5)
    sigma = st.slider("Sigma", 0.5, 5.0, 1.4, 0.1)
    auto = st.checkbox("Automatic thresholds", False)
    low = st.slider("Low threshold", 0, 255, 20, disabled=auto)
    high = st.slider("High threshold", 0, 255, 60, disabled=auto)
    compare = st.checkbox("Compare with OpenCV Canny", True)
    st.subheader("Hough detectors")
    detect_lines = st.checkbox("Detect lines")
    detect_circles = st.checkbox("Detect circles")
    detect_ellipses = st.checkbox("Detect ellipses")
    thickness = st.slider("Overlay line/shape thickness", 1, 10, 4)
    with st.expander("Line parameters", expanded=detect_lines):
        rho_step = st.number_input("Rho step", 1.0, 10.0, 1.0)
        theta_step = st.number_input("Theta step (degrees)", 0.5, 10.0, 1.0)
        line_threshold = st.number_input("Line vote threshold", 1, 1000, 80)
        line_nms = st.slider("Line NMS window", 3, 15, 5, 2)
        max_lines = st.slider("Maximum lines", 1, 64, 16)
    with st.expander("Circle parameters", expanded=detect_circles):
        rmin = st.number_input("Minimum radius", 2, 1000, 10)
        rmax = st.number_input("Maximum radius", 3, 1500, 150)
        rstep = st.number_input("Radius step", 1, 20, 2)
        circle_threshold = st.number_input("Circle vote threshold", 1, 72, 12)
        min_dist = st.number_input("Minimum center distance", 1, 1000, 20)
        max_circles = st.slider("Maximum circles", 1, 32, 8)
        gradient_mode = st.checkbox("Use gradient-guided voting", True)
    with st.expander("Ellipse parameters", expanded=detect_ellipses):
        min_major = st.number_input("Minimum major axis (diameter)", 4, 2000, 30)
        max_major = st.number_input("Maximum major axis (diameter)", 5, 3000, 300)
        min_minor = st.number_input("Minimum minor axis", 2, 1500, 10)
        ellipse_threshold = st.number_input("Ellipse vote threshold", 3, 400, 12)
        max_points = st.number_input("Maximum sampled edge points", 50, 1000, 300)
        max_ellipses = st.slider("Maximum ellipses", 1, 16, 4)

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()
raw = np.frombuffer(uploaded.getvalue(), np.uint8)
image = cv.imdecode(raw, cv.IMREAD_UNCHANGED)
if image is None:
    st.error("Could not decode the uploaded image.")
    st.stop()
if image.ndim == 3 and image.shape[2] == 4:
    image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
image = np.ascontiguousarray(image if image.dtype == np.uint8 else cv.convertScaleAbs(image))
flags = int(detect_lines) | (int(detect_circles) << 1) | (int(detect_ellipses) << 2)
start = time.perf_counter()
result = run_pipeline(
    image, ksize, sigma, low, high, auto, flags, rho_step, theta_step, line_threshold,
    line_nms, max_lines, rmin, rmax, rstep, circle_threshold, min_dist, max_circles,
    gradient_mode, min_major, max_major, min_minor, ellipse_threshold, max_points, max_ellipses,
    thickness,
)
elapsed = (time.perf_counter() - start) * 1000
edge_count = int(np.count_nonzero(result["edges"]))
smooth = np.empty(result["edges"].shape, np.uint8)
magnitude = np.empty(result["edges"].shape, np.uint8)
nms = np.empty(result["edges"].shape, np.uint8)
pipeline_info = np.zeros(4, np.float32)
cpp.canny_run(
    ptr(image), image.shape[0], image.shape[1], 1 if image.ndim == 2 else 3, 0,
    int(ksize), float(sigma), float(low), float(high), int(auto),
    ptr(smooth), ptr(magnitude), ptr(nms), ptr(np.empty_like(result["edges"])),
    ptr(np.empty((image.shape[0], image.shape[1], 3), np.uint8)),
    pipeline_info.ctypes.data_as(F32P),
)
iou = None
if compare:
    reference = np.empty_like(result["edges"])
    cpp.canny_opencv(ptr(image), image.shape[0], image.shape[1], 1 if image.ndim == 2 else 3,
                     int(ksize), float(sigma), float(result["info"][0]), float(result["info"][1]), ptr(reference))
    iou = float(cpp.edge_similarity(ptr(result["edges"]), ptr(reference), result["edges"].size))

metrics = st.columns(6)
metrics[0].metric("Edge pixels", f"{edge_count:,}")
metrics[1].metric("Canny IoU", f"{iou:.3f}" if iou is not None else "Off")
metrics[2].metric("Lines", len(result["lines"]))
metrics[3].metric("Circles", len(result["circles"]))
metrics[4].metric("Ellipses", len(result["ellipses"]))
metrics[5].metric("Execution", f"{elapsed:.1f} ms")

tabs = st.tabs(["Superimposition", "Diagnostics", "Detected data"])
with tabs[0]:
    a, b, c = st.columns(3)
    a.image(image, channels="GRAY" if image.ndim == 2 else "BGR", caption="Original")
    b.image(result["edges"], channels="GRAY", caption="Canny edges")
    c.image(result["overlay"], channels="BGR", caption="Detected shapes")
    ok, encoded = cv.imencode(".png", result["overlay"])
    if ok:
        st.download_button("Download superimposed image", encoded.tobytes(), "shapes.png", "image/png")
with tabs[1]:
    line_display = cv.resize(result["line_acc"], (600, 450), interpolation=cv.INTER_LINEAR)
    circle_display = cv.resize(result["circle_acc"], (600, 450), interpolation=cv.INTER_LINEAR)
    use_heatmap = st.checkbox("Use heatmap diagnostics", True)
    if use_heatmap:
        line_display = cv.applyColorMap(line_display, cv.COLORMAP_INFERNO)
        circle_display = cv.applyColorMap(circle_display, cv.COLORMAP_INFERNO)
    st.image(line_display, caption="Line accumulator (rho x theta)", clamp=True)
    st.image(circle_display, caption="Circle center accumulator", clamp=True)
    d1, d2, d3 = st.columns(3)
    d1.image(smooth, caption="Gaussian smoothing", clamp=True)
    d2.image(magnitude, caption="Sobel magnitude", clamp=True)
    d3.image(nms, caption="Non-maximum suppression", clamp=True)
with tabs[2]:
    st.subheader("Lines: rho, theta, votes, endpoints")
    st.dataframe(result["lines"], use_container_width=True)
    st.subheader("Circles: center_x, center_y, radius, votes")
    st.dataframe(result["circles"], use_container_width=True)
    st.subheader("Ellipses: center_x, center_y, semi_major, semi_minor, angle, votes")
    st.dataframe(result["ellipses"], use_container_width=True)
