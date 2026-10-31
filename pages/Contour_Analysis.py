"""Person 4 - Contour analysis page: snake contour -> chain code, perimeter, area (+ verification).

Streamlit picks this file up automatically as a second page of `streamlit run canny_app.py`
(it can also be run on its own: `streamlit run pages/Contour_Analysis.py`).
The heavy lifting is in contour_engine.cpp; this file only does the UI.
"""
import os
import sys

import cv2 as cv
import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(layout="wide", page_title="Contour Analysis")
st.title("Contour Analysis: chain code, perimeter and area")

try:
    import contour_wrapper as cw
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()
try:
    import snake_wrapper as sw  # Person 3 engine, only needed when the contour comes from a snake run
except FileNotFoundError as err:
    sw = None
    snake_error = str(err)


def decode_image(data):
    image = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    return np.ascontiguousarray(image if image.dtype == np.uint8 else cv.convertScaleAbs(image))


def show(image, caption):
    st.image(image, channels="GRAY" if image.ndim == 2 else "BGR", caption=caption)


@st.cache_data(show_spinner=False, max_entries=16)
def run_snake_cached(image, init, alpha, beta, gamma, balloon, window, max_iter, sigma):
    result = sw.run_snake(image, init, alpha, beta, gamma, balloon, window, max_iter, 0, sigma)
    return {k: result[k] for k in ("contour", "iterations", "converged", "mean_gradient")}


with st.sidebar:
    uploaded = st.file_uploader("Upload grayscale or color image", type=["png", "jpg", "jpeg", "bmp"])
    source = st.radio("Contour source", ["Run snake (Person 3)", "Load saved contour (.npy / .csv)"])
    pixel_size = st.number_input("Pixel size (1 = report in pixels)", 0.0001, 1000.0, 1.0, format="%.4f")

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()
image = decode_image(uploaded.getvalue())
if image is None:
    st.error("Could not decode the uploaded image.")
    st.stop()
H, W = image.shape[:2]

contour, init, snake_info = None, None, None
if source.startswith("Run snake"):
    if sw is None:
        st.error(snake_error)
        st.stop()
    with st.sidebar:
        st.subheader("Initial contour")
        shape = st.selectbox("Shape", ["Circle", "Ellipse", "Rectangle"])
        n_points = st.slider("Number of points", 20, 400, 100, 10)
        if shape == "Rectangle":
            x0, x1 = st.slider("x range", 0, W - 1, (W // 4, 3 * W // 4))
            y0, y1 = st.slider("y range", 0, H - 1, (H // 4, 3 * H // 4))
            init = sw.init_rect(x0, y0, x1, y1, n_points)
        else:
            cx = st.slider("Center x", 0, W - 1, W // 2)
            cy = st.slider("Center y", 0, H - 1, H // 2)
            if shape == "Circle":
                r = st.slider("Radius", 5, max(6, min(H, W) // 2), max(6, min(H, W) // 3))
                init = sw.init_circle(cx, cy, r, n_points)
            else:
                rx = st.slider("Radius x", 5, max(6, W // 2), max(6, W // 3))
                ry = st.slider("Radius y", 5, max(6, H // 2), max(6, H // 3))
                init = sw.init_ellipse(cx, cy, rx, ry, n_points)
        st.subheader("Snake parameters")
        alpha = st.slider("Alpha (continuity)", 0.0, 5.0, 1.0, 0.1)
        beta = st.slider("Beta (curvature)", 0.0, 5.0, 1.0, 0.1)
        gamma = st.slider("Gamma (edge)", 0.0, 5.0, 1.5, 0.1)
        balloon = st.slider("Balloon (+ inflate / - deflate)", -3.0, 3.0, 0.0, 0.1)
        window = st.select_slider("Search half-window", [1, 2, 3], value=2)
        max_iter = st.slider("Max iterations", 10, 1000, 300, 10)
        sigma = st.slider("Gaussian sigma", 0.0, 6.0, 2.0, 0.5)
    snake_info = run_snake_cached(image, init, alpha, beta, gamma, balloon, window, max_iter, sigma)
    contour = snake_info["contour"]
else:
    saved = st.sidebar.file_uploader("Contour file", type=["npy", "csv"])
    if saved is None:
        st.info("Upload a contour.npy or contour.csv written by snake_wrapper.py.")
        st.stop()
    try:
        import io
        contour = np.load(io.BytesIO(saved.getvalue())) if saved.name.endswith(".npy") else \
            np.loadtxt(io.StringIO(saved.getvalue().decode()), delimiter=",", skiprows=1)
        contour = np.asarray(contour, np.float32).reshape(-1, 2)
    except Exception as err:  # malformed file
        st.error(f"Could not read the contour file: {err}")
        st.stop()

try:
    res = cw.analyze_contour(contour, image.shape, pixel_size)
except ValueError as err:
    st.error(str(err))
    st.stop()
unit = "px" if pixel_size == 1.0 else "units"

metrics = st.columns(5)
metrics[0].metric(f"Perimeter ({unit})", f"{res['perimeter']:.2f}")
metrics[1].metric(f"Area, shoelace ({unit}²)", f"{res['area']:.1f}")
metrics[2].metric(f"Area, pixel count ({unit}²)", f"{res['area_pixels']:.0f}")
metrics[3].metric("Chain length", res["length"])
metrics[4].metric("Closed chain", "Yes" if res["closed"] else "No")

tabs = st.tabs(["Result", "Chain code", "Verification"])
with tabs[0]:
    a, b = st.columns(2)
    with a:
        if init is not None:
            show(sw.draw_contour(image, init, (255, 160, 0), 2, False), "Initial contour")
        else:
            show(image, "Original")
    with b:
        show(cw.draw_analysis(image, res, contour), "Final contour (orange), chain-code boundary (green), interior, start pixel (red)")
    if snake_info is not None:
        st.caption(f"Snake: {snake_info['iterations']} iterations, "
                   f"{'converged' if snake_info['converged'] else 'stopped at max iterations'}, "
                   f"mean edge strength {snake_info['mean_gradient']:.0f}/255.")
    ok, encoded = cv.imencode(".png", cw.draw_analysis(image, res, contour))
    if ok:
        st.download_button("Download result image", encoded.tobytes(), "contour_analysis.png", "image/png")
with tabs[1]:
    st.write(f"Start pixel (x, y): **{res['start'][0]}, {res['start'][1]}**. Freeman 8-direction code: "
             "0 right, 1 up-right, 2 up, 3 up-left, 4 left, 5 down-left, 6 down, 7 down-right.")
    chain_text = res["chain_string"]
    diff_text = "".join(map(str, res["differential"].tolist()))
    st.text_area("Chain code", chain_text, height=140)
    st.text_area("First difference (rotation invariant)", diff_text, height=140)
    report = (f"start (x, y): {res['start'][0]}, {res['start'][1]}\nlength: {res['length']}\n"
              f"perimeter: {res['perimeter']:.4f}\narea (shoelace): {res['area']:.4f}\n"
              f"area (pixels): {res['area_pixels']:.0f}\nchain code:\n{chain_text}\n"
              f"differential code:\n{diff_text}\n")
    st.download_button("Download chain code report (.txt)", report, "chain_code.txt", "text/plain")
    st.bar_chart({"count": {name: int(v) for name, v in zip(cw.DIRECTION_NAMES, res["histogram"])}})
    st.caption(f"{res['n_even']} straight steps (length 1) and {res['n_odd']} diagonal steps (length √2): "
               f"perimeter = {res['n_even']} + √2 · {res['n_odd']}.")
with tabs[2]:
    rows = [{"Check": n, "Ours (C++)": round(x, 3), "OpenCV": round(y, 3), "Abs. difference": round(d, 6)}
            for n, x, y, d in cw.verify_analysis(res, image.shape)]
    st.dataframe(rows, hide_index=True)
    st.subheader("Raw snake polygon vs chain code")
    st.dataframe([
        {"Measure": "Perimeter", "Snake polygon": round(res["polygon_perimeter"], 3), "Chain code": round(res["perimeter"], 3)},
        {"Measure": "Area", "Snake polygon": round(res["polygon_area"], 3), "Chain code (shoelace)": round(res["area"], 3),
         "Filled pixels": round(res["area_pixels"], 1)},
    ], hide_index=True)
    st.caption("Chain-code perimeter is slightly larger than the polygon perimeter (up to about 6.6% on round shapes) "
               "because a digital curve is a staircase. The shoelace area counts the boundary pixels as half; "
               "the pixel count counts them fully, so it is larger by about half the boundary length.")
