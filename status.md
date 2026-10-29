# Assignment 2 Status

## Scope

This document records the current structure of the project and compares the
implemented functionality with the requirements shown in the assignment image.

The assignment asks for:

1. Edge detection for grayscale and color images using Canny, followed by
   detection of lines, circles, and ellipses (where present), with the
   detected shapes superimposed on the input image.
2. Initialization and greedy evolution of an Active Contour Model (snake) for a
   given object, with the result represented as chain code and with contour
   perimeter and enclosed area calculated.

## Current Project Structure

```text
CV-Task-2/
├── canny_engine.cpp       # Native C++ image-processing engine
├── canny_engine.dll       # Prebuilt Windows shared library
├── canny_app.py           # Streamlit Python front end and ctypes bridge
├── CMakeLists.txt         # CMake build configuration
├── requirements.txt       # Python dependencies
├── README.md              # Minimal project readme
├── build-log.txt          # Existing build log (currently empty)
├── syntax-log.txt         # Existing syntax log (currently empty)
├── build/                 # Generated CMake files and compiler metadata
├── .vscode/               # Local editor configuration
├── venv/                  # Local Python environment
└── tempCodeRunnerFile.py  # Temporary editor-generated file
```

Generated directories and the prebuilt DLL are environment/build artifacts;
the main source implementation consists of `canny_engine.cpp` and
`canny_app.py`.

## C++ Engine

### Public native API

The C++ file exports a C-compatible API for use from Python:

- `canny_run`: runs the custom Canny pipeline and returns intermediate images,
  the binary edge map, an RGB/BGR overlay, and summary values.
- `canny_opencv`: reference implementation using OpenCV's Canny function.
- `edge_similarity`: computes intersection-over-union similarity between two
  binary edge maps.

### Canny pipeline

`CannyDetector` currently implements the following stages from scratch:

1. Accepts one-channel grayscale input or three-channel BGR input.
2. Converts color input to grayscale by default.
3. Builds a normalized Gaussian kernel and applies separable smoothing.
4. Computes Sobel `gx`, `gy`, and gradient magnitude.
5. Normalizes the magnitude to the range 0--255.
6. Applies non-maximum suppression using four quantized directions.
7. Applies manual double thresholds.
8. Tracks edges by hysteresis using 8-connectivity.
9. Produces a BGR overlay with detected edges painted red.
10. Reports the thresholds used, edge-pixel count, and maximum gradient
    magnitude.

The engine also contains an optional automatic-threshold path based on the
median of surviving non-maximum-suppression pixels, although the current
Python UI always passes manual-threshold mode.

### Current C++ limitations

The engine does **not** currently implement:

- Hough or other line detection.
- Circle detection.
- Ellipse detection.
- Active Contour Model/snake initialization or greedy evolution.
- Chain-code generation.
- Contour perimeter calculation.
- Contour enclosed-area calculation.

Therefore, the C++ implementation currently covers the Canny edge-map portion
only, not the complete assignment.

## Python Front End

`canny_app.py` is a Streamlit application named **Canny Studio**.

### Input and UI

- Allows upload of PNG, JPG, JPEG, and BMP images.
- Accepts grayscale and color images.
- Converts unsupported image depth to 8-bit.
- Converts BGRA images to BGR.
- Provides controls for:
  - Gaussian kernel size: 3, 5, 7, 9, or 11.
  - Gaussian sigma.
  - Low threshold.
  - High threshold.

### Python-to-C++ data flow

1. Streamlit receives the uploaded file.
2. OpenCV decodes it into a NumPy array.
3. The array is made contiguous and converted to `uint8`.
4. `ctypes.CDLL` loads the platform-specific native library.
5. `run_canny` allocates NumPy output buffers.
6. `canny_run` fills those buffers in the C++ engine.
7. Streamlit displays the original image, binary edge map, and edge overlay.
8. The UI displays edge-pixel count and edge density.

The Python layer does not perform the image-processing algorithm itself. It
acts as the UI, input-normalization layer, output-buffer owner, and native
library bridge. Results are cached with Streamlit's `st.cache_data`.

### Current Python limitations

The front end currently exposes only Canny functionality. It has no UI or
binding for:

- Selecting or initializing an object contour.
- Running the greedy snake/Active Contour algorithm.
- Displaying detected lines, circles, or ellipses.
- Displaying chain code.
- Displaying perimeter or area.
- Choosing between the custom Canny result and the OpenCV reference result.
- Calling `canny_opencv` or `edge_similarity`.

## Requirement Status

| Requirement from assignment | Status | Evidence / remaining work |
|---|---|---|
| Accept grayscale images | **Done** | Python decodes grayscale input and passes one channel to `canny_run`. |
| Accept color images | **Done** | Python accepts BGR/BGRA images; the engine supports three-channel input. |
| Detect edges using Canny | **Partially done** | Custom Gaussian, Sobel, NMS, thresholding, and hysteresis stages are implemented. The result is available in the UI. |
| Detect lines | **Done** | Custom polar accumulator, 2D peak suppression, clipped endpoints, and exported result buffers are in `canny_engine.cpp`. |
| Detect circles | **Done** | Custom center/radius voting, center projection, distance suppression, and exported result buffers are in `canny_engine.cpp`. |
| Detect ellipses, if present | **Done** | Pairwise/1D semi-minor-axis voting with edge subsampling and exported ellipse results is in `canny_engine.cpp`. |
| Superimpose detected shapes on images | **Done** | C++ renders lines, circles/center marks, and rotated ellipses over the edge overlay. |
| Initialize a contour for a given object | **Not done** | No contour input/selection or initialization API exists. |
| Evolve an Active Contour Model using greedy algorithm | **Not done** | No snake representation, energy function, neighborhood search, or iteration loop exists. |
| Represent the contour as chain code | **Not done** | No chain-code encoding or output exists. |
| Compute contour perimeter | **Not done** | No perimeter calculation exists. |
| Compute enclosed contour area | **Not done** | No area calculation exists. |
| Provide a usable front end | **Done** | Streamlit exposes Canny comparison, shape controls, diagnostics, tables, timing, and overlay download. |

## Expanded Requirement 1 Specification

The pasted implementation brief adds the following acceptance criteria for
Requirement 1. They are included here so the implementation can be checked
against the detailed specification, not only the assignment slide.

### C++ implementation constraints

- All line, circle, and ellipse Hough algorithms must be implemented from
  scratch in `canny_engine.cpp`.
- Accumulator voting, peak extraction, and non-maximum suppression must be
  implemented in C++.
- High-level detectors such as `cv::HoughLines`, `cv::HoughLinesP`,
  `cv::HoughCircles`, and `cv::fitEllipse` must not be used.
- OpenCV may be used for image decoding/support and, if needed, basic
  rendering, but custom pixel drawing is preferred.

### Required Hough line implementation

- Polar parameterization using a consistent `(rho, theta)` convention.
- Configurable rho and theta resolutions.
- A 2D `(rho, theta)` accumulator with precomputed trigonometric tables.
- Voting from every Canny edge pixel.
- Configurable 2D accumulator peak suppression and vote threshold/top-K
  selection.
- Normalized accumulator output for visualization.
- Detected line parameters, vote counts, and image-boundary-clipped endpoints.

### Required Hough circle implementation

- Configurable center/radius parameter space and radius step.
- Standard polar voting and/or gradient-directed voting.
- Configurable vote threshold.
- 3D peak suppression using minimum center distance and radius separation.
- Detected `(center_x, center_y, radius, votes)` results.
- A 2D center-accumulator projection for visualization.

### Required Hough ellipse implementation

- Implement the efficient pairwise/1D accumulator approach described in the
  brief (Xie & Ji-style), rather than a naive 5D accumulator.
- Extract edge points and optionally subsample them.
- Treat valid edge-point pairs as candidate major-axis endpoints.
- Accumulate possible semi-minor axes from third edge points.
- Apply vote thresholding and candidate suppression or inlier removal.
- Return ellipse center, semi-major axis, semi-minor axis, orientation, and
  votes.
- Include a performance safeguard such as a maximum sampled edge-point count;
  OpenMP is optional.

### Required shape rendering

- Render detected lines, circles, and ellipses on a BGR image in C++.
- Use distinguishable colors and configurable line thickness.
- Draw circle center markers and ellipse center markers.
- Optionally retain the red Canny edges underneath the geometric shapes.

### Required Python UI and integration

- Expose `canny_opencv`, `edge_similarity`, and automatic threshold mode
  through `ctypes` and the UI.
- Add independent enable/disable controls for lines, circles, and ellipses.
- Add parameter controls for each enabled detector.
- Display edge count, optional Canny IoU, shape counts, and execution time.
- Display the original image, binary edge map, and final superimposition.
- Provide a download action for the superimposed image.
- Display line accumulator, circle center accumulator, and Canny intermediate
  stages for diagnostics.
- Display interactive tables containing the mathematical parameters and votes
  for all detected shapes.
- Keep all buffer sizes, channel ordering, coordinate systems, and native
  function signatures consistent across C++ and Python.

### Expanded specification status

| Expanded item | Status |
|---|---|
| Custom Hough line transform | **Done** |
| Line accumulator output and visualization | **Done** |
| Custom Hough circle transform | **Done** |
| Circle accumulator projection | **Done** |
| Efficient pairwise/1D ellipse transform | **Done** |
| Hough peak suppression and configurable thresholds | **Done** |
| C++ shape rendering and shape overlay | **Done** |
| Canny/OpenCV comparison in Python | **Done** |
| Automatic-threshold control in Python | **Done** |
| Hough detector controls in Streamlit | **Done** |
| KPI metrics, diagnostics tabs, and shape tables | **Done** |
| Superimposed-image download | **Done** |

## Overall Status

**Current completion: Requirement 1 implementation complete in source; native
build/runtime verification pending.**

Requirement 1 source implementation now includes Canny, custom Hough shape
detection, C++ superimposition, and the expanded Python UI. Requirement 2
(Active Contours, chain code, perimeter, and area) is still not implemented.

## Required Work

### 1. Fix the reproducible build

- Update `CMakeLists.txt` to build the actual source file,
  `canny_engine.cpp`. It currently refers to `engine.cpp`, which is not
  present in the repository.
- Rebuild the DLL from the corrected CMake target.
- Confirm that the rebuilt DLL exports the functions expected by
  `canny_app.py`.
- Document the required OpenCV and compiler paths instead of relying only on
  local machine paths.

### 2. Validate and tune geometric shape detection

- Exercise the custom Hough line, circle, and efficient ellipse transforms on
  representative images.
- Tune thresholds and peak suppression against the supplied images.
- Add Canny intermediate-stage buffers to the diagnostics tab if required by
  the final grading rubric.
- Test on both grayscale and color images, including images containing no
  detectable shapes.

### 3. Implement Active Contours

- Add a way to initialize a closed contour around or near a selected object.
- Implement the greedy snake update using a documented energy model.
- Add stopping criteria and iteration controls.
- Render the evolving/final contour on the image.
- Convert the final contour to chain code.
- Calculate and display perimeter and enclosed area.
- Decide how the contour is supplied from the UI (for example, polygon
  points, rectangle initialization, or an interactive selection tool).

### 4. Integrate and validate

- Add Python `ctypes` declarations for every new native function.
- Add UI controls for each algorithm and its parameters.
- Verify buffer sizes, channel ordering, coordinate conventions, and ownership
  across the C++/Python boundary.
- Add representative test images and record expected outputs.
- Compare the custom Canny output against the existing OpenCV reference path
  using `edge_similarity`.
- Update `README.md` with build, run, and usage instructions.

## Important Build Note

The source/build mismatch has been fixed. MSYS2 OpenCV is available at
`C:\msys64\mingw64`, and the native build was verified with:

```cmake
cmake -S . -B build-msys2 -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -DOpenCV_DIR=C:/msys64/mingw64/lib/cmake/opencv5
cmake --build build-msys2 --parallel 2
```

The rebuilt DLL exports `canny_run`, `canny_opencv`, `edge_similarity`, and
`canny_detect_shapes`. A direct DLL-load/export smoke test passed with
`C:\msys64\mingw64\bin` added to the DLL search path.
