# Computer Vision Workbench

A Streamlit application for experimenting with computer vision algorithms. The
user interface is written in Python, while the image-processing engine is
implemented in C++ and loaded as a shared library.

## What the app does

Upload a grayscale or color image and choose one of the two available parts:

### Canny & Hough

- Runs a Canny edge detector with configurable Gaussian smoothing and
  thresholds.
- Displays the Canny stages:
  - Gaussian smoothing
  - Sobel gradient magnitude
  - Non-maximum suppression
  - Final hysteresis edges
- Detects lines, circles, and ellipses using configurable Hough-transform
  parameters.
- Shows detected shapes overlaid on the original image.
- Displays Hough accumulators and detected-shape data.
- Allows the superimposed result to be downloaded as a PNG.

### Snake & Contour

- Creates an initial circle, ellipse, or rectangle around an object.
- Lets you place the initial contour by dragging on the image or by using
  sliders.
- Evolves the contour using a greedy active-contour (snake) algorithm.
- Displays the initial and final contours, convergence information, and edge
  strength.
- Generates an 8-direction Freeman chain code.
- Calculates contour perimeter and area using both shoelace geometry and pixel
  counting.
- Supports pixel-size settings and result comparison.

The **Compare** tab can pin a result, change a setting, and show the two
configurations side by side.

## Requirements

- Windows, macOS, or Linux
- Python 3
- CMake 3.15 or newer
- A C++17 compiler
- Python packages listed in [`requirements.txt`](requirements.txt)

On Windows with MinGW, make sure `cmake` and `g++` are available in the
terminal. The application expects the compiled library to be named
`engine.dll` and to be located beside `app.py`.

## Quickstart

Open PowerShell in the project directory:

```powershell
cd "D:\College stuff\Computer Vision\Task 2\CV-Task-2"
```

### 1. Create and activate a Python environment

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands from a terminal where the
virtual environment is already active, or use:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Build the C++ engine

For MinGW on Windows:

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --target engine --parallel 4
```

This creates `engine.dll` beside `app.py`.

For an already configured build directory, the shorter command is:

```powershell
cmake --build build --target engine --parallel 4
```

### 3. Start the app

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, normally:

```text
http://localhost:8501
```

Open that URL in a browser, upload an image, and select a processing part from
the sidebar.

## Troubleshooting

### `engine.dll not found next to app.py`

Build the C++ target from the project root:

```powershell
cmake --build build --target engine
```

Confirm that `engine.dll` exists in the same folder as `app.py`.

### The DLL cannot be loaded

Ensure that the DLL was built with a compiler compatible with the current
environment. For MinGW builds, the application can use the MinGW runtime from
`C:\msys64\mingw64\bin`. If MinGW is installed elsewhere, set `MINGW64_BIN`
before starting Streamlit:

```powershell
$env:MINGW64_BIN = "C:\path\to\mingw64\bin"
python -m streamlit run app.py
```

### The app opens but no image is processed

Upload a supported image format: PNG, JPG/JPEG, or BMP. In the Snake &
Contour part, place the initial contour around the object and press **Run
snake**.

## Project layout

```text
app.py             Streamlit frontend and Python/C++ bridge
engine.cpp         C++ computer vision engine
CMakeLists.txt     CMake build configuration
requirements.txt   Python dependencies
engine.dll         Windows build output, generated beside app.py
build/             CMake build directory
```
