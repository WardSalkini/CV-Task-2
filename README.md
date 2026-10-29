# CV-Task-2

## Computer Vision Workbench

The project contains a custom C++ image-processing engine and a Streamlit
front end. Requirement 1 now includes:

- Custom Canny edge detection for grayscale and BGR images.
- From-scratch polar Hough line detection.
- From-scratch circle accumulator voting.
- Pairwise/1D ellipse candidate detection with edge-point subsampling.
- C++ rendering of detected lines, circles, and ellipses.
- Python controls, accumulator visualizations, result tables, Canny IoU
  comparison, execution timing, and overlay download.

The Active Contour Model, greedy snake evolution, chain code, perimeter, and
area work from Requirement 2 remains to be implemented. See
[`status.md`](status.md) for the complete implementation matrix.

## Quick Start

### 1. Install prerequisites

On Windows, the verified setup uses MSYS2 with the MinGW 64-bit toolchain and
OpenCV:

```powershell
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-opencv
```

Create/activate a Python environment and install the UI dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Build the C++ engine

The native engine requires a C++17 compiler, CMake 3.15 or newer, and an
OpenCV development installation discoverable through `OpenCV_DIR` or
`CMAKE_PREFIX_PATH`.

```powershell
cmake -S . -B build-msys2 -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -DOpenCV_DIR=C:/msys64/mingw64/lib/cmake/opencv5
cmake --build build-msys2 --parallel 2
```

The build produces `canny_engine.dll` beside `canny_app.py` on Windows (or
the corresponding `.so`/`.dylib` on other platforms).

### 3. Run the Streamlit app

```powershell
streamlit run canny_app.py
```

The app automatically adds `C:\msys64\mingw64\bin` to the Windows DLL search
path when that directory exists. If MSYS2 is installed elsewhere, set
`MINGW64_BIN` before launching:

```powershell
$env:MINGW64_BIN = "D:\path\to\msys64\mingw64\bin"
streamlit run canny_app.py
```

Open the displayed local URL, upload a grayscale or color image, and enable
the desired Canny, line, circle, or ellipse detectors in the sidebar.

## Repository artifacts

Generated CMake directories, native binaries, Python environments, caches,
logs, and temporary editor files are excluded by `.gitignore`. Rebuild the
native DLL locally after cloning; it is intentionally not committed.