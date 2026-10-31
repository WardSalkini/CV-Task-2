// =====================================================================
// Person 4 - Contour analysis: Freeman chain code, perimeter and area
//
// Pure C++17, no OpenCV needed. Exposed through a C API so Python can call it with ctypes.
// It takes the final snake contour (Person 3) and turns it into:
//   1. a closed 8-connected pixel boundary  -> Freeman chain code (start point + codes 0..7)
//   2. perimeter  = (#even codes) + sqrt(2) * (#odd codes)
//   3. area       = shoelace formula evaluated on the chain code itself
//      (plus the filled-pixel count, and the same two numbers for the raw float contour)
//
// Chain code convention (Freeman, 8 directions, as seen on screen, image y axis points down):
//        (3) (2) (1)   <- up-left, up, up-right
//      (4) <-- o --> (0)
//        (5) (6) (7)   <- down-left, down, down-right
//   code : 0     1      2     3      4      5      6     7
//   (dx,dy): (1,0) (1,-1) (0,-1) (-1,-1) (-1,0) (-1,1) (0,1) (1,1)
//
// Output coordinates are (x, y) in pixel units, x = column, y = row (same as snake_engine).
// =====================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace std;

#if defined(_WIN32)
  #define API __declspec(dllexport)
#else
  #define API __attribute__((visibility("default")))
#endif

namespace {

const int DX[8] = {1, 1, 0, -1, -1, -1, 0, 1};
const int DY[8] = {0, -1, -1, -1, 0, 1, 1, 1};

// direction code of a single 8-connected step (-1 if the step is not a neighbour move)
inline int codeOf(int dx, int dy) {
    for (int c = 0; c < 8; ++c) if (DX[c] == dx && DY[c] == dy) return c;
    return -1;
}

// Append the 8-connected pixel path (a -> b), excluding a and including b (Bresenham).
void lineSteps(int x0, int y0, int x1, int y1, vector<uint8_t>& codes) {
    int dx = abs(x1 - x0), dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int err = dx - dy, x = x0, y = y0;
    while (x != x1 || y != y1) {
        int e2 = 2 * err, nx = x, ny = y;
        if (e2 > -dy) { err -= dy; nx += sx; }
        if (e2 <  dx) { err += dx; ny += sy; }
        codes.push_back((uint8_t)codeOf(nx - x, ny - y));
        x = nx; y = ny;
    }
}

// Remove spurs: a step immediately followed by its opposite (c, c+4) walks out and back over the same
// pixel and adds nothing to the shape. Handles the wrap-around too, which moves the start point.
void cancelSpurs(vector<uint8_t>& codes, int& sx, int& sy) {
    vector<uint8_t> st;
    for (uint8_t c : codes) {
        if (!st.empty() && st.back() == (c + 4) % 8) st.pop_back(); else st.push_back(c);
    }
    size_t lo = 0, hi = st.size();
    while (hi - lo >= 2 && st[lo] == (st[hi - 1] + 4) % 8) {
        sx += DX[st[lo]]; sy += DY[st[lo]];  // the path now starts one step further along
        ++lo; --hi;
    }
    codes.assign(st.begin() + lo, st.begin() + hi);
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------
// Float contour (x, y) -> closed Freeman chain code.
// xy        : n * 2 floats, ordered closed contour (e.g. the snake output)
// start_xy  : 2 ints, receives the integer start pixel (x, y)
// codes     : caller buffer for the chain code (values 0..7)
// capacity  : size of `codes`
// returns   : chain length. If it is > capacity nothing was written to `codes`; call again with a
//             bigger buffer. Returns 0 for a degenerate contour.
// ---------------------------------------------------------------------
API int contour_to_chain(const float* xy, int n, int* start_xy, unsigned char* codes, int capacity) {
    if (n < 3) return 0;
    vector<int> px(n), py(n);
    for (int i = 0; i < n; ++i) { px[i] = (int)lroundf(xy[2 * i]); py[i] = (int)lroundf(xy[2 * i + 1]); }
    int sx = px[0], sy = py[0];
    vector<uint8_t> c;
    for (int i = 0; i < n; ++i) lineSteps(px[i], py[i], px[(i + 1) % n], py[(i + 1) % n], c);  // closes the loop
    cancelSpurs(c, sx, sy);
    if (c.size() < 4) return 0;
    start_xy[0] = sx; start_xy[1] = sy;
    int len = (int)c.size();
    if (len <= capacity) for (int i = 0; i < len; ++i) codes[i] = c[i];
    return len;
}

// Rotation-invariant first difference of a chain code: d_i = (c_i - c_{i-1}) mod 8 (cyclic).
API void chain_differential(const unsigned char* codes, int len, unsigned char* out) {
    for (int i = 0; i < len; ++i) out[i] = (unsigned char)((codes[i] - codes[(i + len - 1) % len] + 8) % 8);
}

// Rebuild the boundary pixels from the chain code (len points: start, ..., last one before returning to start).
API void chain_decode(const int* start_xy, const unsigned char* codes, int len, int* out_xy) {
    int x = start_xy[0], y = start_xy[1];
    for (int i = 0; i < len; ++i) {
        out_xy[2 * i] = x; out_xy[2 * i + 1] = y;
        x += DX[codes[i] & 7]; y += DY[codes[i] & 7];
    }
}

// Measurements computed from the chain code only.
// out[0] perimeter = n_even + sqrt(2) * n_odd
// out[1] area (shoelace on the decoded boundary)
// out[2] n_even   out[3] n_odd   out[4] closed (1 if the code returns to the start pixel)
API void chain_measure(const unsigned char* codes, int len, double* out) {
    long long x = 0, y = 0, twiceArea = 0;
    int even = 0, odd = 0;
    for (int i = 0; i < len; ++i) {
        int c = codes[i] & 7, dx = DX[c], dy = DY[c];
        twiceArea += x * (long long)dy - y * (long long)dx;  // x_i*y_{i+1} - x_{i+1}*y_i, relative to the start
        x += dx; y += dy;
        if (c & 1) ++odd; else ++even;
    }
    out[0] = even + sqrt(2.0) * odd;
    out[1] = 0.5 * fabs((double)twiceArea);
    out[2] = even; out[3] = odd; out[4] = (x == 0 && y == 0) ? 1.0 : 0.0;
}

// Number of pixels enclosed by the boundary, boundary pixels included.
// The 8-connected boundary is a wall for a 4-connected flood fill started outside the bounding box;
// whatever the flood cannot reach is inside. If `mask` (rows * cols, uint8) is not NULL it receives
// 255 for those pixels (clipped to the image).
API long long chain_fill(const int* start_xy, const unsigned char* codes, int len, int rows, int cols,
                         unsigned char* mask) {
    vector<int> pts((size_t)len * 2);
    chain_decode(start_xy, codes, len, pts.data());
    int x0 = pts[0], x1 = pts[0], y0 = pts[1], y1 = pts[1];
    for (int i = 0; i < len; ++i) {
        x0 = min(x0, pts[2 * i]); x1 = max(x1, pts[2 * i]);
        y0 = min(y0, pts[2 * i + 1]); y1 = max(y1, pts[2 * i + 1]);
    }
    int W = x1 - x0 + 3, H = y1 - y0 + 3;  // 1 pixel margin so the flood can walk around the shape
    vector<uint8_t> grid((size_t)W * H, 0);  // 0 free, 1 boundary, 2 outside
    for (int i = 0; i < len; ++i) grid[(size_t)(pts[2 * i + 1] - y0 + 1) * W + (pts[2 * i] - x0 + 1)] = 1;
    vector<int> stack = {0};
    grid[0] = 2;
    while (!stack.empty()) {
        int p = stack.back(); stack.pop_back();
        int px = p % W, py = p / W;
        for (int k = 0; k < 4; ++k) {
            int nx = px + (k == 0) - (k == 1), ny = py + (k == 2) - (k == 3);
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            size_t q = (size_t)ny * W + nx;
            if (grid[q] == 0) { grid[q] = 2; stack.push_back((int)q); }
        }
    }
    if (mask) fill(mask, mask + (size_t)rows * cols, (unsigned char)0);
    long long count = 0;
    for (int gy = 0; gy < H; ++gy)
        for (int gx = 0; gx < W; ++gx) {
            if (grid[(size_t)gy * W + gx] == 2) continue;
            ++count;
            int ix = gx + x0 - 1, iy = gy + y0 - 1;
            if (mask && ix >= 0 && iy >= 0 && ix < cols && iy < rows) mask[(size_t)iy * cols + ix] = 255;
        }
    return count;
}

// Perimeter and area of the raw float contour (straight segments between the points), for comparison.
// out[0] perimeter   out[1] area (shoelace)
API void polygon_measure(const float* xy, int n, double* out) {
    double per = 0, a2 = 0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        double xi = xy[2 * i], yi = xy[2 * i + 1], xj = xy[2 * j], yj = xy[2 * j + 1];
        per += hypot(xj - xi, yj - yi);
        a2 += xi * yj - xj * yi;
    }
    out[0] = per; out[1] = 0.5 * fabs(a2);
}

}  // extern "C"
