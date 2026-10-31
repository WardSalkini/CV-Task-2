// =====================================================================
// Person 3 - Active Contour Model (snake), greedy algorithm (Williams & Shah, 1992)
//
// Pure C++17, no OpenCV needed. Exposed through a C API so Python can call it with ctypes.
//
// Energy of a contour point v_i that is tried at candidate position p in its neighbourhood:
//
//   E(p) = alpha * E_cont(p) + beta * E_curv(p) + gamma * E_img(p) + E_balloon(p)
//
//   E_cont : (d_avg - |p - v_{i-1}|)^2        keeps the points evenly spaced (elasticity)
//   E_curv : |v_{i-1} - 2p + v_{i+1}|^2       keeps the contour smooth (rigidity)
//   E_img  : -|grad I(p)|                     pulls the contour onto strong edges
//   E_bal  : -balloon * (p - v_i) . n_out     optional inflate (+) / deflate (-) force,
//            faded to 0 where the gradient is strong so the contour stops on the object boundary
//
// Every term is normalised inside the search window (E_cont and E_curv by their window maximum,
// E_img by (g - gmin) / (gmax - gmin)), so alpha / beta / gamma are comparable weights.
// Each iteration visits every point once, moves it to the lowest-energy neighbour, then
// relaxes beta at sharp corners. The loop stops when fewer than `move_tol` points moved.
//
// The contour is CLOSED: point n-1 is connected back to point 0.
// Output coordinates are (x, y) in pixel units, x = column, y = row.
// =====================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace std;

#if defined(_WIN32)
  #define API __declspec(dllexport)
#else
  #define API __attribute__((visibility("default")))
#endif

namespace {

struct Pt { float x, y; };

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---------- image energy: Gaussian blur + Sobel magnitude (from scratch) ----------
vector<float> toGray(const unsigned char* in, int rows, int cols, int channels) {
    vector<float> g((size_t)rows * cols);
    if (channels == 1) {
        for (size_t i = 0; i < g.size(); ++i) g[i] = in[i];
    } else {  // BGR
        for (size_t i = 0; i < g.size(); ++i) {
            const unsigned char* p = in + i * 3;
            g[i] = 0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2];
        }
    }
    return g;
}

void gaussianBlur(vector<float>& img, int rows, int cols, float sigma) {
    if (sigma <= 0.0f) return;
    int half = max(1, (int)ceilf(3.0f * sigma));
    vector<float> k(2 * half + 1);
    float sum = 0;
    for (int i = -half; i <= half; ++i) { k[i + half] = expf(-(i * i) / (2 * sigma * sigma)); sum += k[i + half]; }
    for (auto& v : k) v /= sum;
    vector<float> tmp(img.size());
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            float s = 0;
            for (int i = -half; i <= half; ++i) s += k[i + half] * img[(size_t)y * cols + clampi(x + i, 0, cols - 1)];
            tmp[(size_t)y * cols + x] = s;
        }
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            float s = 0;
            for (int i = -half; i <= half; ++i) s += k[i + half] * tmp[(size_t)clampi(y + i, 0, rows - 1) * cols + x];
            img[(size_t)y * cols + x] = s;
        }
}

// returns gradient magnitude scaled to 0..255
vector<float> gradientMagnitude(const vector<float>& img, int rows, int cols) {
    vector<float> mag((size_t)rows * cols, 0.0f);
    auto at = [&](int y, int x) { return img[(size_t)clampi(y, 0, rows - 1) * cols + clampi(x, 0, cols - 1)]; };
    float mx = 0;
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            float gx = (at(y - 1, x + 1) + 2 * at(y, x + 1) + at(y + 1, x + 1))
                     - (at(y - 1, x - 1) + 2 * at(y, x - 1) + at(y + 1, x - 1));
            float gy = (at(y + 1, x - 1) + 2 * at(y + 1, x) + at(y + 1, x + 1))
                     - (at(y - 1, x - 1) + 2 * at(y - 1, x) + at(y - 1, x + 1));
            float m = sqrtf(gx * gx + gy * gy);
            mag[(size_t)y * cols + x] = m;
            mx = max(mx, m);
        }
    if (mx > 0) for (auto& v : mag) v = v * 255.0f / mx;
    return mag;
}

inline float dist(const Pt& a, const Pt& b) { return hypotf(a.x - b.x, a.y - b.y); }

float signedArea(const vector<Pt>& c) {  // shoelace, positive when the order is clockwise on screen (y down)
    double a = 0;
    size_t n = c.size();
    for (size_t i = 0; i < n; ++i) { const Pt& p = c[i]; const Pt& q = c[(i + 1) % n]; a += (double)p.x * q.y - (double)q.x * p.y; }
    return (float)(0.5 * a);
}

}  // namespace

extern "C" {

// in         : rows x cols x channels (1 = gray, 3 = BGR), uint8
// init_xy    : n * 2 floats (x0, y0, x1, y1, ...), initial closed contour
// out_xy     : n * 2 floats, final contour
// out_info   : 6 floats -> [iterations run, points moved in last iteration, final total energy,
//                           mean |gradient| on final contour (0..255), converged (1/0), reserved]
// hist_xy    : optional (may be NULL). Receives snapshots of the contour, n*2 floats each; a snapshot is
//              stored for iteration 0 and then every `snap_every` iterations.
// hist_count : number of snapshots written (<= hist_capacity)
API void snake_greedy(const unsigned char* in, int rows, int cols, int channels,
                      const float* init_xy, int n,
                      float alpha, float beta, float gamma, float balloon,
                      int window,            // half-size of the search window (1 -> 3x3, 2 -> 5x5)
                      int max_iter, int move_tol,
                      float sigma,           // Gaussian sigma applied before the gradient
                      float corner_thresh,   // curvature threshold for beta relaxation (<= 0 disables)
                      float* out_xy, float* out_info,
                      float* hist_xy, int hist_capacity, int snap_every, int* hist_count) {
    if (hist_count) *hist_count = 0;
    if (n < 3 || rows <= 0 || cols <= 0) { if (out_info) fill(out_info, out_info + 6, 0.0f); return; }
    window = max(1, window);
    snap_every = max(1, snap_every);

    vector<float> gray = toGray(in, rows, cols, channels);
    gaussianBlur(gray, rows, cols, sigma);
    vector<float> grad = gradientMagnitude(gray, rows, cols);
    auto gradAt = [&](float x, float y) {  // bilinear
        x = min(max(x, 0.0f), (float)(cols - 1)); y = min(max(y, 0.0f), (float)(rows - 1));
        int x0 = (int)floorf(x), y0 = (int)floorf(y);
        int x1 = min(x0 + 1, cols - 1), y1 = min(y0 + 1, rows - 1);
        float fx = x - x0, fy = y - y0;
        return (1 - fy) * ((1 - fx) * grad[(size_t)y0 * cols + x0] + fx * grad[(size_t)y0 * cols + x1])
             + fy * ((1 - fx) * grad[(size_t)y1 * cols + x0] + fx * grad[(size_t)y1 * cols + x1]);
    };

    vector<Pt> v(n);
    for (int i = 0; i < n; ++i) v[i] = {init_xy[2 * i], init_xy[2 * i + 1]};
    vector<float> betaI(n, beta);

    auto snapshot = [&](int it) {
        if (!hist_xy || !hist_count || *hist_count >= hist_capacity) return;
        if (it % snap_every != 0) return;
        float* dst = hist_xy + (size_t)(*hist_count) * n * 2;
        for (int i = 0; i < n; ++i) { dst[2 * i] = v[i].x; dst[2 * i + 1] = v[i].y; }
        ++(*hist_count);
    };
    snapshot(0);

    const int W = 2 * window + 1;
    vector<float> eCont(W * W), eCurv(W * W), eImg(W * W), eBal(W * W);
    int iter = 0, moved = 0;
    bool converged = false;

    for (iter = 1; iter <= max_iter; ++iter) {
        moved = 0;
        // average spacing of the current contour (continuity reference)
        float dAvg = 0;
        for (int i = 0; i < n; ++i) dAvg += dist(v[i], v[(i + 1) % n]);
        dAvg /= n;
        const bool cw = signedArea(v) > 0;  // orientation, used for the outward normal of the balloon force

        for (int i = 0; i < n; ++i) {
            const Pt prev = v[(i + n - 1) % n], next = v[(i + 1) % n], cur = v[i];
            // outward unit normal at the current point
            Pt tng = {next.x - prev.x, next.y - prev.y};
            float tl = hypotf(tng.x, tng.y);
            Pt nOut = {0, 0};
            if (tl > 1e-6f) { tng.x /= tl; tng.y /= tl; nOut = cw ? Pt{tng.y, -tng.x} : Pt{-tng.y, tng.x}; }

            float maxCont = 1e-6f, maxCurv = 1e-6f, gMin = 1e30f, gMax = -1e30f;
            int k = 0;
            for (int dy = -window; dy <= window; ++dy)
                for (int dx = -window; dx <= window; ++dx, ++k) {
                    Pt p = {cur.x + dx, cur.y + dy};
                    p.x = min(max(p.x, 0.0f), (float)(cols - 1));
                    p.y = min(max(p.y, 0.0f), (float)(rows - 1));
                    float d = dist(p, prev) - dAvg;
                    eCont[k] = d * d;
                    float cx = prev.x - 2 * p.x + next.x, cy = prev.y - 2 * p.y + next.y;
                    eCurv[k] = cx * cx + cy * cy;
                    eImg[k] = gradAt(p.x, p.y);
                    eBal[k] = -(float)(dx * nOut.x + dy * nOut.y) / (float)(window * 1.4142f);
                    maxCont = max(maxCont, eCont[k]); maxCurv = max(maxCurv, eCurv[k]);
                    gMin = min(gMin, eImg[k]); gMax = max(gMax, eImg[k]);
                }
            if (gMax - gMin < 60.0f) gMax = gMin + 60.0f;  // flat window (noise only): do not stretch it to 0..1

            int best = window * W + window;  // default: stay
            float bestE = 1e30f;
            k = 0;
            for (int dy = -window; dy <= window; ++dy)
                for (int dx = -window; dx <= window; ++dx, ++k) {
                    float e = alpha * eCont[k] / maxCont + betaI[i] * eCurv[k] / maxCurv
                            + gamma * (-(eImg[k] - gMin) / (gMax - gMin))
                            + balloon * eBal[k] * max(0.0f, 1.0f - eImg[k] / 80.0f);  // balloon fades on edges
                    if (e < bestE - 1e-9f) { bestE = e; best = k; }
                }
            int bdx = best % W - window, bdy = best / W - window;
            if (bdx != 0 || bdy != 0) {
                v[i].x = min(max(cur.x + bdx, 0.0f), (float)(cols - 1));
                v[i].y = min(max(cur.y + bdy, 0.0f), (float)(rows - 1));
                ++moved;
            }
        }

        // corner relaxation: a point whose curvature is a local maximum and above the threshold gets beta = 0
        if (corner_thresh > 0) {
            vector<float> curv(n);
            for (int i = 0; i < n; ++i) {
                const Pt &a = v[(i + n - 1) % n], &b = v[i], &c = v[(i + 1) % n];
                float u1x = b.x - a.x, u1y = b.y - a.y, u2x = c.x - b.x, u2y = c.y - b.y;
                float l1 = hypotf(u1x, u1y) + 1e-6f, l2 = hypotf(u2x, u2y) + 1e-6f;
                float dx = u1x / l1 - u2x / l2, dy = u1y / l1 - u2y / l2;
                curv[i] = hypotf(dx, dy);
            }
            for (int i = 0; i < n; ++i) {
                bool peak = curv[i] > curv[(i + n - 1) % n] && curv[i] > curv[(i + 1) % n];
                float gHere = gradAt(v[i].x, v[i].y);
                betaI[i] = (peak && curv[i] > corner_thresh && gHere > 30.0f) ? 0.0f : beta;
            }
        }

        snapshot(iter);
        if (moved <= move_tol) { converged = true; break; }
    }
    if (iter > max_iter) iter = max_iter;

    // final energy report (same terms, un-normalised) and mean edge strength
    double total = 0, meanGrad = 0;
    float dAvg = 0;
    for (int i = 0; i < n; ++i) dAvg += dist(v[i], v[(i + 1) % n]);
    dAvg /= n;
    for (int i = 0; i < n; ++i) {
        const Pt &a = v[(i + n - 1) % n], &b = v[i], &c = v[(i + 1) % n];
        float d = dist(b, a) - dAvg;
        float cx = a.x - 2 * b.x + c.x, cy = a.y - 2 * b.y + c.y;
        float g = gradAt(b.x, b.y);
        total += alpha * d * d + beta * (cx * cx + cy * cy) - gamma * g;
        meanGrad += g;
    }
    for (int i = 0; i < n; ++i) { out_xy[2 * i] = v[i].x; out_xy[2 * i + 1] = v[i].y; }
    if (out_info) {
        out_info[0] = (float)iter; out_info[1] = (float)moved; out_info[2] = (float)total;
        out_info[3] = (float)(meanGrad / n); out_info[4] = converged ? 1.0f : 0.0f; out_info[5] = 0.0f;
    }
}

// Helper: Gaussian-smoothed gradient magnitude (0..255) so the UI can display the image-energy map.
API void snake_gradient_map(const unsigned char* in, int rows, int cols, int channels, float sigma,
                            unsigned char* out_mag) {
    vector<float> g = toGray(in, rows, cols, channels);
    gaussianBlur(g, rows, cols, sigma);
    vector<float> m = gradientMagnitude(g, rows, cols);
    for (size_t i = 0; i < m.size(); ++i) out_mag[i] = (unsigned char)min(255.0f, max(0.0f, m[i] + 0.5f));
}

}  // extern "C"
