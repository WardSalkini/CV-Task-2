// =====================================================================
// engine.cpp - backend of Assignment 2 (edge and boundary detection)
//
// Everything is written from scratch: no OpenCV, no third-party code. app.py loads the compiled library
// (engine.dll / engine.so) through ctypes and calls the functions marked "exported" below.
//
// WHERE IS MY PART?   (search for the tag in this file, and for the same tag in app.py)
//   TASK 1 - PERSON 1 : Canny edge detector            canny_run, edge_similarity
//   TASK 1 - PERSON 2 : Hough lines, circles, ellipses canny_detect_shapes
//   TASK 2 - PERSON 3 : Snake (greedy active contour)  snake_greedy, snake_gradient_map
//   TASK 2 - PERSON 4 : Chain code, perimeter, area    contour_to_chain, chain_differential, chain_decode,
//                                                      chain_measure, chain_fill, polygon_measure
//
// Build:  cmake (see CMakeLists.txt)   or   cl /O2 /EHsc /std:c++17 /LD engine.cpp /Fe:engine.dll
// =====================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

using namespace std;

// Export macro: required on MSVC, harmless on MinGW / Linux
#if defined(_WIN32)
  #define API __declspec(dllexport)
#else
  #define API __attribute__((visibility("default")))
#endif


// #####################################################################
// TASK 1 - PERSON 1 - CANNY EDGE DETECTOR
//   internals: class CannyDetector
//   exported: canny_run, edge_similarity
// #####################################################################

// =====================================================================
// Canny edge detector, implemented from scratch:
//   1. Gaussian smoothing (separable)
//   2. Sobel gradients (magnitude + direction)
//   3. Non-maximum suppression (4 quantized directions)
//   4. Double threshold (strong / weak)
//   5. Edge tracking by hysteresis
// Works on gray images (1 channel) and color images (3 channels, BGR).
//   color_mode 0: convert to gray first
//   color_mode 1: smooth and differentiate every channel, and at each pixel
//                 keep the gradient of the channel with the strongest response
// =====================================================================
class CannyDetector {
private:
    int rows_, cols_;
    vector<vector<float>> planes_;           // 1 or 3 float planes (B, G, R order)
    vector<float> gx_, gy_, mag_, magN_;     // gradient, magN_ = magnitude scaled to 0..255
    vector<float> nms_;
    vector<uint8_t> edges_;
    float maxMag_ = 0.0f, lowUsed_ = 0.0f, highUsed_ = 0.0f;

    inline int idx(int y, int x) const { return y * cols_ + x; }
    static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

    static vector<float> gaussianKernel(int ksize, float sigma) {
        if (ksize < 3) ksize = 3;
        if (ksize % 2 == 0) ++ksize;
        if (sigma <= 0.0f) sigma = 0.3f * ((ksize - 1) * 0.5f - 1.0f) + 0.8f;
        vector<float> k(ksize);
        int half = ksize / 2;
        float sum = 0.0f;
        for (int i = -half; i <= half; ++i) {
            k[i + half] = expf(-(float)(i * i) / (2.0f * sigma * sigma));
            sum += k[i + half];
        }
        for (auto& v : k) v /= sum;
        return k;
    }

    // separable Gaussian, replicated borders
    void smooth(vector<float>& p, const vector<float>& k) const {
        int half = (int)k.size() / 2;
        vector<float> tmp(p.size());
        for (int y = 0; y < rows_; ++y)
            for (int x = 0; x < cols_; ++x) {
                float s = 0.0f;
                for (int i = -half; i <= half; ++i)
                    s += p[idx(y, clampi(x + i, 0, cols_ - 1))] * k[i + half];
                tmp[idx(y, x)] = s;
            }
        for (int y = 0; y < rows_; ++y)
            for (int x = 0; x < cols_; ++x) {
                float s = 0.0f;
                for (int i = -half; i <= half; ++i)
                    s += tmp[idx(clampi(y + i, 0, rows_ - 1), x)] * k[i + half];
                p[idx(y, x)] = s;
            }
    }

    // Sobel on one plane at (y, x), replicated borders
    inline void sobelAt(const vector<float>& p, int y, int x, float& gx, float& gy) const {
        int y0 = clampi(y - 1, 0, rows_ - 1), y1 = y, y2 = clampi(y + 1, 0, rows_ - 1);
        int x0 = clampi(x - 1, 0, cols_ - 1), x1 = x, x2 = clampi(x + 1, 0, cols_ - 1);
        float a = p[idx(y0, x0)], b = p[idx(y0, x1)], c = p[idx(y0, x2)];
        float d = p[idx(y1, x0)],                      f = p[idx(y1, x2)];
        float g = p[idx(y2, x0)], h = p[idx(y2, x1)], i = p[idx(y2, x2)];
        gx = (c + 2.0f * f + i) - (a + 2.0f * d + g);
        gy = (g + 2.0f * h + i) - (a + 2.0f * b + c);
    }

    static uint8_t toU8(float v) {
        return (uint8_t)std::min(255.0f, std::max(0.0f, v + 0.5f));
    }

public:
    CannyDetector(const uint8_t* data, int rows, int cols, int channels, int colorMode)
        : rows_(rows), cols_(cols) {
        size_t n = (size_t)rows * cols;
        if (channels == 1) {
            planes_.assign(1, vector<float>(n));
            for (size_t i = 0; i < n; ++i) planes_[0][i] = data[i];
        } else if (colorMode == 0) {
            planes_.assign(1, vector<float>(n));
            for (size_t i = 0; i < n; ++i)
                planes_[0][i] = 0.114f * data[3 * i] + 0.587f * data[3 * i + 1] + 0.299f * data[3 * i + 2];
        } else {
            planes_.assign(3, vector<float>(n));
            for (size_t i = 0; i < n; ++i)
                for (int c = 0; c < 3; ++c) planes_[c][i] = data[3 * i + c];
        }
    }

    void run(int ksize, float sigma, float low, float high, bool autoThr) {
        size_t n = (size_t)rows_ * cols_;
        // 1. smoothing
        vector<float> k = gaussianKernel(ksize, sigma);
        for (auto& p : planes_) smooth(p, k);

        // 2. gradients (strongest channel per pixel)
        gx_.assign(n, 0.0f); gy_.assign(n, 0.0f); mag_.assign(n, 0.0f);
        maxMag_ = 0.0f;
        for (int y = 0; y < rows_; ++y)
            for (int x = 0; x < cols_; ++x) {
                float bestGx = 0, bestGy = 0, bestM = -1;
                for (auto& p : planes_) {
                    float gx, gy;
                    sobelAt(p, y, x, gx, gy);
                    float m = gx * gx + gy * gy;
                    if (m > bestM) { bestM = m; bestGx = gx; bestGy = gy; }
                }
                int i = idx(y, x);
                gx_[i] = bestGx; gy_[i] = bestGy;
                mag_[i] = sqrtf(bestM);
                maxMag_ = std::max(maxMag_, mag_[i]);
            }
        magN_.assign(n, 0.0f);
        if (maxMag_ > 0.0f)
            for (size_t i = 0; i < n; ++i) magN_[i] = mag_[i] / maxMag_ * 255.0f;

        // 3. non-maximum suppression
        nms_.assign(n, 0.0f);
        for (int y = 0; y < rows_; ++y)
            for (int x = 0; x < cols_; ++x) {
                int i = idx(y, x);
                float m = magN_[i];
                if (m <= 0.0f) continue;
                float ang = atan2f(gy_[i], gx_[i]) * 57.29578f;   // degrees
                if (ang < 0.0f) ang += 180.0f;
                int dx, dy;
                if (ang < 22.5f || ang >= 157.5f) { dx = 1;  dy = 0; }   // horizontal gradient
                else if (ang < 67.5f)             { dx = 1;  dy = 1; }
                else if (ang < 112.5f)            { dx = 0;  dy = 1; }   // vertical gradient
                else                              { dx = 1;  dy = -1; }
                float m1 = 0.0f, m2 = 0.0f;
                int x1 = x + dx, y1 = y + dy, x2 = x - dx, y2 = y - dy;
                if (x1 >= 0 && x1 < cols_ && y1 >= 0 && y1 < rows_) m1 = magN_[idx(y1, x1)];
                if (x2 >= 0 && x2 < cols_ && y2 >= 0 && y2 < rows_) m2 = magN_[idx(y2, x2)];
                if (m >= m1 && m >= m2) nms_[i] = m;
            }

        // 4. thresholds (manual, or automatic from the median of the surviving ridge pixels)
        if (autoThr) {
            vector<float> vals;
            vals.reserve(n / 8);
            for (size_t i = 0; i < n; ++i) if (nms_[i] > 0.0f) vals.push_back(nms_[i]);
            float med = 0.0f;
            if (!vals.empty()) {
                nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
                med = vals[vals.size() / 2];
            }
            low = std::max(1.0f, 0.66f * med);
            high = std::min(255.0f, 1.33f * med);
        }
        if (low > high) std::swap(low, high);
        lowUsed_ = low; highUsed_ = high;

        // 5. hysteresis: grow from strong pixels through weak ones (8-connectivity)
        edges_.assign(n, 0);
        vector<int> stack;
        for (size_t i = 0; i < n; ++i)
            if (nms_[i] >= high) { edges_[i] = 255; stack.push_back((int)i); }
        while (!stack.empty()) {
            int i = stack.back(); stack.pop_back();
            int y = i / cols_, x = i % cols_;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int yy = y + dy, xx = x + dx;
                    if (yy < 0 || yy >= rows_ || xx < 0 || xx >= cols_) continue;
                    int j = idx(yy, xx);
                    if (!edges_[j] && nms_[j] >= low) { edges_[j] = 255; stack.push_back(j); }
                }
        }
    }

    // ---- outputs (all single channel, rows x cols, except the overlay) ----
    void copySmoothed(uint8_t* out) const {
        size_t n = (size_t)rows_ * cols_;
        for (size_t i = 0; i < n; ++i) {
            float v = (planes_.size() == 1)
                ? planes_[0][i]
                : 0.114f * planes_[0][i] + 0.587f * planes_[1][i] + 0.299f * planes_[2][i];
            out[i] = toU8(v);
        }
    }
    void copyMagnitude(uint8_t* out) const { for (size_t i = 0; i < magN_.size(); ++i) out[i] = toU8(magN_[i]); }
    void copyNms(uint8_t* out) const { for (size_t i = 0; i < nms_.size(); ++i) out[i] = toU8(nms_[i]); }
    void copyEdges(uint8_t* out) const { memcpy(out, edges_.data(), edges_.size()); }

    // original image with the edges painted red (BGR output)
    void copyOverlay(const uint8_t* in, int channels, uint8_t* out, int thickness = 1) const {
        size_t n = (size_t)rows_ * cols_;
        for (size_t i = 0; i < n; ++i) {
            if (channels == 1) { out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = in[i]; }
            else { out[3 * i] = in[3 * i]; out[3 * i + 1] = in[3 * i + 1]; out[3 * i + 2] = in[3 * i + 2]; }
        }
        int half = max(0, max(1, thickness) / 2);
        for (int y = 0; y < rows_; ++y)
            for (int x = 0; x < cols_; ++x)
                if (edges_[(size_t)y * cols_ + x])
                    for (int yy = max(0, y - half); yy <= min(rows_ - 1, y + half); ++yy)
                        for (int xx = max(0, x - half); xx <= min(cols_ - 1, x + half); ++xx) {
                            size_t p = ((size_t)yy * cols_ + xx) * 3;
                            out[p] = 0; out[p + 1] = 0; out[p + 2] = 255;
                        }
    }

    void info(float* out) const {   // low, high, edge pixel count, max gradient magnitude
        size_t count = 0;
        for (auto e : edges_) if (e) ++count;
        out[0] = lowUsed_; out[1] = highUsed_; out[2] = (float)count; out[3] = maxMag_;
    }

    const vector<uint8_t>& edges() const { return edges_; }
    const vector<float>& gx() const { return gx_; }
    const vector<float>& gy() const { return gy_; }
};


extern "C" {
    // in: rows x cols x channels (1 = gray, 3 = BGR). low/high are on the 0..255 scale of the
    // normalized gradient magnitude. out_overlay is rows x cols x 3. out_info has 4 floats.
    API void canny_run(const unsigned char* in, int rows, int cols, int channels, int color_mode,
                       int ksize, float sigma, float low, float high, int auto_thr,
                       unsigned char* out_smooth, unsigned char* out_mag, unsigned char* out_nms,
                       unsigned char* out_edges, unsigned char* out_overlay, float* out_info) {
        CannyDetector det(in, rows, cols, channels, color_mode);
        det.run(ksize, sigma, low, high, auto_thr != 0);
        det.copySmoothed(out_smooth);
        det.copyMagnitude(out_mag);
        det.copyNms(out_nms);
        det.copyEdges(out_edges);
        det.copyOverlay(in, channels, out_overlay);
        det.info(out_info);
    }

    // Intersection over union of two binary edge maps (1 = identical)
    API float edge_similarity(const unsigned char* a, const unsigned char* b, int n) {
        long inter = 0, uni = 0;
        for (int i = 0; i < n; ++i) {
            bool x = a[i] != 0, y = b[i] != 0;
            inter += (x && y);
            uni += (x || y);
        }
        return uni ? (float)inter / (float)uni : 1.0f;
    }

}  // extern "C"  (Person 1)

// #####################################################################
// TASK 1 - PERSON 2 - HOUGH TRANSFORM (lines, circles, ellipses)
//   internals: detectLines, detectCircles, detectEllipses and the drawing helpers
//   exported: canny_detect_shapes (runs Canny, then the selected detectors)
// #####################################################################

struct LineResult { float rho, theta, votes, x1, y1, x2, y2; };
struct CircleResult { float x, y, radius, votes; };
struct EllipseResult { float x, y, a, b, angle, votes; };

static void putPixel(vector<uint8_t>& out, int w, int h, int x, int y,
                     uint8_t b, uint8_t g, uint8_t r, int thickness) {
    int half = max(0, thickness / 2);
    for (int yy = y - half; yy <= y + half; ++yy)
        for (int xx = x - half; xx <= x + half; ++xx)
            if (xx >= 0 && xx < w && yy >= 0 && yy < h) {
                size_t p = (size_t)(yy * w + xx) * 3;
                out[p] = b; out[p + 1] = g; out[p + 2] = r;
            }
}

static void drawSegment(vector<uint8_t>& out, int w, int h, int x1, int y1,
                        int x2, int y2, uint8_t b, uint8_t g, uint8_t r, int t) {
    int dx = abs(x2 - x1), sx = x1 < x2 ? 1 : -1;
    int dy = -abs(y2 - y1), sy = y1 < y2 ? 1 : -1, err = dx + dy;
    while (true) {
        putPixel(out, w, h, x1, y1, b, g, r, t);
        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x1 += sx; }
        if (e2 <= dx) { err += dx; y1 += sy; }
    }
}

static vector<LineResult> detectLines(const vector<uint8_t>& edges, int w, int h,
                                      float rhoStep, float thetaStep,
                                      int threshold, int nms, int maxLines,
                                      vector<uint8_t>& accOut, int& accW, int& accH) {
    const float pi = 3.14159265359f;
    int thetaBins = max(1, (int)ceil(180.0f / thetaStep));
    int rhoMax = (int)ceil(sqrt((double)w * w + (double)h * h));
    int rhoBins = (int)floor((2.0f * rhoMax) / rhoStep) + 1;
    vector<int> acc((size_t)rhoBins * thetaBins, 0);
    vector<float> sins(thetaBins), coss(thetaBins);
    for (int t = 0; t < thetaBins; ++t) {
        float a = (-90.0f + t * thetaStep) * pi / 180.0f;
        sins[t] = sinf(a); coss[t] = cosf(a);
    }
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
        if (edges[(size_t)y * w + x])
            for (int t = 0; t < thetaBins; ++t) {
                float rho = y * coss[t] - x * sins[t];
                int ri = (int)lround((rho + rhoMax) / rhoStep);
                if (ri >= 0 && ri < rhoBins) ++acc[(size_t)ri * thetaBins + t];
            }
    int maxVote = 1;
    for (int v : acc) maxVote = max(maxVote, v);
    accW = thetaBins; accH = rhoBins; accOut.assign((size_t)accW * accH, 0);
    double logMax = log1p((double)maxVote);
    for (size_t i = 0; i < acc.size(); ++i)
        accOut[i] = (uint8_t)lround(255.0 * log1p((double)acc[i]) / logMax);
    vector<LineResult> candidates;
    int radius = max(1, nms / 2);
    for (int ri = 0; ri < rhoBins; ++ri)
        for (int ti = 0; ti < thetaBins; ++ti) {
            int votes = acc[(size_t)ri * thetaBins + ti];
            if (votes < threshold) continue;
            bool peak = true;
            for (int dr = -radius; dr <= radius && peak; ++dr)
                for (int dt = -radius; dt <= radius; ++dt) {
                    if (!dr && !dt) continue;
                    int rj = ri + dr, tj = (ti + dt + thetaBins) % thetaBins;
                    if (rj >= 0 && rj < rhoBins &&
                        acc[(size_t)rj * thetaBins + tj] > votes) { peak = false; break; }
                }
            if (!peak) continue;
            float rho = ri * rhoStep - rhoMax;
            float theta = -90.0f + ti * thetaStep;
            float a = theta * pi / 180.0f, sn = sinf(a), cs = cosf(a);
            vector<pair<float, float>> points;
            if (fabs(sn) > 1e-5f) {
                for (int y : {0, h - 1})
                    points.push_back({(y * cs - rho) / sn, (float)y});
            }
            if (fabs(cs) > 1e-5f) {
                for (int x : {0, w - 1})
                    points.push_back({(float)x, (rho + x * sn) / cs});
            }
            vector<pair<float, float>> valid;
            for (auto p : points) if (p.first >= 0 && p.first < w && p.second >= 0 && p.second < h) {
                bool duplicate = false;
                for (auto q : valid) if (hypot(p.first - q.first, p.second - q.second) <= 1.0f) {
                    duplicate = true;
                    break;
                }
                if (!duplicate) valid.push_back(p);
            }
            if (valid.size() >= 2)
                candidates.push_back({rho, theta, (float)votes, valid[0].first, valid[0].second, valid[1].first, valid[1].second});
        }
    sort(candidates.begin(), candidates.end(),
         [](const LineResult& a, const LineResult& b) { return a.votes > b.votes; });
    if ((int)candidates.size() > maxLines) candidates.resize(maxLines);
    return candidates;
}

static vector<CircleResult> detectCircles(const vector<uint8_t>& edges,
                                          const vector<float>& gx, const vector<float>& gy,
                                          int w, int h, int rmin, int rmax, int rstep,
                                          int threshold, int minDist, bool gradientMode,
                                          vector<uint8_t>& projection) {
    rmin = max(1, rmin); rmax = max(rmin, rmax); rstep = max(1, rstep);
    int numR = (rmax - rmin) / rstep + 1;
    vector<int> acc((size_t)numR * w * h, 0);
    const float pi = 3.14159265359f;
    vector<pair<float, float>> dirs;
    for (int k = 0; k < 72; ++k) { float a = 2 * pi * k / 72; dirs.push_back({cosf(a), sinf(a)}); }
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        size_t edgeIndex = (size_t)y * w + x;
        if (!edges[edgeIndex]) continue;
        vector<pair<float, float>> voteDirs;
        if (gradientMode) {
            float angle = atan2f(gy[edgeIndex], gx[edgeIndex]);
            for (int offset = -3; offset <= 3; ++offset) {
                float delta = offset * 5.0f * pi / 180.0f;
                voteDirs.push_back({cosf(angle + delta), sinf(angle + delta)});
                voteDirs.push_back({cosf(angle + pi + delta), sinf(angle + pi + delta)});
            }
        } else {
            voteDirs = dirs;
        }
        for (int ri = 0, r = rmin; r <= rmax; ++ri, r += rstep)
            for (auto d : voteDirs) {
                int cx = (int)lround(x - r * d.first), cy = (int)lround(y - r * d.second);
                if (cx >= 0 && cx < w && cy >= 0 && cy < h)
                    ++acc[((size_t)ri * h + cy) * w + cx];
            }
    }
    int maxVote = 1; for (int v : acc) maxVote = max(maxVote, v);
    projection.assign((size_t)w * h, 0);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        int best = 0;
        for (int ri = 0; ri < numR; ++ri) best = max(best, acc[((size_t)ri * h + y) * w + x]);
        projection[(size_t)y * w + x] = (uint8_t)lround(255.0 * log1p((double)best) / log1p((double)maxVote));
    }
    struct CirclePeak { int x, y, ri, votes; };
    vector<CirclePeak> candidates;
    for (int ri = 0; ri < numR; ++ri) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        int votes = acc[((size_t)ri * h + y) * w + x];
        if (votes < threshold) continue;
        bool peak = true;
        for (int dr = -1; dr <= 1 && peak; ++dr)
            for (int dy = -1; dy <= 1 && peak; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dr && !dy && !dx) continue;
                    int nr = ri + dr, nx = x + dx, ny = y + dy;
                    if (nr >= 0 && nr < numR && nx >= 0 && nx < w && ny >= 0 && ny < h &&
                        acc[((size_t)nr * h + ny) * w + nx] > votes) { peak = false; break; }
                }
        if (peak) candidates.push_back({x, y, ri, votes});
    }
    sort(candidates.begin(), candidates.end(),
         [](const CirclePeak& a, const CirclePeak& b) { return a.votes > b.votes; });
    vector<CircleResult> result;
    for (const auto& candidate : candidates) {
        float radius = (float)(rmin + candidate.ri * rstep);
        bool keep = true;
        for (const auto& circle : result)
            if (hypot(circle.x - candidate.x, circle.y - candidate.y) < minDist &&
                fabs(circle.radius - radius) < 2.0f * rstep) { keep = false; break; }
        if (keep) {
            result.push_back({(float)candidate.x, (float)candidate.y, radius, (float)candidate.votes});
            if (result.size() >= 32) break;
        }
    }
    return result;
}

static vector<EllipseResult> detectEllipses(const vector<uint8_t>& edges, int w, int h,
                                            int minMajor, int maxMajor, int minMinor,
                                            int threshold, int maxPoints) {
    vector<pair<int, int>> pts;
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
        if (edges[(size_t)y * w + x]) pts.push_back({x, y});
    if ((int)pts.size() > maxPoints) {
        vector<pair<int, int>> sampled;
        int stride = (int)ceil((double)pts.size() / maxPoints);
        for (int i = 0; i < (int)pts.size(); i += stride) sampled.push_back(pts[i]);
        pts.swap(sampled);
    }
    struct Candidate { EllipseResult ellipse; vector<int> inliers; };
    vector<Candidate> candidates;
    for (int i = 0; i < (int)pts.size(); ++i)
        for (int j = i + 1; j < (int)pts.size(); ++j) {
            float dx = (float)pts[j].first - pts[i].first, dy = (float)pts[j].second - pts[i].second;
            float a = 0.5f * hypot(dx, dy);
            if (2 * a < minMajor || 2 * a > maxMajor || a < minMinor) continue;
            float cx = (pts[i].first + pts[j].first) * 0.5f, cy = (pts[i].second + pts[j].second) * 0.5f;
            float ang = atan2f(dy, dx);
            vector<int> accB((int)ceil(a) + 1, 0);
            for (auto q : pts) {
                float ux = q.first - cx, uy = q.second - cy;
                float xr = ux * cosf(ang) + uy * sinf(ang), yr = -ux * sinf(ang) + uy * cosf(ang);
                if (fabs(xr) >= a || fabs(yr) <= 1.0f) continue;
                float denominator = a * a - xr * xr;
                if (denominator <= 1e-5f) continue;
                float b2 = (a * a * yr * yr) / denominator;
                float b = sqrtf(max(0.0f, b2));
                if (b >= minMinor && b <= a) {
                    int bin = (int)lround(b);
                    if (bin >= 0 && bin < (int)accB.size()) {
                        ++accB[bin];
                        if (bin > 0) ++accB[bin - 1];
                        if (bin + 1 < (int)accB.size()) ++accB[bin + 1];
                    }
                }
            }
            int bestBin = 0, maxVotes = 0;
            for (int b = max(1, minMinor); b < (int)accB.size(); ++b)
                if (accB[b] > maxVotes) { maxVotes = accB[b]; bestBin = b; }
            if (maxVotes < threshold) continue;
            EllipseResult ellipse{cx, cy, a, (float)bestBin,
                                  ang * 180.0f / 3.14159265359f, (float)maxVotes};
            vector<int> inliers;
            for (int k = 0; k < (int)pts.size(); ++k) {
                float ux = pts[k].first - cx, uy = pts[k].second - cy;
                float xr = ux * cosf(ang) + uy * sinf(ang), yr = -ux * sinf(ang) + uy * cosf(ang);
                float error = fabs((xr * xr) / (a * a) + (yr * yr) / (bestBin * bestBin) - 1.0f);
                if (error < 0.12f) inliers.push_back(k);
            }
            candidates.push_back({ellipse, inliers});
        }
    sort(candidates.begin(), candidates.end(),
         [](const Candidate& a, const Candidate& b) { return a.ellipse.votes > b.ellipse.votes; });
    vector<EllipseResult> result;
    vector<bool> used(pts.size(), false);
    for (const auto& candidate : candidates) {
        bool duplicate = false;
        for (const auto& existing : result)
            if (hypot(existing.x - candidate.ellipse.x, existing.y - candidate.ellipse.y) < 20.0f &&
                fabs(existing.a - candidate.ellipse.a) < 15.0f &&
                fabs(existing.b - candidate.ellipse.b) < 15.0f) { duplicate = true; break; }
        if (duplicate) continue;
        bool hasUnusedInlier = false;
        for (int index : candidate.inliers)
            if (!used[index]) { hasUnusedInlier = true; break; }
        if (!hasUnusedInlier) continue;
        result.push_back(candidate.ellipse);
        for (int index : candidate.inliers) used[index] = true;
        if (result.size() >= 8) break;
    }
    return result;
}

static void drawEllipse(vector<uint8_t>& out, int w, int h, const EllipseResult& e,
                        uint8_t b, uint8_t g, uint8_t r, int t) {
    const float pi = 3.14159265359f, a = e.angle * pi / 180.0f;
    int px = -1, py = -1;
    for (int k = 0; k <= 360; ++k) {
        float q = k * pi / 180.0f, x = e.x + e.a * cosf(q) * cosf(a) - e.b * sinf(q) * sinf(a);
        float y = e.y + e.a * cosf(q) * sinf(a) + e.b * sinf(q) * cosf(a);
        int ix = (int)lround(x), iy = (int)lround(y);
        if (px >= 0) drawSegment(out, w, h, px, py, ix, iy, b, g, r, t);
        px = ix; py = iy;
    }
}


extern "C" {
    // flags: bit 0 lines, bit 1 circles, bit 2 ellipses.
    // Arrays are caller-owned. line_data has max_lines * 7 floats,
    // circle_data has max_circles * 4 floats, ellipse_data has max_ellipses * 6 floats.
    API void canny_detect_shapes(
        const unsigned char* in, int rows, int cols, int channels,
        int ksize, float sigma, float low, float high, int auto_thr, int flags,
        float rho_step, float theta_step, int line_threshold, int line_nms, int max_lines,
        int rmin, int rmax, int rstep, int circle_threshold, int min_dist, int max_circles,
        int gradient_mode,
        int min_major, int max_major, int min_minor, int ellipse_threshold, int max_points,
        int max_ellipses, int thickness, unsigned char* edges_out, unsigned char* overlay_out,
        unsigned char* line_accum, int line_accum_capacity, int* line_accum_w, int* line_accum_h,
        unsigned char* circle_accum, float* line_data, int* line_count,
        float* circle_data, int* circle_count, float* ellipse_data, int* ellipse_count,
        float* info_out) {
        CannyDetector det(in, rows, cols, channels, 0);
        det.run(ksize, sigma, low, high, auto_thr != 0);
        det.copyEdges(edges_out);
        det.copyOverlay(in, channels, overlay_out, thickness);
        det.info(info_out);
        *line_count = *circle_count = *ellipse_count = 0;
        *line_accum_w = *line_accum_h = 0;
        if (flags & 1) {
            vector<uint8_t> acc; int aw = 0, ah = 0;
            auto lines = detectLines(det.edges(), cols, rows, rho_step, theta_step,
                                     line_threshold, line_nms, max_lines, acc, aw, ah);
            int copyN = min((int)acc.size(), line_accum_capacity);
            memcpy(line_accum, acc.data(), copyN);
            *line_accum_w = aw; *line_accum_h = min(ah, line_accum_capacity / max(1, aw));
            *line_count = min((int)lines.size(), max_lines);
            for (int i = 0; i < *line_count; ++i) {
                const auto& v = lines[i];
                line_data[i * 7] = v.rho; line_data[i * 7 + 1] = v.theta;
                line_data[i * 7 + 2] = v.votes; line_data[i * 7 + 3] = v.x1;
                line_data[i * 7 + 4] = v.y1; line_data[i * 7 + 5] = v.x2;
                line_data[i * 7 + 6] = v.y2;
            }
            // Draw directly after copying to avoid exposing temporary image ownership.
            vector<uint8_t> rendered(overlay_out, overlay_out + (size_t)rows * cols * 3);
            for (int i = 0; i < *line_count; ++i)
                drawSegment(rendered, cols, rows, (int)lines[i].x1, (int)lines[i].y1,
                            (int)lines[i].x2, (int)lines[i].y2, 0, 255, 0, thickness);
            memcpy(overlay_out, rendered.data(), rendered.size());
        }
        if (flags & 2) {
            vector<uint8_t> projection;
            auto circles = detectCircles(det.edges(), det.gx(), det.gy(), cols, rows,
                                         rmin, rmax, rstep, circle_threshold, min_dist,
                                         gradient_mode != 0, projection);
            memcpy(circle_accum, projection.data(), projection.size());
            *circle_count = min((int)circles.size(), max_circles);
            vector<uint8_t> rendered(overlay_out, overlay_out + (size_t)rows * cols * 3);
            for (int i = 0; i < *circle_count; ++i) {
                const auto& c = circles[i];
                circle_data[i * 4] = c.x; circle_data[i * 4 + 1] = c.y;
                circle_data[i * 4 + 2] = c.radius; circle_data[i * 4 + 3] = c.votes;
                int px = (int)lround(c.x), py = (int)lround(c.y), pr = (int)lround(c.radius);
                int lastX = -1, lastY = -1;
                for (int k = 0; k <= 360; ++k) {
                    float a = 3.14159265359f * k / 180.0f;
                    int x = (int)lround(c.x + pr * cosf(a)), y = (int)lround(c.y + pr * sinf(a));
                    if (lastX >= 0) drawSegment(rendered, cols, rows, lastX, lastY, x, y, 0, 220, 255, thickness);
                    lastX = x; lastY = y;
                }
                int cross = 3 * max(1, thickness);
                int crossThickness = max(2, thickness / 2);
                drawSegment(rendered, cols, rows, px - cross, py, px + cross, py, 0, 220, 255, crossThickness);
                drawSegment(rendered, cols, rows, px, py - cross, px, py + cross, 0, 220, 255, crossThickness);
            }
            memcpy(overlay_out, rendered.data(), rendered.size());
        }
        if (flags & 4) {
            auto ellipses = detectEllipses(det.edges(), cols, rows, min_major, max_major,
                                           min_minor, ellipse_threshold, max_points);
            *ellipse_count = min((int)ellipses.size(), max_ellipses);
            vector<uint8_t> rendered(overlay_out, overlay_out + (size_t)rows * cols * 3);
            for (int i = 0; i < *ellipse_count; ++i) {
                const auto& e = ellipses[i];
                ellipse_data[i * 6] = e.x; ellipse_data[i * 6 + 1] = e.y;
                ellipse_data[i * 6 + 2] = e.a; ellipse_data[i * 6 + 3] = e.b;
                ellipse_data[i * 6 + 4] = e.angle; ellipse_data[i * 6 + 5] = e.votes;
                drawEllipse(rendered, cols, rows, e, 255, 0, 220, thickness);
                int cross = 3 * max(1, thickness);
                drawSegment(rendered, cols, rows, (int)lround(e.x) - cross, (int)lround(e.y),
                            (int)lround(e.x) + cross, (int)lround(e.y), 255, 0, 220, max(2, thickness / 2));
                drawSegment(rendered, cols, rows, (int)lround(e.x), (int)lround(e.y) - cross,
                            (int)lround(e.x), (int)lround(e.y) + cross, 255, 0, 220, max(2, thickness / 2));
            }
            memcpy(overlay_out, rendered.data(), rendered.size());
        }
    }
}  // extern "C"  (Person 2)

// #####################################################################
// TASK 2 - PERSON 3 - ACTIVE CONTOUR (SNAKE), GREEDY ALGORITHM
//   exported: snake_greedy, snake_gradient_map
// #####################################################################

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

namespace snake_impl {

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

}  // namespace snake_impl

using namespace snake_impl;

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


// #####################################################################
// TASK 2 - PERSON 4 - CHAIN CODE, PERIMETER, AREA
//   exported: contour_to_chain, chain_differential, chain_decode, chain_measure, chain_fill, polygon_measure
// #####################################################################

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

namespace chain_impl {

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

}  // namespace chain_impl

using namespace chain_impl;

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

