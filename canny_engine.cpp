#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
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

    // Reference: OpenCV's Canny (gray, Gaussian blur first, L2 gradient, 3x3 Sobel).
    // abs_low / abs_high are raw gradient magnitudes (not the 0..255 scale).
    API void canny_opencv(const unsigned char* in, int rows, int cols, int channels,
                          int ksize, float sigma, float abs_low, float abs_high,
                          unsigned char* out_edges) {
        cv::Mat src(rows, cols, channels == 1 ? CV_8UC1 : CV_8UC3, const_cast<unsigned char*>(in));
        cv::Mat gray, blurred, edges;
        if (channels == 1) gray = src; else cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        if (ksize % 2 == 0) ++ksize;
        cv::GaussianBlur(gray, blurred, cv::Size(ksize, ksize), sigma);
        cv::Canny(blurred, edges, abs_low, abs_high, 3, true);
        memcpy(out_edges, edges.data, (size_t)rows * cols);
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
}