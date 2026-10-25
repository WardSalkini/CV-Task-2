#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
    void copyOverlay(const uint8_t* in, int channels, uint8_t* out) const {
        size_t n = (size_t)rows_ * cols_;
        for (size_t i = 0; i < n; ++i) {
            if (edges_[i]) { out[3 * i] = 0; out[3 * i + 1] = 0; out[3 * i + 2] = 255; }
            else if (channels == 1) { out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = in[i]; }
            else { out[3 * i] = in[3 * i]; out[3 * i + 1] = in[3 * i + 1]; out[3 * i + 2] = in[3 * i + 2]; }
        }
    }

    void info(float* out) const {   // low, high, edge pixel count, max gradient magnitude
        size_t count = 0;
        for (auto e : edges_) if (e) ++count;
        out[0] = lowUsed_; out[1] = highUsed_; out[2] = (float)count; out[3] = maxMag_;
    }
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
}