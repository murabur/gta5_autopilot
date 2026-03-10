#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(1); \
        } \
    } while(0)
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include "dxgi_capture.hpp"

// ─── Constants ────────────────────────────────────────────────────────────────
static constexpr int   INPUT_W       = 640;
static constexpr int   INPUT_H       = 640;
static constexpr int   NUM_CLASSES   = 6;
static constexpr int   MASK_DIM      = 32;       // proto channels
static constexpr int   PROTO_H       = 160;
static constexpr int   PROTO_W       = 160;
static constexpr float CONF_THRESH   = 0.90f;
static constexpr float NMS_THRESH    = 0.45f;
static constexpr int   MAX_DETS      = 20;

// Screen capture region
static constexpr int   CAP_X         = 0;
static constexpr int   CAP_Y         = 40;
static constexpr int   CAP_W         = 1280;
static constexpr int   CAP_H         = 760;

// Class names
static const char* CLASS_NAMES[NUM_CLASSES] = {
    "road", "sidewalk", "car", "motorcycle", "person", "traffic_light"
};

// Per-class overlay colors (BGR)
static const cv::Vec3b CLASS_COLORS[NUM_CLASSES] = {
    {128, 64,  128},  // road       – purple
    {232, 35,  244},  // sidewalk   – pink
    {70,  70,  70 },  // car        – dark gray
    {0,   0,   142},  // motorcycle – dark blue
    {220, 20,  60 },  // person     – crimson
    {250, 170, 30 },  // traffic_light – orange
};

// ─── Structs ──────────────────────────────────────────────────────────────────
struct Detection {
    float x1, y1, x2, y2;   // bounding box in original image coords
    float conf;
    int   cls;
    float mask_coeffs[MASK_DIM];
};

// ─── TRT Logger ───────────────────────────────────────────────────────────────
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
};

// ─── Main class ───────────────────────────────────────────────────────────────
class YoloSegTRT {
public:
    explicit YoloSegTRT(const std::string& engine_path);
    ~YoloSegTRT();

    // Run full pipeline: capture → infer → post-process → overlay
    cv::Mat run();                          // eski senkron API (warm-up için)
    cv::Mat captureAndPreprocess();        // Producer thread çağırır
    cv::Mat inferAndDraw(const cv::Mat&);  // Consumer thread çağırır

private:
    // ── Init helpers ──
    void loadEngine(const std::string& path);
    void allocateBuffers();

    // ── Pipeline stages ──
    cv::Mat captureScreen();                              // X11 screen grab
    void    preprocess(const cv::Mat& img);              // letterbox + HWC→NCHW fp16, zero-copy pinned
    void    infer();
    std::vector<Detection> postprocess(int orig_w, int orig_h);
    cv::Mat applyMasks(cv::Mat& img, const std::vector<Detection>& dets);

    // ── NMS ──
    std::vector<int> nms(std::vector<Detection>& dets);

    // ── TRT objects ──
    Logger                                logger_;
    nvinfer1::IRuntime*                   runtime_  = nullptr;
    nvinfer1::ICudaEngine*                engine_   = nullptr;
    nvinfer1::IExecutionContext*          context_  = nullptr;

    // ── Binding indices (TRT 10.x'te artık sadece isim kullanılıyor) ──
    int idx_input_   = -1;
    int idx_output0_ = -1;
    int idx_output1_ = -1;

    // ── Tensor isimleri (TRT 10.x API) ──
    std::string input_name_;
    std::string output0_name_;
    std::string output1_name_;

    // ── Device buffers (owned) ──
    void* d_input_   = nullptr;
    void* d_output0_ = nullptr;
    void* d_output1_ = nullptr;

    // ── Pinned host buffers (zero-copy staging) ──
    void* h_input_   = nullptr;   // fp16 NCHW
    void* h_output0_ = nullptr;
    void* h_output1_ = nullptr;

    // ── Sizes (bytes) ──
    size_t sz_input_   = 0;
    size_t sz_output0_ = 0;
    size_t sz_output1_ = 0;

    // ── Output dims ──
    int out0_rows_ = 0;  // num_anchors  e.g. 8400
    int out0_cols_ = 0;  // 4+NUM_CLASSES+MASK_DIM

    cudaStream_t stream_ = nullptr;

    // ── DXGI Screen Capture ──
    std::unique_ptr<DXGICapture> dxgi_;
    cv::Mat last_frame_;  // DXGI timeout durumunda önceki frame'i kullan
};
