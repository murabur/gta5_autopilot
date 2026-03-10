#include "yolo_seg.hpp"     //daha sonra öğrenilecek

#include <NvInferRuntime.h> //nvidia kütüphanesi
#include <fstream>          //dosya okuma yazma işlemlerini yönetir.
#include <stdexcept>        //hata mesajlarını fırlatmak için kullanılır
#include <algorithm>        //sıralama, arama, en küçüğü bulma gibi veriler üzerinde işlem yapan hazır algoritmalarıı sunar örnek std::min 
#include <numeric>          //sayı dizileri üzerinde toplama çarpma, ardışık sayı üretme gibi matematiksel işlemleri yapar.
#include <cstring>          //C dilinmden miras kalan düşük seviyeli bellek kopyalama ve bellek sıfırlam işlemşeri
#include <chrono>           //Hassas zaman ölçümü (milisaniye, mikrosaniye) yapmanı sağlar.

// ─── Helpers ──────────────────────────────────────────────────────────────────

static size_t volumeOf(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= (size_t)d.d[i];
    return v;
}

// ─── Constructor / Destructor ─────────────────────────────────────────────────

YoloSegTRT::YoloSegTRT(const std::string& engine_path) {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    dxgi_ = std::make_unique<DXGICapture>(0);  // 0 = birincil monitör
    loadEngine(engine_path);
    allocateBuffers();
}

YoloSegTRT::~YoloSegTRT() {
    cudaFreeHost(h_input_);
    cudaFreeHost(h_output0_);
    cudaFreeHost(h_output1_);
    cudaFree(d_input_);
    cudaFree(d_output0_);
    cudaFree(d_output1_);
    // TRT 10.x: destroy() kaldırıldı, delete kullan
    delete context_;
    delete engine_;
    delete runtime_;
    cudaStreamDestroy(stream_);
}

// ─── Load Engine ─────────────────────────────────────────────────────────────

void YoloSegTRT::loadEngine(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open engine: " + path);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);

    runtime_ = nvinfer1::createInferRuntime(logger_);
    engine_  = runtime_->deserializeCudaEngine(buf.data(), sz);
    if (!engine_) throw std::runtime_error("Failed to deserialize engine");
    context_ = engine_->createExecutionContext();

    // TRT 10.x API: getNbIOTensors() + getTensorName() + getTensorShape()
    int nb = engine_->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = engine_->getIOTensorName(i);  // TRT 10.x
        auto mode  = engine_->getTensorIOMode(name);
        auto shape = engine_->getTensorShape(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            idx_input_ = i;
            input_name_ = name;
        } else {
            // Ultralytics export: output0 = [1, max_det, 38]  (NMS engine içinde)
            //                     output1 = [1, 32, 160, 160] (proto masks)
            if (shape.nbDims == 3) {
                idx_output0_ = i;
                output0_name_ = name;
                out0_rows_ = shape.d[1];  // 300 (max_det)
                out0_cols_ = shape.d[2];  // 38  (4+NC+MASK_DIM)
            } else if (shape.nbDims == 4) {
                idx_output1_ = i;
                output1_name_ = name;
            }
        }
    }

    if (idx_input_ < 0 || idx_output0_ < 0 || idx_output1_ < 0)
        throw std::runtime_error("Binding bulunamadi. Engine tensor isimlerini kontrol edin.");

    // ── DEBUG: Tüm tensor shape'lerini dosyaya yaz ──
    FILE* dbg = fopen("C:/develop/debug_tensors.txt", "w");
    if (dbg) {
        fprintf(dbg, "[DEBUG] Engine tensor bilgileri:\n");
        for (int i = 0; i < nb; ++i) {
            const char* name = engine_->getIOTensorName(i);
            auto shape = engine_->getTensorShape(name);
            auto mode  = engine_->getTensorIOMode(name);
            fprintf(dbg, "  [%d] %s  (%s)  shape=[",
                i, name,
                mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT");
            for (int d = 0; d < shape.nbDims; ++d)
                fprintf(dbg, "%lld%s", (long long)shape.d[d], d < shape.nbDims-1 ? "," : "");
            fprintf(dbg, "]\n");
        }
        fclose(dbg);
    }
}

// ─── Allocate Buffers ─────────────────────────────────────────────────────────

void YoloSegTRT::allocateBuffers() {
    // fp32 input: 1 x 3 x 640 x 640  (engine fp16 olsa da host→device fp32 göndeririz)
    sz_input_   = (size_t)1 * 3 * INPUT_H * INPUT_W * sizeof(float);
    // output0: 1 x out0_rows_ x out0_cols_  → [1, 300, 38]
    sz_output0_ = (size_t)1 * out0_rows_ * out0_cols_ * sizeof(float);
    // output1: 1 x 32 x 160 x 160  (fp32)
    sz_output1_ = (size_t)1 * MASK_DIM * PROTO_H * PROTO_W * sizeof(float);

    // Device allocations
    CUDA_CHECK(cudaMalloc(&d_input_,   sz_input_));
    CUDA_CHECK(cudaMalloc(&d_output0_, sz_output0_));
    CUDA_CHECK(cudaMalloc(&d_output1_, sz_output1_));

    // Pinned host allocations (zero-copy staging)
    CUDA_CHECK(cudaMallocHost(&h_input_,   sz_input_));
    CUDA_CHECK(cudaMallocHost(&h_output0_, sz_output0_));
    CUDA_CHECK(cudaMallocHost(&h_output1_, sz_output1_));

    // Set bindings
    void* bindings[3];
    bindings[idx_input_]   = d_input_;
    bindings[idx_output0_] = d_output0_;
    bindings[idx_output1_] = d_output1_;
    // Store for infer
    // We'll set context bindings in infer()
}

// ─── Screen Capture (DXGI — Desktop Duplication API) ─────────────────────────
// GPU framebuffer → staging texture → cv::Mat
// GDI'ya göre 3-5x daha hızlı, CPU kullanımı çok düşük

cv::Mat YoloSegTRT::captureScreen() {
    cv::Mat frame = dxgi_->capture(CAP_X, CAP_Y, CAP_W, CAP_H);
    // Frame gelmezse önceki frame'i döndür (DXGI timeout durumu)
    if (frame.empty() && !last_frame_.empty())
        return last_frame_;
    if (!frame.empty())
        last_frame_ = frame;
    return frame;
}

// ─── Preprocess ───────────────────────────────────────────────────────────────
// Letterbox + normalize + HWC→NCHW fp16 → pinned buffer → async H2D copy

void YoloSegTRT::preprocess(const cv::Mat& img) {
    // ── Letterbox ──
    float scale = std::min((float)INPUT_W / img.cols, (float)INPUT_H / img.rows);
    int new_w   = (int)std::round(img.cols * scale);
    int new_h   = (int)std::round(img.rows * scale);
    int pad_x   = (INPUT_W - new_w) / 2;
    int pad_y   = (INPUT_H - new_h) / 2;

    cv::Mat resized;
    cv::resize(img, resized, {new_w, new_h}, 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // ── BGR→RGB, normalize, HWC→NCHW  (OpenCV ile hızlı) ──
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

    // fp32'ye çevir ve normalize et [0,255]→[0,1]
    cv::Mat fp32;
    rgb.convertTo(fp32, CV_32F, 1.0f / 255.0f);

    // 3 kanala ayır ve NCHW düzenine koy
    std::vector<cv::Mat> channels(3);
    cv::split(fp32, channels);

    float* ptr = reinterpret_cast<float*>(h_input_);
    const int plane = INPUT_H * INPUT_W;
    for (int c = 0; c < 3; ++c)
        std::memcpy(ptr + c * plane, channels[c].ptr<float>(), plane * sizeof(float));

    // Async H2D — consumer cudaStreamSynchronize çağırır
    CUDA_CHECK(cudaMemcpyAsync(d_input_, h_input_, sz_input_,
                               cudaMemcpyHostToDevice, stream_));
    // NOT: stream sync burada YOK — async pipeline için consumer sync eder
}

// ─── Infer ────────────────────────────────────────────────────────────────────

void YoloSegTRT::infer() {
    // TRT 10.x: setTensorAddress() + enqueueV3()
    context_->setTensorAddress(input_name_.c_str(),   d_input_);
    context_->setTensorAddress(output0_name_.c_str(), d_output0_);
    context_->setTensorAddress(output1_name_.c_str(), d_output1_);

    bool ok = context_->enqueueV3(stream_);
    if (!ok) throw std::runtime_error("TRT enqueueV3 failed");

    // Async D2H
    CUDA_CHECK(cudaMemcpyAsync(h_output0_, d_output0_, sz_output0_,
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_output1_, d_output1_, sz_output1_,
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ─── Post-process ─────────────────────────────────────────────────────────────
// Ultralytics NMS output0: [1, 300, 38]
// Her satır: [x1, y1, x2, y2, conf, cls_id, mask0..mask31]
// x1,y1,x2,y2 → 640x640 letterbox space piksel koordinatı

std::vector<Detection> YoloSegTRT::postprocess(int orig_w, int orig_h) {
    float* out0 = reinterpret_cast<float*>(h_output0_);

    float scale = std::min((float)INPUT_W / orig_w, (float)INPUT_H / orig_h);
    int pad_x   = (INPUT_W  - (int)std::round(orig_w * scale)) / 2;
    int pad_y   = (INPUT_H  - (int)std::round(orig_h * scale)) / 2;

    std::vector<Detection> dets;
    dets.reserve(out0_rows_);

    for (int a = 0; a < out0_rows_; ++a) {
        float* row = out0 + a * out0_cols_;

        float conf = row[4];
        if (conf < CONF_THRESH) continue;

        int cls = (int)row[5];
        if (cls < 0 || cls >= NUM_CLASSES) continue;

        // x1,y1,x2,y2 letterbox space → orijinal image coords
        float x1 = (row[0] - pad_x) / scale;
        float y1 = (row[1] - pad_y) / scale;
        float x2 = (row[2] - pad_x) / scale;
        float y2 = (row[3] - pad_y) / scale;

        Detection d;
        d.conf = conf;
        d.cls  = cls;
        d.x1 = std::max(0.f, std::min((float)orig_w, x1));
        d.y1 = std::max(0.f, std::min((float)orig_h, y1));
        d.x2 = std::max(0.f, std::min((float)orig_w, x2));
        d.y2 = std::max(0.f, std::min((float)orig_h, y2));

        if (d.x2 - d.x1 < 1.f || d.y2 - d.y1 < 1.f) continue;

        for (int m = 0; m < MASK_DIM; ++m)
            d.mask_coeffs[m] = row[6 + m];

        dets.push_back(d);
        if ((int)dets.size() >= MAX_DETS) break;
    }

    return dets;
}

// ─── NMS ─────────────────────────────────────────────────────────────────────

std::vector<int> YoloSegTRT::nms(std::vector<Detection>& dets) {
    // Sort by conf desc
    std::vector<int> idx(dets.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        return dets[a].conf > dets[b].conf;
    });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<int>  keep;

    for (size_t i = 0; i < idx.size(); ++i) {
        int a = idx[i];
        if (suppressed[a]) continue;
        keep.push_back(a);
        if ((int)keep.size() >= MAX_DETS) break;

        float ax1=dets[a].x1, ay1=dets[a].y1, ax2=dets[a].x2, ay2=dets[a].y2;
        float aArea = (ax2-ax1)*(ay2-ay1);

        for (size_t j = i+1; j < idx.size(); ++j) {
            int b = idx[j];
            if (suppressed[b]) continue;
            if (dets[b].cls != dets[a].cls) continue;  // class-wise NMS

            float bx1=dets[b].x1, by1=dets[b].y1, bx2=dets[b].x2, by2=dets[b].y2;
            float interX1 = std::max(ax1, bx1);
            float interY1 = std::max(ay1, by1);
            float interX2 = std::min(ax2, bx2);
            float interY2 = std::min(ay2, by2);
            float interW  = std::max(0.f, interX2 - interX1);
            float interH  = std::max(0.f, interY2 - interY1);
            float interArea = interW * interH;
            float bArea = (bx2-bx1)*(by2-by1);
            float iou = interArea / (aArea + bArea - interArea + 1e-6f);
            if (iou > NMS_THRESH) suppressed[b] = true;
        }
    }
    return keep;
}

// ─── Apply Masks ──────────────────────────────────────────────────────────────
// For each detection, compute soft mask = sigmoid(coeffs @ proto)
// then threshold & overlay on image

cv::Mat YoloSegTRT::applyMasks(cv::Mat& img, const std::vector<Detection>& dets) {
    float* proto = reinterpret_cast<float*>(h_output1_);

    cv::Mat result = img.clone();
    const float alpha = 0.45f;

    float scale = std::min((float)INPUT_W / img.cols, (float)INPUT_H / img.rows);
    int pad_x   = (INPUT_W - (int)std::round(img.cols * scale)) / 2;
    int pad_y   = (INPUT_H - (int)std::round(img.rows * scale)) / 2;

    // proto scale: 160/640 = 0.25
    const float proto_sw = (float)PROTO_W / INPUT_W;
    const float proto_sh = (float)PROTO_H / INPUT_H;

    for (const auto& det : dets) {
        cv::Vec3b color = CLASS_COLORS[det.cls];

        // Küçük bbox'lar için mask hesabı atla (traffic_light gibi)
        float bbox_area = (det.x2 - det.x1) * (det.y2 - det.y1);
        bool  do_mask   = (bbox_area > 2000.f);

        // BBox → letterbox space
        float bx1 = det.x1 * scale + pad_x;
        float by1 = det.y1 * scale + pad_y;
        float bx2 = det.x2 * scale + pad_x;
        float by2 = det.y2 * scale + pad_y;

        // BBox → proto space (clamp)
        int px1 = std::max(0,       (int)(bx1 * proto_sw));
        int py1 = std::max(0,       (int)(by1 * proto_sh));
        int px2 = std::min(PROTO_W, (int)std::ceil(bx2 * proto_sw));
        int py2 = std::min(PROTO_H, (int)std::ceil(by2 * proto_sh));
        if (px2 <= px1 || py2 <= py1) continue;

        int pw = px2 - px1, ph = py2 - py1;

        // ── Mask hesabı SADECE bbox bölgesinde (proto space'de) ──
        cv::Mat mask_crop(ph, pw, CV_32F, 0.f);
        float* mc = mask_crop.ptr<float>();

        for (int k = 0; k < MASK_DIM; ++k) {
            float coef = det.mask_coeffs[k];
            if (std::abs(coef) < 1e-6f) continue;
            const float* pp = proto + k * PROTO_H * PROTO_W;
            for (int y = 0; y < ph; ++y) {
                const float* src = pp + (py1 + y) * PROTO_W + px1;
                float*       dst = mc + y * pw;
                for (int x = 0; x < pw; ++x)
                    dst[x] += coef * src[x];
            }
        }

        // Sigmoid — OpenCV vectorized
        cv::Mat neg_crop;
        cv::multiply(mask_crop, -1.0f, neg_crop);
        cv::exp(neg_crop, neg_crop);
        cv::add(neg_crop, 1.0f, neg_crop);
        cv::divide(1.0f, neg_crop, mask_crop);

        // ── Resize mask_crop → orijinal bbox boyutuna ──
        int ox1 = std::max(0, (int)det.x1);
        int oy1 = std::max(0, (int)det.y1);
        int ox2 = std::min(img.cols, (int)det.x2);
        int oy2 = std::min(img.rows, (int)det.y2);
        int ow = ox2 - ox1, oh = oy2 - oy1;
        if (ow <= 0 || oh <= 0) continue;

        cv::Mat mask_final;
        cv::resize(mask_crop, mask_final, {ow, oh}, 0, 0, cv::INTER_LINEAR);

        // ── Overlay ──
        for (int y = 0; y < oh; ++y) {
            const float* mrow = mask_final.ptr<float>(y);
            cv::Vec3b*   rrow = result.ptr<cv::Vec3b>(oy1 + y);
            for (int x = 0; x < ow; ++x) {
                if (mrow[x] > 0.5f) {
                    auto& px = rrow[ox1 + x];
                    px[0] = (uchar)(px[0] * (1-alpha) + color[0] * alpha);
                    px[1] = (uchar)(px[1] * (1-alpha) + color[1] * alpha);
                    px[2] = (uchar)(px[2] * (1-alpha) + color[2] * alpha);
                }
            }
        }

        // ── BBox + label ──
        cv::rectangle(result,
            cv::Point(ox1, oy1), cv::Point(ox2, oy2),
            cv::Scalar(color[0], color[1], color[2]), 2);

        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", CLASS_NAMES[det.cls], det.conf);
        int baseline = 0;
        cv::Size tsz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int lx = ox1, ly = oy1 - 4;
        if (ly - tsz.height - baseline < 0) ly = oy1 + tsz.height + 4;
        cv::rectangle(result,
            cv::Point(lx, ly - tsz.height - baseline),
            cv::Point(lx + tsz.width, ly + baseline),
            cv::Scalar(color[0], color[1], color[2]), cv::FILLED);
        cv::putText(result, label, {lx, ly},
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    }

    return result;
}

// ─── Async API ────────────────────────────────────────────────────────────────

cv::Mat YoloSegTRT::captureAndPreprocess() {
    cv::Mat frame = captureScreen();
    if (frame.empty()) return {};
    preprocess(frame);   // pinned buffer'a yazar + async H2D başlatır
    return frame;        // orijinal frame'i consumer'a döndür
}

cv::Mat YoloSegTRT::inferAndDraw(const cv::Mat& frame) {
    // H2D zaten producer'da başlatıldı, sadece stream sync et
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    infer();
    auto dets = postprocess(frame.cols, frame.rows);
    cv::Mat f = frame.clone();
    return applyMasks(f, dets);
}

// ─── Public run() ─────────────────────────────────────────────────────────────

cv::Mat YoloSegTRT::run() {
    using clk = std::chrono::high_resolution_clock;

    auto t0 = clk::now();
    cv::Mat frame = captureScreen();
    auto t1 = clk::now();

    preprocess(frame);
    auto t2 = clk::now();

    infer();
    auto t3 = clk::now();

    auto dets = postprocess(frame.cols, frame.rows);
    auto t4 = clk::now();

    auto result = applyMasks(frame, dets);
    auto t5 = clk::now();

    auto ms = [](auto a, auto b){
        return std::chrono::duration<double,std::milli>(b-a).count();
    };

    printf("capture=%.1f  pre=%.1f  infer=%.1f  post=%.1f  mask=%.1f  dets=%zu\n",
        ms(t0,t1), ms(t1,t2), ms(t2,t3), ms(t3,t4), ms(t4,t5), dets.size());

    return result;
}

