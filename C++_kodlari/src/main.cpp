#include "yolo_seg.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <csignal>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

static std::atomic<bool> g_running{true};
void sigHandler(int) { g_running = false; }

// ── Double-buffer pipeline ───────────────────────────────────────────────────
// Producer: capture + preprocess → buffer
// Consumer: infer + mask + display
// ─────────────────────────────────────────────────────────────────────────────

struct FrameBuffer {
    cv::Mat frame;          // orijinal frame (mask overlay için)
    bool    ready = false;
};

static FrameBuffer  g_buf[2];
static int          g_write_idx = 0;   // producer yazar
static int          g_read_idx  = 1;   // consumer okur
static std::mutex   g_buf_mutex;
static std::condition_variable g_buf_cv;
static bool         g_buf_swapped = false;

// ── Display thread ───────────────────────────────────────────────────────────
static cv::Mat  g_disp_frame;
static std::mutex g_disp_mutex;
static std::atomic<bool> g_disp_ready{false};

void displayThread() {
    cv::namedWindow("YOLO-Seg TRT", cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLO-Seg TRT", CAP_W, CAP_H);
    while (g_running) {
        if (g_disp_ready.load()) {
            cv::Mat f;
            { std::lock_guard<std::mutex> lk(g_disp_mutex); f = g_disp_frame; g_disp_ready = false; }
            cv::imshow("YOLO-Seg TRT", f);
        }
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { g_running = false; break; }
        if (key == 's') {
            std::lock_guard<std::mutex> lk(g_disp_mutex);
            cv::imwrite("frame_saved.jpg", g_disp_frame);
            printf("[Main] Frame saved.\n");
        }
    }
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <engine.trt> [--save output.avi]\n", argv[0]);
        return 1;
    }
    std::signal(SIGINT,  sigHandler);
    std::signal(SIGTERM, sigHandler);

    bool save_video = false;
    std::string out_path;
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--save" && i+1 < argc) { save_video = true; out_path = argv[++i]; }

    printf("[Main] Loading engine: %s\n", argv[1]);
    YoloSegTRT detector(argv[1]);
    printf("[Main] Engine ready. Press 'q' to stop.\n");

    cv::VideoWriter writer;
    if (save_video)
        writer.open(out_path, cv::VideoWriter::fourcc('X','V','I','D'), 30, {CAP_W, CAP_H});

    // Warm-up
    try { detector.run(); } catch (...) {}

    // Display thread
    std::thread disp_thread(displayThread);

    using clk = std::chrono::high_resolution_clock;
    double fps_acc = 0; int fps_n = 0;

    // ── Producer thread: capture + preprocess ────────────────────────────────
    // Producer, detector.captureAndPreprocess() çağırır — sonucu buffer'a yazar
    // Consumer, detector.inferAndDraw() çağırır  — buffer'dan okur

    std::thread producer([&]() {
        while (g_running) {
            // Capture + preprocess
            cv::Mat frame = detector.captureAndPreprocess();
            if (frame.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }

            // Write buffer'a yaz, consumer'ı uyandır
            {
                std::lock_guard<std::mutex> lk(g_buf_mutex);
                g_buf[g_write_idx].frame = frame;
                g_buf[g_write_idx].ready = true;
                g_buf_swapped = true;
            }
            g_buf_cv.notify_one();
        }
    });

    // ── Consumer (main thread): infer + mask + display ────────────────────────
    while (g_running) {
        // Buffer hazır olana kadar bekle
        {
            std::unique_lock<std::mutex> lk(g_buf_mutex);
            g_buf_cv.wait(lk, []{ return g_buf_swapped || !g_running; });
            if (!g_running) break;
            std::swap(g_write_idx, g_read_idx);
            g_buf_swapped = false;
        }

        FrameBuffer& buf = g_buf[g_read_idx];
        if (!buf.ready) continue;
        buf.ready = false;

        auto t0 = clk::now();
        cv::Mat result = detector.inferAndDraw(buf.frame);
        auto t1 = clk::now();

        double ms  = std::chrono::duration<double,std::milli>(t1-t0).count();
        double fps = 1000.0 / ms;
        fps_acc += fps; fps_n++;

        char buf_str[32];
        snprintf(buf_str, sizeof(buf_str), "FPS: %.1f", fps);
        cv::putText(result, buf_str, {10,30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);

        { std::lock_guard<std::mutex> lk(g_disp_mutex); g_disp_frame = result; g_disp_ready = true; }

        if (save_video && writer.isOpened()) writer.write(result);

        if (fps_n % 60 == 0) {
            printf("[Main] Avg FPS: %.1f\n", fps_acc / fps_n);
            fps_acc = 0; fps_n = 0;
        }
    }

    g_running = false;
    g_buf_cv.notify_all();
    producer.join();
    disp_thread.join();
    return 0;
}
