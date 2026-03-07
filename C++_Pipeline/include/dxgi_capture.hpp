#pragma once

#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

class DXGICapture {
public:
    explicit DXGICapture(int monitor_idx = 0) {
        initD3D(monitor_idx);
    }

    ~DXGICapture() {
        if (duplication_) duplication_->ReleaseFrame();
    }

    cv::Mat capture(int crop_x, int crop_y, int crop_w, int crop_h) {
        DXGI_OUTDUPL_FRAME_INFO frameInfo{};
        ComPtr<IDXGIResource>   dxgiRes;

        duplication_->ReleaseFrame();

        HRESULT hr = duplication_->AcquireNextFrame(0, &frameInfo, &dxgiRes);

        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            if (!has_frame_) return {};
        } else if (FAILED(hr)) {
            duplication_.Reset();
            staging_tex_.Reset();
            has_frame_ = false;
            try { initD3D(monitor_idx_); } catch (...) {}
            return {};
        } else {
            ComPtr<ID3D11Texture2D> tex;
            if (SUCCEEDED(dxgiRes.As(&tex))) {
                device_ctx_->CopyResource(staging_tex_.Get(), tex.Get());
                has_frame_ = true;
            }
        }

        if (!has_frame_) return {};

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = device_ctx_->Map(staging_tex_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) return {};

        int x = std::max(0, crop_x);
        int y = std::max(0, crop_y);
        int w = std::min(crop_w, screen_w_ - x);
        int h = std::min(crop_h, screen_h_ - y);

        cv::Mat full_bgra(screen_h_, screen_w_, CV_8UC4, mapped.pData, mapped.RowPitch);
        cv::Mat cropped = full_bgra(cv::Rect(x, y, w, h)).clone();
        device_ctx_->Unmap(staging_tex_.Get(), 0);

        cv::Mat bgr;
        cv::cvtColor(cropped, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }

    int screenW() const { return screen_w_; }
    int screenH() const { return screen_h_; }

private:
    void initD3D(int monitor_idx) {
        monitor_idx_ = monitor_idx;

        D3D_FEATURE_LEVEL featureLevel;
        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            0, nullptr, 0, D3D11_SDK_VERSION,
            &device_, &featureLevel, &device_ctx_);
        if (FAILED(hr)) throw std::runtime_error("D3D11CreateDevice failed");

        ComPtr<IDXGIDevice>  dxgiDevice;
        ComPtr<IDXGIAdapter> adapter;
        device_.As(&dxgiDevice);
        dxgiDevice->GetAdapter(&adapter);

        ComPtr<IDXGIOutput> output;
        hr = adapter->EnumOutputs(monitor_idx, &output);
        if (FAILED(hr)) throw std::runtime_error("Monitor bulunamadi");

        ComPtr<IDXGIOutput1> output1;
        output.As(&output1);

        hr = output1->DuplicateOutput(device_.Get(), &duplication_);
        if (FAILED(hr)) throw std::runtime_error("DuplicateOutput failed");

        DXGI_OUTPUT_DESC desc{};
        output->GetDesc(&desc);
        screen_w_ = desc.DesktopCoordinates.right  - desc.DesktopCoordinates.left;
        screen_h_ = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;

        D3D11_TEXTURE2D_DESC texDesc{};
        texDesc.Width            = screen_w_;
        texDesc.Height           = screen_h_;
        texDesc.MipLevels        = 1;
        texDesc.ArraySize        = 1;
        texDesc.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
        texDesc.SampleDesc.Count = 1;
        texDesc.Usage            = D3D11_USAGE_STAGING;
        texDesc.CPUAccessFlags   = D3D11_CPU_ACCESS_READ;

        hr = device_->CreateTexture2D(&texDesc, nullptr, &staging_tex_);
        if (FAILED(hr)) throw std::runtime_error("CreateTexture2D failed");

        printf("[DXGI] Ekran: %dx%d (monitor %d)\n", screen_w_, screen_h_, monitor_idx);
    }

    ComPtr<ID3D11Device>           device_;
    ComPtr<ID3D11DeviceContext>    device_ctx_;
    ComPtr<IDXGIOutputDuplication> duplication_;
    ComPtr<ID3D11Texture2D>        staging_tex_;
    int  screen_w_    = 0;
    int  screen_h_    = 0;
    int  monitor_idx_ = 0;
    bool has_frame_   = false;
};
