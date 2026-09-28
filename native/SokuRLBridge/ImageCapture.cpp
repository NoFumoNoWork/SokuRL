#include "ImageCapture.hpp"
#include <Windows.h>
#include <d3d9.h>
#include <TextureManager.hpp>
#include <cstdio>
#include <cstring>

namespace SokuRLBridge {
namespace {
HANDLE mapping = nullptr;
ImageFrame *image = nullptr;
IDirect3DSurface9 *readback = nullptr;
D3DSURFACE_DESC previous{};

HRESULT readImage()
{
    auto *device = SokuLib::pd3dDev;
    if (!device)
        return D3DERR_DEVICELOST;
    IDirect3DSurface9 *source = nullptr;
    HRESULT result = device->GetRenderTarget(0, &source);
    if (FAILED(result))
        return result;
    D3DSURFACE_DESC desc{};
    result = source->GetDesc(&desc);
    if (SUCCEEDED(result) && (desc.MultiSampleType != D3DMULTISAMPLE_NONE ||
        (desc.Format != D3DFMT_A8R8G8B8 && desc.Format != D3DFMT_X8R8G8B8) ||
        desc.Width < IMAGE_WIDTH || desc.Height < IMAGE_HEIGHT))
        result = D3DERR_INVALIDCALL;
    if (SUCCEEDED(result) && (!readback || previous.Width != desc.Width ||
        previous.Height != desc.Height || previous.Format != desc.Format)) {
        if (readback)
            readback->Release();
        readback = nullptr;
        result = device->CreateOffscreenPlainSurface(desc.Width, desc.Height, desc.Format,
            D3DPOOL_SYSTEMMEM, &readback, nullptr);
        previous = desc;
    }
    if (SUCCEEDED(result))
        result = device->GetRenderTargetData(source, readback);
    source->Release();
    if (FAILED(result))
        return result;
    D3DLOCKED_RECT locked{};
    result = readback->LockRect(&locked, nullptr, D3DLOCK_READONLY);
    if (FAILED(result))
        return result;
    image->sourceWidth = desc.Width;
    image->sourceHeight = desc.Height;
    for (unsigned y = 0; y < IMAGE_HEIGHT; ++y) {
        const auto *row = static_cast<const unsigned char *>(locked.pBits) +
            (y * desc.Height / IMAGE_HEIGHT) * locked.Pitch;
        for (unsigned x = 0; x < IMAGE_WIDTH; ++x) {
            const auto *pixel = row + (x * desc.Width / IMAGE_WIDTH) * 4;
            auto *output = &image->rgb[(y * IMAGE_WIDTH + x) * 3];
            output[0] = pixel[2];
            output[1] = pixel[1];
            output[2] = pixel[0];
        }
    }
    return readback->UnlockRect();
}
}

bool initializeImageCapture()
{
    wchar_t name[64]{};
    swprintf_s(name, L"Local\\SokuRLImage_%lu", GetCurrentProcessId());
    mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
        sizeof(ImageFrame), name);
    if (!mapping)
        return false;
    image = static_cast<ImageFrame *>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0,
        sizeof(ImageFrame)));
    if (!image) {
        closeImageCapture();
        return false;
    }
    std::memset(image, 0, sizeof(*image));
    image->magic = 0x474D4953;
    image->version = 2;
    image->result = E_PENDING;
    image->frame = UINT64_MAX;
    image->width = IMAGE_WIDTH;
    image->height = IMAGE_HEIGHT;
    return true;
}

void captureImage(std::uint64_t frame)
{
    if (!image || image->frame == frame)
        return;
    InterlockedIncrement(&image->sequence);
    image->result = readImage();
    if (SUCCEEDED(image->result))
        captureRenderState(image->renderState);
    image->frame = frame;
    InterlockedIncrement(&image->sequence);
}

void closeImageCapture()
{
    if (readback)
        readback->Release();
    readback = nullptr;
    if (image)
        UnmapViewOfFile(image);
    image = nullptr;
    if (mapping)
        CloseHandle(mapping);
    mapping = nullptr;
}
}
