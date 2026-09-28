#pragma once
#include <cstdint>
#include "RenderState.hpp"

namespace SokuRLBridge {
// A separate versioned mapping keeps the existing diagnostic ABI unchanged.
constexpr unsigned IMAGE_WIDTH = 320;
constexpr unsigned IMAGE_HEIGHT = 240;
#pragma pack(push, 4)
struct ImageFrame {
    std::uint32_t magic;
    std::uint32_t version;
    volatile long sequence;
    std::int32_t result;
    std::uint64_t frame;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t sourceWidth;
    std::uint32_t sourceHeight;
    RenderState renderState;
    unsigned char rgb[IMAGE_WIDTH * IMAGE_HEIGHT * 3];
};
#pragma pack(pop)

bool initializeImageCapture(bool pixels);
void captureImage(std::uint64_t frame);
void closeImageCapture();
}
