#pragma once
#include <cstdint>

namespace SokuRLBridge {
constexpr unsigned RENDER_OBJECTS_PER_PLAYER = 64;
#pragma pack(push, 4)
struct RenderEntity {
    float x;
    float y;
    float alpha;
    std::int32_t facing;
    std::uint32_t drawable;
};

struct RenderState {
    float cameraX;
    float cameraY;
    float cameraScale;
    std::uint32_t weather;
    RenderEntity players[2];
    RenderEntity objects[2][RENDER_OBJECTS_PER_PLAYER];
    std::uint32_t counts[2];
    std::uint32_t overflow;
};
#pragma pack(pop)

void captureRenderState(RenderState &state);
}
