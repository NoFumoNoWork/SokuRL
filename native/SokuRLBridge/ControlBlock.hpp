#pragma once

#include <cstddef>
#include <cstdint>

namespace SokuRLBridge
{
constexpr std::uint32_t CONTROL_MAGIC = 0x554B4F53; // "SOKU" in little-endian memory.
constexpr std::uint32_t CONTROL_VERSION = 1;
constexpr wchar_t MAPPING_NAME[] = L"Local\\SokuRLBridge";
constexpr std::uint32_t MAX_DURATION_FRAMES = 10000;

enum class CommandType : std::uint32_t {
    None = 0,
    Input = 1,
    Release = 2,
};

enum class ResultCode : std::uint32_t {
    Idle = 0,
    Accepted = 1,
    Complete = 2,
    Released = 3,
    NotInGameplay = 4,
    InvalidCommand = 5,
};

#pragma pack(push, 4)
struct ControlBlock {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t structSize;

    std::uint32_t commandSeq;
    std::uint32_t ackSeq;
    std::uint32_t commandType;

    std::int32_t horizontalAxis;
    std::int32_t verticalAxis;
    std::uint32_t a;
    std::uint32_t b;
    std::uint32_t c;
    std::uint32_t d;
    std::uint32_t durationFrames;

    std::uint32_t framesRemaining;
    std::uint64_t gameFrame;
    std::uint32_t connected;
    std::uint32_t inGameplay;
    std::uint32_t resultCode;
    std::uint32_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(ControlBlock) == 80, "ControlBlock ABI size changed");
static_assert(offsetof(ControlBlock, commandSeq) == 12, "commandSeq ABI offset changed");
static_assert(offsetof(ControlBlock, horizontalAxis) == 24, "horizontalAxis ABI offset changed");
static_assert(offsetof(ControlBlock, durationFrames) == 48, "durationFrames ABI offset changed");
static_assert(offsetof(ControlBlock, gameFrame) == 56, "gameFrame ABI offset changed");
static_assert(offsetof(ControlBlock, resultCode) == 72, "resultCode ABI offset changed");
}
