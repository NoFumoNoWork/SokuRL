#pragma once

#include <cstddef>
#include <cstdint>

namespace SokuRLBridge
{
constexpr std::uint32_t CONTROL_MAGIC = 0x554B4F53;
constexpr std::uint32_t CONTROL_VERSION = 7;
constexpr wchar_t MAPPING_NAME_FORMAT[] = L"Local\\SokuRLBridge_%lu";
constexpr std::uint32_t MAX_DURATION_FRAMES = 10000;
constexpr std::uint32_t FRAME_RING_CAPACITY = 512;
constexpr std::uint32_t INPUT_HISTORY_CAPACITY = 4096;
constexpr std::uint32_t MAX_OBJECTS_PER_PLAYER = 64;
constexpr std::uint64_t NO_FRAME = UINT64_MAX;

enum class CommandType : std::uint32_t {
    None = 0, Input = 1, Release = 2, Run = 3, Pause = 4,
    StepFrames = 5, EstablishCheckpoint = 6, GotoFrame = 7,
    MenuConfirm = 8, StepWithInputs = 9, ApplySimpleState = 10,
};

enum class ResultCode : std::uint32_t {
    Idle = 0, Accepted = 1, Complete = 2, Released = 3,
    NotInGameplay = 4, InvalidCommand = 5, NoCheckpoint = 6,
    TargetUnavailable = 7, Restarting = 8, Diverged = 9,
    CheckpointInvalidated = 10, HistoryFull = 11,
    CheckpointRestoreUnsupported = 12,
};

enum class RunState : std::uint32_t {
    Running = 0, Paused = 1, Stepping = 2, Reconstructing = 3,
};

enum class ValidationState : std::uint32_t {
    Unknown = 0, Deterministic = 1, Diverged = 2,
};

#pragma pack(push, 4)
struct LogicalInput {
    std::int32_t horizontalAxis;
    std::int32_t verticalAxis;
    std::int32_t a;
    std::int32_t b;
    std::int32_t c;
    std::int32_t d;
    std::int32_t changeCard;
    std::int32_t spellcard;
};

struct PlayerState {
    std::uint32_t characterId;
    float x;
    float y;
    float speedX;
    float speedY;
    std::int32_t facing;
    std::int32_t hp;
    std::int32_t spirit;
    std::int32_t maxSpirit;
    std::uint32_t cardGauge;
    std::uint32_t cardCount;
    std::int32_t handIds[5];
    std::uint32_t actionId;
    std::uint32_t sequenceId;
    std::uint32_t subsequenceId;
    std::uint32_t animationFrame;
    std::uint32_t elapsedInSubsequence;
    std::uint32_t hitstop;
    std::uint32_t untech;
    std::uint32_t airborne;
    std::uint32_t frameFlags;
    std::uint32_t attackFlags;
    std::uint32_t objectCount;
    LogicalInput input;
};

struct ObjectState {
    std::uint32_t ownerIndex;
    std::uint32_t listIndex;
    std::uint32_t typeId;
    std::uint32_t actionId;
    std::uint32_t actionBlockId;
    std::uint32_t animationCounter;
    std::uint32_t animationSubFrame;
    std::uint32_t frameCount;
    float x;
    float y;
    float speedX;
    float speedY;
    float gravity;
    std::int32_t direction;
    std::int32_t hp;
    std::uint32_t hitstop;
    std::uint32_t hitBoxCount;
    std::uint32_t hurtBoxCount;
    std::uint32_t characterIndex;
    std::uint32_t isActive;
};

struct SimplePlayerState {
    float x;
    float y;
    float speedX;
    float speedY;
    std::int32_t facing;
    std::int32_t hp;
    std::int32_t spirit;
    std::int32_t maxSpirit;
    std::uint32_t cardGauge;
    std::uint32_t cardCount;
};

struct SimpleStatePatch {
    std::uint32_t timeElapsedRaw;
    std::uint32_t activeWeather;
    std::uint32_t displayedWeather;
    std::uint32_t weatherCounter;
    SimplePlayerState p1;
    SimplePlayerState p2;
};

struct ReconstructionFrame {
    LogicalInput p1Input;
    LogicalInput p2Input;
    SimpleStatePatch simple;
    std::uint64_t stateHash;
};

struct RawFrameState {
    std::uint64_t frameId;
    std::uint32_t segmentId;
    std::uint32_t sceneId;
    std::uint32_t battleMode;
    std::uint32_t battleSubMode;
    std::uint32_t stageId;
    std::uint32_t roundId;
    std::uint32_t timeElapsedRaw;
    std::uint32_t activeWeather;
    std::uint32_t displayedWeather;
    std::uint32_t weatherCounter;
    std::uint32_t randomSeed;
    PlayerState p1;
    PlayerState p2;
    std::uint32_t p1ObjectCount;
    std::uint32_t p2ObjectCount;
    std::uint32_t p1ObjectOverflow;
    std::uint32_t p2ObjectOverflow;
    ObjectState p1Objects[MAX_OBJECTS_PER_PLAYER];
    ObjectState p2Objects[MAX_OBJECTS_PER_PLAYER];
    std::uint64_t stateHash;
};

struct ControlBlock {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t structSize;
    std::uint32_t mappingSize;
    std::uint32_t commandSeq;
    std::uint32_t ackSeq;
    std::uint32_t commandType;
    std::uint32_t resultCode;
    LogicalInput commandInput;
    LogicalInput commandInputP2;
    std::uint32_t durationFrames;
    std::uint32_t inputFramesRemaining;
    std::uint64_t commandArgument;
    std::uint32_t statusSeq;
    std::uint32_t connected;
    std::uint32_t inGameplay;
    std::uint32_t runState;
    std::uint32_t checkpointValid;
    std::uint32_t validationState;
    std::uint32_t reconstructing;
    std::uint32_t stepsRemaining;
    std::uint64_t currentFrame;
    std::uint64_t recordedFrames;
    std::uint64_t lastVerifiedFrame;
    std::uint64_t firstDivergentFrame;
    std::uint32_t droppedFrames;
    std::uint32_t ringWriteSeq;
    std::uint32_t ringReadSeq;
    std::uint32_t ringCapacity;
    SimpleStatePatch commandPatch;
    RawFrameState latest;
};

struct BridgeMapping {
    ControlBlock control;
    RawFrameState frames[FRAME_RING_CAPACITY];
    ReconstructionFrame history[INPUT_HISTORY_CAPACITY];
};
#pragma pack(pop)

static_assert(sizeof(LogicalInput) == 32, "LogicalInput ABI size changed");
static_assert(sizeof(PlayerState) == 140, "PlayerState ABI size changed");
static_assert(sizeof(ObjectState) == 80, "ObjectState ABI size changed");
static_assert(sizeof(SimplePlayerState) == 40, "SimplePlayerState ABI size changed");
static_assert(sizeof(SimpleStatePatch) == 96, "SimpleStatePatch ABI size changed");
static_assert(sizeof(ReconstructionFrame) == 168, "ReconstructionFrame ABI size changed");
static_assert(sizeof(RawFrameState) == 10596, "RawFrameState ABI size changed");
static_assert(sizeof(ControlBlock) == 10884, "ControlBlock ABI size changed");
static_assert(offsetof(ControlBlock, commandSeq) == 16, "commandSeq ABI offset changed");
static_assert(offsetof(ControlBlock, commandInput) == 32, "commandInput ABI offset changed");
static_assert(offsetof(ControlBlock, currentFrame) == 144, "currentFrame ABI offset changed");
static_assert(offsetof(ControlBlock, latest) == 288, "latest ABI offset changed");
static_assert(offsetof(BridgeMapping, frames) == sizeof(ControlBlock), "ring ABI offset changed");
}
