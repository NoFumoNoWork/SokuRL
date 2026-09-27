#include "ControlBlock.hpp"

#include <BattleManager.hpp>
#include <BattleMode.hpp>
#include <Character.hpp>
#include <Hash.hpp>
#include <InputManager.hpp>
#include <PracticeSettings.hpp>
#include <Scenes.hpp>
#include <SokuAddresses.hpp>
#include <Tamper.hpp>
#include <UnionCast.hpp>
#include <VTables.hpp>
#include <Weather.hpp>

#include <Windows.h>
#include <algorithm>
#include <cstring>
#include <vector>

namespace
{
using SetInputsMethod = void (SokuLib::KeymapManager::*)();
using BattleProcessMethod = int (SokuLib::Battle::*)();
using BattleManagerProcessMethod = int (SokuLib::BattleManager::*)();

constexpr DWORD KEYMAP_SET_INPUTS_HOOK = 0x0040A45D;
constexpr std::uint32_t LOCAL_BATTLE_SCENE = 5;
constexpr std::uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

HANDLE g_fileMapping = nullptr;
SokuRLBridge::BridgeMapping *g_mapping = nullptr;
SokuRLBridge::ControlBlock *g_control = nullptr;
SetInputsMethod g_originalSetInputs = nullptr;
BattleProcessMethod g_originalBattleProcess = nullptr;
BattleManagerProcessMethod g_originalBattleManagerProcess = nullptr;

std::uint32_t g_lastCommandSeq = 0;
std::uint32_t g_segmentId = 0;
std::uint64_t g_currentFrame = 0;
std::uint32_t g_stepsRemaining = 0;
bool g_paused = false;
bool g_inSimulationUpdate = false;
bool g_reconstructing = false;
bool g_restartRequested = false;
bool g_awaitingRestart = false;
bool g_establishAfterRestart = false;
std::uint64_t g_reconstructionTarget = 0;
std::uint64_t g_replayInputFrame = 0;

SokuRLBridge::LogicalInput g_effectiveInputs[2]{};
SokuLib::KeyInput g_activeInput{};
std::uint32_t g_activeInputFrames = 0;
bool g_activeInputEnabled = false;
bool g_neutralPending = false;

struct CheckpointIdentity {
    std::uint32_t leftCharacter;
    std::uint32_t rightCharacter;
    std::uint32_t stage;
    std::uint32_t randomSeed;
    std::uint32_t practiceWeather;
    std::uint32_t dummyState;
    std::uint32_t position;
    std::uint32_t guard;
    std::uint32_t counter;
    std::uint32_t airtech;
};

CheckpointIdentity g_checkpoint{};
std::vector<SokuRLBridge::RawFrameState> g_history;

std::uint32_t load32(const volatile std::uint32_t *value)
{
    return static_cast<std::uint32_t>(InterlockedCompareExchange(
        reinterpret_cast<volatile LONG *>(const_cast<volatile std::uint32_t *>(value)), 0, 0));
}

void store32(volatile std::uint32_t *target, std::uint32_t value)
{
    InterlockedExchange(reinterpret_cast<volatile LONG *>(target), static_cast<LONG>(value));
}

void beginStatusWrite()
{
    InterlockedIncrement(reinterpret_cast<volatile LONG *>(&g_control->statusSeq));
    MemoryBarrier();
}

void endStatusWrite()
{
    MemoryBarrier();
    InterlockedIncrement(reinterpret_cast<volatile LONG *>(&g_control->statusSeq));
}

bool isPracticeGameplay()
{
    return *reinterpret_cast<const int *>(SokuLib::ADDR_SCENE_ID) == LOCAL_BATTLE_SCENE &&
        SokuLib::mainMode == SokuLib::BATTLE_MODE_PRACTICE;
}

SokuRLBridge::LogicalInput toLogicalInput(const SokuLib::KeyInput &input)
{
    return {input.horizontalAxis, input.verticalAxis, input.a, input.b, input.c, input.d,
        input.changeCard, input.spellcard};
}

SokuLib::KeyInput toKeyInput(const SokuRLBridge::LogicalInput &input)
{
    return {input.horizontalAxis, input.verticalAxis, input.a, input.b, input.c, input.d,
        input.changeCard, input.spellcard};
}

bool isBoolean(std::int32_t value)
{
    return value == 0 || value == 1;
}

bool isValidInput(const SokuRLBridge::LogicalInput &input, std::uint32_t duration)
{
    return input.horizontalAxis >= -1 && input.horizontalAxis <= 1 &&
        input.verticalAxis >= -1 && input.verticalAxis <= 1 &&
        isBoolean(input.a) && isBoolean(input.b) && isBoolean(input.c) && isBoolean(input.d) &&
        isBoolean(input.changeCard) && isBoolean(input.spellcard) &&
        duration >= 1 && duration <= SokuRLBridge::MAX_DURATION_FRAMES;
}

void publishResult(SokuRLBridge::ResultCode result)
{
    store32(&g_control->resultCode, static_cast<std::uint32_t>(result));
}

void acknowledge(std::uint32_t sequence)
{
    MemoryBarrier();
    store32(&g_control->ackSeq, sequence);
}

CheckpointIdentity readIdentity()
{
    CheckpointIdentity identity{};
    identity.leftCharacter = static_cast<std::uint32_t>(SokuLib::gameParams.leftPlayerInfo.character);
    identity.rightCharacter = static_cast<std::uint32_t>(SokuLib::gameParams.rightPlayerInfo.character);
    identity.stage = SokuLib::gameParams.stageId;
    identity.randomSeed = SokuLib::gameParams.randomSeed;
    if (SokuLib::practiceSettings) {
        identity.practiceWeather = static_cast<std::uint32_t>(SokuLib::practiceSettings->weather);
        identity.dummyState = static_cast<std::uint32_t>(SokuLib::practiceSettings->state);
        identity.position = SokuLib::practiceSettings->position;
        identity.guard = static_cast<std::uint32_t>(SokuLib::practiceSettings->guard);
        identity.counter = static_cast<std::uint32_t>(SokuLib::practiceSettings->counter);
        identity.airtech = static_cast<std::uint32_t>(SokuLib::practiceSettings->airtech);
    }
    return identity;
}

bool identityMatchesCheckpoint()
{
    const auto current = readIdentity();
    return std::memcmp(&current, &g_checkpoint, sizeof(current)) == 0;
}

void setCheckpointValid(bool valid)
{
    store32(&g_control->checkpointValid, valid ? 1U : 0U);
}

void invalidateCheckpoint(SokuRLBridge::ResultCode reason)
{
    if (!load32(&g_control->checkpointValid))
        return;
    setCheckpointValid(false);
    g_history.clear();
    publishResult(reason);
}

std::uint64_t hashBytes(std::uint64_t hash, const void *data, std::size_t size)
{
    const auto *bytes = static_cast<const unsigned char *>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

std::uint64_t stateHash(const SokuRLBridge::RawFrameState &state)
{
    std::uint64_t hash = FNV_OFFSET;
    hash = hashBytes(hash, &state.frameId, sizeof(state.frameId));
    hash = hashBytes(hash, &state.sceneId, sizeof(state.sceneId));
    hash = hashBytes(hash, &state.battleMode, sizeof(state.battleMode));
    hash = hashBytes(hash, &state.battleSubMode, sizeof(state.battleSubMode));
    hash = hashBytes(hash, &state.stageId, sizeof(state.stageId));
    hash = hashBytes(hash, &state.roundId, sizeof(state.roundId));
    hash = hashBytes(hash, &state.activeWeather, sizeof(state.activeWeather));
    hash = hashBytes(hash, &state.displayedWeather, sizeof(state.displayedWeather));
    hash = hashBytes(hash, &state.weatherCounter, sizeof(state.weatherCounter));
    hash = hashBytes(hash, &state.randomSeed, sizeof(state.randomSeed));
    hash = hashBytes(hash, &state.p1, sizeof(state.p1));
    return hashBytes(hash, &state.p2, sizeof(state.p2));
}

void captureHand(const SokuLib::CharacterManager &manager, SokuRLBridge::PlayerState &state)
{
    std::fill(std::begin(state.handIds), std::end(state.handIds), -1);
    const auto count = std::min<unsigned>(manager.hand.handCardCount < 0 ? 0U :
        static_cast<unsigned>(manager.hand.handCardCount), 5U);
    if (!manager.hand.handCardBase || manager.hand.handCardMax <= 0)
        return;
    for (unsigned i = 0; i < count; ++i) {
        const auto index = (manager.hand.selectedCard + static_cast<int>(i)) % manager.hand.handCardMax;
        const auto *card = manager.hand.handCardBase[index < 0 ? index + manager.hand.handCardMax : index];
        if (card)
            state.handIds[i] = card->id;
    }
}

void capturePlayer(const SokuLib::CharacterManager &manager, const SokuRLBridge::LogicalInput &input,
    SokuRLBridge::PlayerState &state)
{
    std::memset(&state, 0, sizeof(state));
    state.characterId = manager.characterIndex;
    state.x = manager.objectBase.position.x;
    state.y = manager.objectBase.position.y;
    state.speedX = manager.objectBase.speed.x;
    state.speedY = manager.objectBase.speed.y;
    state.facing = manager.objectBase.direction;
    state.hp = manager.objectBase.hp;
    state.spirit = manager.currentSpirit;
    state.maxSpirit = manager.maxSpirit;
    state.cardGauge = manager.cardGauge;
    state.cardCount = manager.cardCount;
    captureHand(manager, state);
    state.actionId = static_cast<std::uint32_t>(manager.objectBase.action);
    state.sequenceId = manager.objectBase.actionBlockId;
    state.subsequenceId = manager.objectBase.animationCounter;
    state.animationFrame = manager.objectBase.frameData ? manager.objectBase.frameData->number : 0;
    state.elapsedInSubsequence = manager.objectBase.frameCount;
    state.hitstop = manager.objectBase.hitstop;
    state.untech = manager.untech;
    if (manager.objectBase.frameData) {
        state.airborne = manager.objectBase.frameData->frameFlags.airborne ? 1U : 0U;
        state.frameFlags = manager.objectBase.frameData->frameFlags.value;
        state.attackFlags = manager.objectBase.frameData->attackFlags.value;
    }
    state.objectCount = manager.objects.list.size;
    state.input = input;
}

SokuRLBridge::RawFrameState captureState(SokuLib::BattleManager *manager, std::uint64_t frame)
{
    SokuRLBridge::RawFrameState state{};
    state.frameId = frame;
    state.segmentId = g_segmentId;
    state.sceneId = *reinterpret_cast<const std::uint32_t *>(SokuLib::ADDR_SCENE_ID);
    state.battleMode = static_cast<std::uint32_t>(SokuLib::mainMode);
    state.battleSubMode = static_cast<std::uint32_t>(SokuLib::subMode);
    state.stageId = SokuLib::gameParams.stageId;
    state.roundId = static_cast<unsigned char>(manager->currentRound);
    state.timeElapsedRaw = *reinterpret_cast<const std::uint32_t *>(SokuLib::ADDR_TIME_ELAPSED);
    state.activeWeather = static_cast<std::uint32_t>(SokuLib::activeWeather);
    state.displayedWeather = static_cast<std::uint32_t>(SokuLib::displayedWeather);
    state.weatherCounter = SokuLib::weatherCounter;
    state.randomSeed = SokuLib::gameParams.randomSeed;
    capturePlayer(manager->leftCharacterManager, g_effectiveInputs[0], state.p1);
    capturePlayer(manager->rightCharacterManager, g_effectiveInputs[1], state.p2);
    state.stateHash = stateHash(state);
    return state;
}

void publishLatest(const SokuRLBridge::RawFrameState &state)
{
    beginStatusWrite();
    g_control->currentFrame = state.frameId;
    g_control->latest = state;
    g_control->recordedFrames = g_history.empty() ? 0 : g_history.size();
    g_control->stepsRemaining = g_stepsRemaining;
    endStatusWrite();
}

void pushRing(const SokuRLBridge::RawFrameState &state)
{
    const auto write = load32(&g_control->ringWriteSeq);
    const auto read = load32(&g_control->ringReadSeq);
    if (write - read >= SokuRLBridge::FRAME_RING_CAPACITY) {
        InterlockedIncrement(reinterpret_cast<volatile LONG *>(&g_control->droppedFrames));
        return;
    }
    g_mapping->frames[write % SokuRLBridge::FRAME_RING_CAPACITY] = state;
    MemoryBarrier();
    store32(&g_control->ringWriteSeq, write + 1);
}

void appendRecordedFrame(const SokuRLBridge::RawFrameState &state)
{
    if (load32(&g_control->checkpointValid)) {
        if (g_history.size() != state.frameId) {
            invalidateCheckpoint(SokuRLBridge::ResultCode::CheckpointInvalidated);
        } else if (g_history.size() < SokuRLBridge::INPUT_HISTORY_CAPACITY) {
            g_history.push_back(state);
        } else {
            invalidateCheckpoint(SokuRLBridge::ResultCode::HistoryFull);
        }
    }
    publishLatest(state);
    pushRing(state);
}

void clearControlledInput(SokuRLBridge::ResultCode result, bool neutral)
{
    g_activeInput = {};
    g_activeInputFrames = 0;
    g_activeInputEnabled = false;
    g_neutralPending = neutral;
    store32(&g_control->inputFramesRemaining, 0);
    publishResult(result);
}

void consumeCommand(bool gameplay)
{
    const auto sequence = load32(&g_control->commandSeq);
    if (sequence == g_lastCommandSeq)
        return;
    MemoryBarrier();
    const auto type = static_cast<SokuRLBridge::CommandType>(load32(&g_control->commandType));
    const auto input = g_control->commandInput;
    const auto duration = g_control->durationFrames;
    const auto argument = g_control->commandArgument;
    g_lastCommandSeq = sequence;

    if (type == SokuRLBridge::CommandType::Release) {
        clearControlledInput(SokuRLBridge::ResultCode::Released, gameplay);
    } else if (!gameplay) {
        publishResult(SokuRLBridge::ResultCode::NotInGameplay);
    } else if (type == SokuRLBridge::CommandType::Input && isValidInput(input, duration)) {
        g_activeInput = toKeyInput(input);
        g_activeInputFrames = duration;
        g_activeInputEnabled = true;
        g_neutralPending = false;
        store32(&g_control->inputFramesRemaining, duration);
        publishResult(SokuRLBridge::ResultCode::Accepted);
    } else if (type == SokuRLBridge::CommandType::Run) {
        g_paused = false;
        g_stepsRemaining = 0;
        publishResult(SokuRLBridge::ResultCode::Accepted);
    } else if (type == SokuRLBridge::CommandType::Pause) {
        g_paused = true;
        g_stepsRemaining = 0;
        publishResult(SokuRLBridge::ResultCode::Complete);
    } else if (type == SokuRLBridge::CommandType::StepFrames && duration > 0 && duration <= 10000) {
        g_paused = true;
        g_stepsRemaining = duration;
        publishResult(SokuRLBridge::ResultCode::Accepted);
    } else if (type == SokuRLBridge::CommandType::EstablishCheckpoint) {
        // Returning SCENE_LOADING from an active Practice battle crashes th123
        // 1.10a. Keep checkpoint restore unavailable until a proven reset entry
        // point is identified.
        publishResult(SokuRLBridge::ResultCode::CheckpointRestoreUnsupported);
    } else if (type == SokuRLBridge::CommandType::GotoFrame) {
        (void)argument;
        publishResult(SokuRLBridge::ResultCode::CheckpointRestoreUnsupported);
    } else {
        publishResult(SokuRLBridge::ResultCode::InvalidCommand);
    }
    acknowledge(sequence);
}

int playerIndexFor(SokuLib::KeymapManager *self)
{
    if (!isPracticeGameplay())
        return -1;
    auto &manager = SokuLib::getBattleMgr();
    const auto left = manager.leftCharacterManager.keyManager;
    const auto right = manager.rightCharacterManager.keyManager;
    if (left && left->keymapManager == self)
        return 0;
    if (right && right->keymapManager == self)
        return 1;
    return -1;
}

void __fastcall keymapManagerSetInputs(SokuLib::KeymapManager *self)
{
    (self->*g_originalSetInputs)();
    if (!g_control)
        return;
    if (*reinterpret_cast<const int *>(SokuLib::ADDR_SCENE_ID) == 3) {
        const auto sequence = load32(&g_control->commandSeq);
        if (sequence != g_lastCommandSeq) {
            MemoryBarrier();
            const auto type = static_cast<SokuRLBridge::CommandType>(load32(&g_control->commandType));
            if (type == SokuRLBridge::CommandType::MenuConfirm) {
                self->input.a = 1;
                g_lastCommandSeq = sequence;
                publishResult(SokuRLBridge::ResultCode::Complete);
                acknowledge(sequence);
            }
        }
        return;
    }
    if (!g_inSimulationUpdate)
        return;
    const auto player = playerIndexFor(self);
    if (player < 0)
        return;

    if (g_reconstructing && g_replayInputFrame < g_history.size()) {
        const auto replayIndex = static_cast<std::size_t>(g_replayInputFrame);
        const auto &recorded = player == 0 ? g_history[replayIndex].p1.input :
            g_history[replayIndex].p2.input;
        self->input = toKeyInput(recorded);
    } else if (player == 0) {
        if (g_neutralPending) {
            self->input = {};
            g_neutralPending = false;
        } else if (g_activeInputEnabled && g_activeInputFrames) {
            self->input = g_activeInput;
        }
    }
    g_effectiveInputs[player] = toLogicalInput(self->input);
}

int callSimulationUpdate(SokuLib::BattleManager *manager)
{
    g_effectiveInputs[0] = {};
    g_effectiveInputs[1] = {};
    g_inSimulationUpdate = true;
    const auto result = (manager->*g_originalBattleManagerProcess)();
    g_inSimulationUpdate = false;
    if (!g_reconstructing && g_activeInputEnabled && g_activeInputFrames) {
        --g_activeInputFrames;
        store32(&g_control->inputFramesRemaining, g_activeInputFrames);
        if (!g_activeInputFrames) {
            g_activeInputEnabled = false;
            g_neutralPending = true;
            publishResult(SokuRLBridge::ResultCode::Complete);
        }
    }
    return result;
}

bool initializeRestartedBattle(SokuLib::BattleManager *manager)
{
    if (!g_awaitingRestart)
        return false;
    g_awaitingRestart = false;
    ++g_segmentId;
    g_currentFrame = 0;
    g_stepsRemaining = 0;
    g_paused = true;
    g_effectiveInputs[0] = {};
    g_effectiveInputs[1] = {};
    auto initial = captureState(manager, 0);

    if (g_establishAfterRestart) {
        g_history.clear();
        g_history.push_back(initial);
        setCheckpointValid(true);
        store32(&g_control->validationState,
            static_cast<std::uint32_t>(SokuRLBridge::ValidationState::Unknown));
        g_control->lastVerifiedFrame = SokuRLBridge::NO_FRAME;
        g_control->firstDivergentFrame = SokuRLBridge::NO_FRAME;
        publishLatest(initial);
        pushRing(initial);
        publishResult(SokuRLBridge::ResultCode::Complete);
        return true;
    }

    g_reconstructing = true;
    store32(&g_control->reconstructing, 1);
    store32(&g_control->runState, static_cast<std::uint32_t>(SokuRLBridge::RunState::Reconstructing));
    std::uint64_t lastVerified = SokuRLBridge::NO_FRAME;
    std::uint64_t firstDivergent = SokuRLBridge::NO_FRAME;
    if (g_history.empty() || initial.stateHash != g_history[0].stateHash) {
        firstDivergent = 0;
    } else {
        lastVerified = 0;
        for (std::uint64_t frame = 1; frame <= g_reconstructionTarget; ++frame) {
            g_replayInputFrame = frame;
            const auto result = callSimulationUpdate(manager);
            g_currentFrame = frame;
            auto reconstructed = captureState(manager, frame);
            publishLatest(reconstructed);
            const auto historyIndex = static_cast<std::size_t>(frame);
            if (reconstructed.stateHash != g_history[historyIndex].stateHash) {
                firstDivergent = frame;
                break;
            }
            lastVerified = frame;
            if (result > 0 && result < 4 && frame != g_reconstructionTarget) {
                firstDivergent = frame;
                break;
            }
        }
    }
    g_reconstructing = false;
    store32(&g_control->reconstructing, 0);
    g_control->lastVerifiedFrame = lastVerified;
    g_control->firstDivergentFrame = firstDivergent;
    if (firstDivergent != SokuRLBridge::NO_FRAME) {
        store32(&g_control->validationState,
            static_cast<std::uint32_t>(SokuRLBridge::ValidationState::Diverged));
        publishResult(SokuRLBridge::ResultCode::Diverged);
    } else {
        store32(&g_control->validationState,
            static_cast<std::uint32_t>(SokuRLBridge::ValidationState::Deterministic));
        publishResult(SokuRLBridge::ResultCode::Complete);
    }
    return true;
}

int __fastcall battleManagerOnProcess(SokuLib::BattleManager *manager)
{
    if (!g_control)
        return (manager->*g_originalBattleManagerProcess)();
    const bool gameplay = isPracticeGameplay();
    store32(&g_control->inGameplay, gameplay ? 1U : 0U);
    consumeCommand(gameplay);
    if (!gameplay) {
        invalidateCheckpoint(SokuRLBridge::ResultCode::CheckpointInvalidated);
        return (manager->*g_originalBattleManagerProcess)();
    }
    if (initializeRestartedBattle(manager) || g_restartRequested) {
        store32(&g_control->runState, static_cast<std::uint32_t>(SokuRLBridge::RunState::Paused));
        return 0;
    }
    if (load32(&g_control->checkpointValid) && !identityMatchesCheckpoint())
        invalidateCheckpoint(SokuRLBridge::ResultCode::CheckpointInvalidated);
    if (g_paused && !g_stepsRemaining) {
        store32(&g_control->runState, static_cast<std::uint32_t>(SokuRLBridge::RunState::Paused));
        return 0;
    }

    const auto updates = g_stepsRemaining ? g_stepsRemaining : 1U;
    store32(&g_control->runState, static_cast<std::uint32_t>(
        g_stepsRemaining ? SokuRLBridge::RunState::Stepping : SokuRLBridge::RunState::Running));
    int result = 0;
    for (std::uint32_t i = 0; i < updates; ++i) {
        if (load32(&g_control->checkpointValid) && g_history.size() > g_currentFrame + 1) {
            g_history.resize(static_cast<std::size_t>(g_currentFrame + 1));
            store32(&g_control->validationState,
                static_cast<std::uint32_t>(SokuRLBridge::ValidationState::Unknown));
        }
        result = callSimulationUpdate(manager);
        ++g_currentFrame;
        appendRecordedFrame(captureState(manager, g_currentFrame));
        if (g_stepsRemaining)
            --g_stepsRemaining;
        if (result > 0 && result < 4)
            break;
    }
    if (g_paused || g_stepsRemaining == 0 && updates > 1) {
        g_paused = true;
        g_stepsRemaining = 0;
        store32(&g_control->runState, static_cast<std::uint32_t>(SokuRLBridge::RunState::Paused));
        publishResult(SokuRLBridge::ResultCode::Complete);
    }
    return result;
}

int __fastcall battleOnProcess(SokuLib::Battle *battle)
{
    const auto result = (battle->*g_originalBattleProcess)();
    if (!g_restartRequested)
        return result;
    g_restartRequested = false;
    g_awaitingRestart = true;
    SokuLib::gameParams.randomSeed = g_checkpoint.randomSeed;
    return SokuLib::SCENE_LOADING;
}

bool createMapping()
{
    wchar_t mappingName[64]{};
    if (swprintf_s(mappingName, SokuRLBridge::MAPPING_NAME_FORMAT, GetCurrentProcessId()) < 0)
        return false;
    g_fileMapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
        sizeof(SokuRLBridge::BridgeMapping), mappingName);
    if (!g_fileMapping)
        return false;
    g_mapping = static_cast<SokuRLBridge::BridgeMapping *>(MapViewOfFile(
        g_fileMapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SokuRLBridge::BridgeMapping)));
    if (!g_mapping) {
        CloseHandle(g_fileMapping);
        g_fileMapping = nullptr;
        return false;
    }
    std::memset(g_mapping, 0, sizeof(*g_mapping));
    g_control = &g_mapping->control;
    g_control->magic = SokuRLBridge::CONTROL_MAGIC;
    g_control->version = SokuRLBridge::CONTROL_VERSION;
    g_control->structSize = sizeof(SokuRLBridge::ControlBlock);
    g_control->mappingSize = sizeof(SokuRLBridge::BridgeMapping);
    g_control->ringCapacity = SokuRLBridge::FRAME_RING_CAPACITY;
    g_control->lastVerifiedFrame = SokuRLBridge::NO_FRAME;
    g_control->firstDivergentFrame = SokuRLBridge::NO_FRAME;
    g_control->connected = 1;
    return true;
}

void closeMapping()
{
    if (g_control)
        store32(&g_control->connected, 0);
    if (g_mapping)
        UnmapViewOfFile(g_mapping);
    if (g_fileMapping)
        CloseHandle(g_fileMapping);
    g_mapping = nullptr;
    g_control = nullptr;
    g_fileMapping = nullptr;
}

bool installHooks()
{
    DWORD textProtection = 0;
    if (!VirtualProtect(reinterpret_cast<void *>(TEXT_SECTION_OFFSET), TEXT_SECTION_SIZE,
            PAGE_EXECUTE_WRITECOPY, &textProtection))
        return false;
    g_originalSetInputs = SokuLib::union_cast<SetInputsMethod>(
        SokuLib::TamperNearJmpOpr(KEYMAP_SET_INPUTS_HOOK, keymapManagerSetInputs));
    DWORD ignored = 0;
    VirtualProtect(reinterpret_cast<void *>(TEXT_SECTION_OFFSET), TEXT_SECTION_SIZE,
        textProtection, &ignored);

    DWORD rdataProtection = 0;
    if (!VirtualProtect(reinterpret_cast<void *>(RDATA_SECTION_OFFSET), RDATA_SECTION_SIZE,
            PAGE_EXECUTE_WRITECOPY, &rdataProtection))
        return false;
    g_originalBattleManagerProcess = SokuLib::TamperDword(
        &SokuLib::VTable_BattleManager.onProcess, battleManagerOnProcess);
    VirtualProtect(reinterpret_cast<void *>(RDATA_SECTION_OFFSET), RDATA_SECTION_SIZE,
        rdataProtection, &ignored);
    FlushInstructionCache(GetCurrentProcess(), nullptr, 0);
    return g_originalSetInputs && g_originalBattleManagerProcess;
}
}

extern "C" __declspec(dllexport) bool CheckVersion(const BYTE hash[16])
{
    return std::memcmp(hash, SokuLib::targetHash, sizeof(SokuLib::targetHash)) == 0;
}

extern "C" __declspec(dllexport) bool Initialize(HMODULE, HMODULE)
{
    static_assert(sizeof(void *) == 4, "SokuRLBridge must be built for Win32/x86");
    g_history.reserve(SokuRLBridge::INPUT_HISTORY_CAPACITY);
    if (!createMapping())
        return false;
    if (!installHooks()) {
        closeMapping();
        return false;
    }
    return true;
}

BOOL APIENTRY DllMain(HMODULE, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_DETACH)
        closeMapping();
    return TRUE;
}
