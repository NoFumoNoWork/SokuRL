#include "ControlBlock.hpp"
#include "ImageCapture.hpp"
#include "AudioMute.hpp"
#include "SceneReset.hpp"

#include <BattleManager.hpp>
#include <BattleMode.hpp>
#include <Character.hpp>
#include <Hash.hpp>
#include <InputManager.hpp>
#include <PracticeSettings.hpp>
#include <Profile.hpp>
#include <Scenes.hpp>
#include <SokuAddresses.hpp>
#include <Tamper.hpp>
#include <UnionCast.hpp>
#include <VTables.hpp>
#include <Weather.hpp>

#include <Windows.h>
#include <algorithm>
#include <cwchar>
#include <cstring>
#include <limits>
#include <vector>

namespace
{
using SetInputsMethod = void (SokuLib::KeymapManager::*)();
using BattleProcessMethod = int (SokuLib::Battle::*)();
using BattleRenderMethod = int (SokuLib::Battle::*)();
using BattleManagerProcessMethod = int (SokuLib::BattleManager::*)();
using SelectProcessMethod = int (SokuLib::Select::*)();
using TitleProcessMethod = int (SokuLib::Title::*)();
using ProfileInitializeMethod = void (__thiscall *)(SokuLib::Profile *, char);
using WaitForSingleObjectFunction = DWORD (WINAPI *)(HANDLE, DWORD);

constexpr DWORD KEYMAP_SET_INPUTS_HOOK = 0x0040A45D;
constexpr DWORD P1_KEYMAP_MANAGER_PTR = 0x008989A0;
constexpr DWORD P2_KEYMAP_MANAGER_PTR = 0x0089918C;
constexpr DWORD P1_INPUT_MANAGER_PTR = 0x00898680;
constexpr DWORD P2_INPUT_MANAGER_PTR = 0x00898684;
constexpr DWORD P1_INPUT_DEVICE = 0x00898678;
constexpr DWORD INPUT_MANAGER_CLUSTER_DEVICE = 0x0089A2BC;
constexpr DWORD FALLBACK_KEY_MANAGER = 0x008986A8;
constexpr DWORD PROFILE_INITIALIZE = 0x00434BF0;
constexpr std::uint32_t LOCAL_BATTLE_SCENE = 5;
constexpr DWORD RENDER_BRANCH = 0x00407FAE;
constexpr DWORD RENDER_PATH = 0x00407FB4;
constexpr DWORD SKIP_RENDER_PATH = 0x00408048;
constexpr DWORD FRAME_WAIT_CALL_OPERAND = 0x00419689;
constexpr DWORD WAIT_FOR_SINGLE_OBJECT_IAT = 0x008570A0;
constexpr std::uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

HANDLE g_fileMapping = nullptr;
SokuRLBridge::BridgeMapping *g_mapping = nullptr;
SokuRLBridge::ControlBlock *g_control = nullptr;
SetInputsMethod g_originalSetInputs = nullptr;
BattleProcessMethod g_originalBattleProcess = nullptr;
BattleRenderMethod g_originalBattleRender = nullptr;
bool g_captureImages = false;
bool g_captureStateOnly = false;
bool g_renderPending = true;
BattleManagerProcessMethod g_originalBattleManagerProcess = nullptr;
SelectProcessMethod g_originalSelectProcess = nullptr;
TitleProcessMethod g_originalTitleProcess = nullptr;

std::uint32_t g_lastCommandSeq = 0;
std::uint32_t g_segmentId = 0;
std::uint64_t g_currentFrame = 0;
std::uint32_t g_stepsRemaining = 0;
bool g_paused = false;
bool g_inSimulationUpdate = false;
bool g_reconstructing = false;
bool g_battleActive = false;
bool g_checkpointArmed = false;
bool g_checkpointSeedRequested = false;
std::uint32_t g_requestedCheckpointSeed = 0;
bool g_restartRequested = false;
bool g_awaitingRestart = false;
bool g_establishAfterRestart = false;
std::uint64_t g_reconstructionTarget = 0;
std::uint64_t g_replayInputFrame = 0;

SokuRLBridge::LogicalInput g_effectiveInputs[2]{};
SokuLib::KeyInput g_activeInputs[2]{};
std::uint32_t g_activeInputMask = 0;
std::uint32_t g_activeInputFrames = 0;
bool g_activeInputEnabled = false;
bool g_neutralPending = false;
bool g_vsBootstrapArmed = false;
bool g_vsBootstrapComplete = false;
bool g_episodeResetRequested = false;
bool g_headlessRender = false;
bool g_unlimitedPacing = false;
bool g_vsPauseAtStart = false;
bool g_vsSeedRequested = false;
std::uint32_t g_vsSeed = 0;
std::uint32_t g_vsP1Character = 1;
std::uint32_t g_vsP2Character = 0;
std::uint32_t g_vsP1Palette = 0;
std::uint32_t g_vsP2Palette = 0;
std::uint32_t g_vsP1Deck = 0;
std::uint32_t g_vsP2Deck = 0;
std::uint32_t g_vsStage = 0;
std::uint32_t g_vsMusic = 0;

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

bool isReplayGameplay()
{
    return *reinterpret_cast<const int *>(SokuLib::ADDR_SCENE_ID) == LOCAL_BATTLE_SCENE &&
        SokuLib::subMode == SokuLib::BATTLE_SUBMODE_REPLAY;
}

bool isLocalVersusGameplay()
{
    return *reinterpret_cast<const int *>(SokuLib::ADDR_SCENE_ID) == LOCAL_BATTLE_SCENE &&
        SokuLib::mainMode == SokuLib::BATTLE_MODE_VSPLAYER;
}

bool isSupportedGameplay()
{
    return isPracticeGameplay() || isLocalVersusGameplay() || isReplayGameplay();
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

std::uint32_t environmentValue(const wchar_t *name, std::uint32_t fallback)
{
    wchar_t value[16]{};
    const auto length = GetEnvironmentVariableW(name, value, _countof(value));
    if (!length || length >= _countof(value))
        return fallback;
    wchar_t *end = nullptr;
    const auto parsed = wcstoul(value, &end, 10);
    return end && *end == L'\0' ? static_cast<std::uint32_t>(parsed) : fallback;
}

void __declspec(naked) renderBranchDispatch()
{
    __asm {
        // Preserve the original JNE first. The injected JMP does not alter EFLAGS.
        jne originalSkip
        cmp dword ptr ds:[008A0044h], 5
        jne originalRender
        cmp byte ptr [g_headlessRender], 0
        jne originalSkip
        cmp byte ptr [g_captureImages], 0
        je originalRender
        cmp byte ptr [g_renderPending], 0
        je originalSkip
    originalRender:
        push 00407FB4h
        ret
    originalSkip:
        push 00408048h
        ret
    }
}

bool installHeadlessRenderHook()
{
    if (!g_headlessRender && !g_captureImages)
        return true;

    auto *branch = reinterpret_cast<unsigned char *>(RENDER_BRANCH);
    if (branch[0] != 0x0F || branch[1] != 0x85)
        return false;
    std::int32_t originalDisplacement = 0;
    std::memcpy(&originalDisplacement, branch + 2, sizeof(originalDisplacement));
    if (RENDER_BRANCH + 6 + originalDisplacement != SKIP_RENDER_PATH)
        return false;

    const auto displacement = static_cast<std::int32_t>(
        reinterpret_cast<std::uintptr_t>(renderBranchDispatch) - (RENDER_BRANCH + 5));
    branch[0] = 0xE9;
    std::memcpy(branch + 1, &displacement, sizeof(displacement));
    branch[5] = 0x90;
    return true;
}

DWORD WINAPI framePacingWait(HANDLE object, DWORD timeout)
{
    if (g_unlimitedPacing &&
        *reinterpret_cast<const int *>(SokuLib::ADDR_SCENE_ID) == LOCAL_BATTLE_SCENE &&
        SokuLib::mainMode == SokuLib::BATTLE_MODE_VSPLAYER)
        timeout = 0;
    return WaitForSingleObject(object, timeout);
}

WaitForSingleObjectFunction g_framePacingWait = framePacingWait;

bool installUnlimitedPacingHook()
{
    if (!g_unlimitedPacing)
        return true;
    auto *operand = reinterpret_cast<DWORD *>(FRAME_WAIT_CALL_OPERAND);
    if (*operand != WAIT_FOR_SINGLE_OBJECT_IAT)
        return false;
    *operand = reinterpret_cast<DWORD>(&g_framePacingWait);
    return true;
}

void configureVsPlayer(SokuLib::PlayerInfo &info, bool right, std::uint32_t character,
    std::uint32_t palette, std::uint32_t deck)
{
    const auto initializeProfile = reinterpret_cast<ProfileInitializeMethod>(PROFILE_INITIALIZE);
    info.character = static_cast<SokuLib::Character>(character);
    info.isRight = right;
    info.palette = static_cast<unsigned char>(palette);
    info.deck = static_cast<unsigned char>(deck);
    info.effectiveDeck.clear();

    if (right) {
        *reinterpret_cast<SokuLib::KeyManager **>(P2_INPUT_MANAGER_PTR) =
            reinterpret_cast<SokuLib::KeyManager *>(FALLBACK_KEY_MANAGER);
        initializeProfile(&SokuLib::profile2, -1);
        info.keyManager = reinterpret_cast<SokuLib::KeyManager **>(P2_KEYMAP_MANAGER_PTR);
        auto &source = SokuLib::profile2.cards[character][deck];
        for (int i = 0; i < source.size; ++i)
            info.effectiveDeck.push_back(source[i]);
        return;
    }

    *reinterpret_cast<signed char *>(P1_INPUT_DEVICE) = -1;
    *reinterpret_cast<SokuLib::KeyManager **>(P1_INPUT_MANAGER_PTR) =
        reinterpret_cast<SokuLib::KeyManager *>(FALLBACK_KEY_MANAGER);
    initializeProfile(&SokuLib::profile1, -1);
    info.keyManager = reinterpret_cast<SokuLib::KeyManager **>(P1_KEYMAP_MANAGER_PTR);
    auto &source = SokuLib::profile1.cards[character][deck];
    for (int i = 0; i < source.size; ++i)
        info.effectiveDeck.push_back(source[i]);
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
    hash = hashBytes(hash, &state.timeElapsedRaw, sizeof(state.timeElapsedRaw));
    hash = hashBytes(hash, &state.activeWeather, sizeof(state.activeWeather));
    hash = hashBytes(hash, &state.displayedWeather, sizeof(state.displayedWeather));
    hash = hashBytes(hash, &state.weatherCounter, sizeof(state.weatherCounter));
    hash = hashBytes(hash, &state.randomSeed, sizeof(state.randomSeed));
    hash = hashBytes(hash, &state.p1, sizeof(state.p1));
    hash = hashBytes(hash, &state.p2, sizeof(state.p2));
    hash = hashBytes(hash, &state.p1ObjectCount, sizeof(state.p1ObjectCount));
    hash = hashBytes(hash, &state.p2ObjectCount, sizeof(state.p2ObjectCount));
    hash = hashBytes(hash, &state.p1ObjectOverflow, sizeof(state.p1ObjectOverflow));
    hash = hashBytes(hash, &state.p2ObjectOverflow, sizeof(state.p2ObjectOverflow));
    hash = hashBytes(hash, state.p1Objects, sizeof(state.p1Objects));
    return hashBytes(hash, state.p2Objects, sizeof(state.p2Objects));
}

void captureObject(const SokuLib::ObjectManager &object, std::uint32_t owner,
    std::uint32_t index, SokuRLBridge::ObjectState &state)
{
    const auto &projectile = reinterpret_cast<const SokuLib::ProjectileManager &>(object);
    state.ownerIndex = owner;
    state.listIndex = index;
    state.typeId = reinterpret_cast<std::uint32_t>(object.vtable);
    state.actionId = static_cast<std::uint32_t>(object.action);
    state.actionBlockId = object.actionBlockId;
    state.animationCounter = object.animationCounter;
    state.animationSubFrame = object.animationSubFrame;
    state.frameCount = object.frameCount;
    state.x = object.position.x;
    state.y = object.position.y;
    state.speedX = object.speed.x;
    state.speedY = object.speed.y;
    // Several projectile classes leave this inherited slot uninitialized.
    // Position and speed are stable; expose a canonical value instead of heap noise.
    state.gravity = 0.0f;
    state.direction = object.direction;
    state.hp = object.hp;
    state.hitstop = object.hitstop;
    state.hitBoxCount = object.hitBoxCount;
    state.hurtBoxCount = object.hurtBoxCount;
    state.characterIndex = projectile.characterIndex;
    state.isActive = projectile.isActive;
}

void captureObjects(const SokuLib::CharacterManager &manager, std::uint32_t owner,
    SokuRLBridge::ObjectState (&states)[SokuRLBridge::MAX_OBJECTS_PER_PLAYER],
    std::uint32_t &count, std::uint32_t &overflow)
{
    const auto &list = manager.objects.list;
    overflow = list.size > SokuRLBridge::MAX_OBJECTS_PER_PLAYER ? 1U : 0U;
    if (!list.head || !list.size)
        return;

    auto *node = list.head->next;
    while (node && node != list.head && count < SokuRLBridge::MAX_OBJECTS_PER_PLAYER) {
        if (node->val)
            captureObject(*node->val, owner, count, states[count]);
        else
            overflow = 1;
        ++count;
        node = node->next;
    }
    if (count < std::min<std::uint32_t>(list.size, SokuRLBridge::MAX_OBJECTS_PER_PLAYER))
        overflow = 1;
}

void captureHand(const SokuLib::CharacterManager &manager, SokuRLBridge::PlayerState &state)
{
    std::fill(std::begin(state.handIds), std::end(state.handIds), -1);
    const auto count = std::min<unsigned>(manager.cardCount, 5U);
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
    state.spirit = static_cast<std::int16_t>(manager.currentSpirit);
    state.maxSpirit = static_cast<std::int16_t>(manager.maxSpirit);
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
    state.p1.characterId = static_cast<std::uint32_t>(SokuLib::gameParams.leftPlayerInfo.character);
    state.p2.characterId = static_cast<std::uint32_t>(SokuLib::gameParams.rightPlayerInfo.character);
    captureObjects(manager->leftCharacterManager, 0, state.p1Objects,
        state.p1ObjectCount, state.p1ObjectOverflow);
    captureObjects(manager->rightCharacterManager, 1, state.p2Objects,
        state.p2ObjectCount, state.p2ObjectOverflow);
    state.stateHash = stateHash(state);
    return state;
}

SokuRLBridge::SimpleStatePatch simplePatchFrom(const SokuRLBridge::RawFrameState &state)
{
    SokuRLBridge::SimpleStatePatch patch{};
    patch.timeElapsedRaw = state.timeElapsedRaw;
    patch.activeWeather = state.activeWeather;
    patch.displayedWeather = state.displayedWeather;
    patch.weatherCounter = state.weatherCounter;
    const auto copyPlayer = [](const SokuRLBridge::PlayerState &source,
        SokuRLBridge::SimplePlayerState &target) {
        target.x = source.x;
        target.y = source.y;
        target.speedX = source.speedX;
        target.speedY = source.speedY;
        target.facing = source.facing;
        target.hp = source.hp;
        target.spirit = source.spirit;
        target.maxSpirit = source.maxSpirit;
        target.cardGauge = source.cardGauge;
        target.cardCount = source.cardCount;
    };
    copyPlayer(state.p1, patch.p1);
    copyPlayer(state.p2, patch.p2);
    return patch;
}

void publishReconstructionFrame(const SokuRLBridge::RawFrameState &state)
{
    if (state.frameId >= SokuRLBridge::INPUT_HISTORY_CAPACITY)
        return;
    auto &target = g_mapping->history[state.frameId];
    target.p1Input = state.p1.input;
    target.p2Input = state.p2.input;
    target.simple = simplePatchFrom(state);
    target.stateHash = state.stateHash;
}

void applySimplePlayerState(SokuLib::CharacterManager &manager,
    const SokuRLBridge::SimplePlayerState &state)
{
    manager.objectBase.position.x = state.x;
    manager.objectBase.position.y = state.y;
    manager.objectBase.speed.x = state.speedX;
    manager.objectBase.speed.y = state.speedY;
    manager.objectBase.direction = static_cast<SokuLib::Direction>(state.facing);
    manager.objectBase.hp = static_cast<short>(state.hp);
    manager.currentSpirit = static_cast<unsigned short>(static_cast<std::int16_t>(state.spirit));
    manager.maxSpirit = static_cast<unsigned short>(static_cast<std::int16_t>(state.maxSpirit));
    manager.cardGauge = static_cast<unsigned short>(state.cardGauge);
    manager.cardCount = static_cast<unsigned char>(state.cardCount);
}

bool isValidSimplePlayerState(const SokuRLBridge::SimplePlayerState &state)
{
    constexpr auto minimum = std::numeric_limits<std::int16_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int16_t>::max();
    return state.spirit >= minimum && state.spirit <= maximum &&
        state.maxSpirit >= minimum && state.maxSpirit <= maximum;
}

void applySimpleState(SokuLib::BattleManager &manager,
    const SokuRLBridge::SimpleStatePatch &state)
{
    *reinterpret_cast<std::uint32_t *>(SokuLib::ADDR_TIME_ELAPSED) = state.timeElapsedRaw;
    SokuLib::activeWeather = static_cast<SokuLib::Weather>(state.activeWeather);
    SokuLib::displayedWeather = static_cast<SokuLib::Weather>(state.displayedWeather);
    SokuLib::weatherCounter = static_cast<unsigned short>(state.weatherCounter);
    applySimplePlayerState(manager.leftCharacterManager, state.p1);
    applySimplePlayerState(manager.rightCharacterManager, state.p2);
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
            publishReconstructionFrame(state);
        } else {
            invalidateCheckpoint(SokuRLBridge::ResultCode::HistoryFull);
        }
    }
    publishLatest(state);
    pushRing(state);
}

void clearControlledInput(SokuRLBridge::ResultCode result, bool neutral)
{
    g_activeInputs[0] = {};
    g_activeInputs[1] = {};
    g_activeInputMask = 0;
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
    const auto inputP2 = g_control->commandInputP2;
    const auto duration = g_control->durationFrames;
    const auto argument = g_control->commandArgument;
    g_lastCommandSeq = sequence;

    if (type == SokuRLBridge::CommandType::Release) {
        clearControlledInput(SokuRLBridge::ResultCode::Released, gameplay);
    } else if (!gameplay) {
        publishResult(SokuRLBridge::ResultCode::NotInGameplay);
    } else if (type == SokuRLBridge::CommandType::ResetEpisode &&
        g_vsBootstrapArmed && isLocalVersusGameplay() && argument < 0xFFFFFFFFULL) {
        g_vsSeed = static_cast<std::uint32_t>(argument);
        g_vsSeedRequested = true;
        g_episodeResetRequested = true;
        g_paused = true;
        g_stepsRemaining = 0;
        clearControlledInput(SokuRLBridge::ResultCode::Restarting, true);
    } else if (type == SokuRLBridge::CommandType::Input && isValidInput(input, duration)) {
        g_activeInputs[0] = toKeyInput(input);
        g_activeInputMask = 1;
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
    } else if (type == SokuRLBridge::CommandType::StepWithInputs &&
        isValidInput(input, 1) && isValidInput(inputP2, 1)) {
        g_activeInputs[0] = toKeyInput(input);
        g_activeInputs[1] = toKeyInput(inputP2);
        g_activeInputMask = 3;
        g_activeInputFrames = 1;
        g_activeInputEnabled = true;
        g_neutralPending = false;
        g_paused = true;
        g_stepsRemaining = 1;
        store32(&g_control->inputFramesRemaining, 1);
        publishResult(SokuRLBridge::ResultCode::Accepted);
    } else if (type == SokuRLBridge::CommandType::ApplySimpleState && g_paused &&
        !g_stepsRemaining && isValidSimplePlayerState(g_control->commandPatch.p1) &&
        isValidSimplePlayerState(g_control->commandPatch.p2)) {
        auto &manager = SokuLib::getBattleMgr();
        applySimpleState(manager, g_control->commandPatch);
        const auto patched = captureState(&manager, g_currentFrame);
        publishLatest(patched);
        pushRing(patched);
        publishResult(SokuRLBridge::ResultCode::Complete);
    } else if (type == SokuRLBridge::CommandType::EstablishCheckpoint && !g_battleActive) {
        g_checkpointArmed = true;
        g_paused = true;
        publishResult(SokuRLBridge::ResultCode::Accepted);
    } else if (type == SokuRLBridge::CommandType::EstablishCheckpoint) {
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
    if (!isSupportedGameplay())
        return -1;
    if (self == *reinterpret_cast<SokuLib::KeymapManager **>(P1_KEYMAP_MANAGER_PTR))
        return 0;
    if (self == *reinterpret_cast<SokuLib::KeymapManager **>(P2_KEYMAP_MANAGER_PTR))
        return 1;
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
            } else if (type == SokuRLBridge::CommandType::EstablishCheckpoint) {
                const auto seed = g_control->commandArgument;
                if (SokuLib::practiceSettings)
                    SokuLib::practiceSettings->state = SokuLib::DUMMY_STATE_2P_CONTROL;
                g_checkpointSeedRequested = seed != SokuRLBridge::NO_FRAME;
                if (g_checkpointSeedRequested) {
                    g_requestedCheckpointSeed = static_cast<std::uint32_t>(seed);
                    SokuLib::gameParams.randomSeed = g_requestedCheckpointSeed;
                }
                g_checkpointArmed = true;
                g_paused = true;
                g_lastCommandSeq = sequence;
                publishResult(SokuRLBridge::ResultCode::Accepted);
                acknowledge(sequence);
            }
        }
        return;
    }
    const auto player = playerIndexFor(self);
    if (player < 0)
        return;

    const auto pendingSequence = load32(&g_control->commandSeq);
    if (pendingSequence != g_lastCommandSeq) {
        MemoryBarrier();
        const auto pendingType = static_cast<SokuRLBridge::CommandType>(
            load32(&g_control->commandType));
        if (pendingType == SokuRLBridge::CommandType::StepWithInputs) {
            const auto pending = player == 0 ? g_control->commandInput :
                g_control->commandInputP2;
            self->input = toKeyInput(pending);
            g_effectiveInputs[player] = pending;
            return;
        }
    }
    if (!g_inSimulationUpdate)
        return;

    if (g_reconstructing && g_replayInputFrame < g_history.size()) {
        const auto replayIndex = static_cast<std::size_t>(g_replayInputFrame);
        const auto &recorded = player == 0 ? g_history[replayIndex].p1.input :
            g_history[replayIndex].p2.input;
        self->input = toKeyInput(recorded);
    } else if (g_activeInputEnabled && (g_activeInputMask & (1U << player)) &&
        g_activeInputFrames) {
        self->input = g_activeInputs[player];
    } else if (player == 0) {
        if (g_neutralPending) {
            self->input = {};
            g_neutralPending = false;
        }
    }
    g_effectiveInputs[player] = toLogicalInput(self->input);
}

void applySimulationInputs(SokuLib::BattleManager *manager)
{
    SokuLib::KeyInput inputs[2]{};
    std::uint32_t mask = 0;
    if (g_reconstructing && g_replayInputFrame < g_history.size()) {
        const auto &recorded = g_history[static_cast<std::size_t>(g_replayInputFrame)];
        inputs[0] = toKeyInput(recorded.p1.input);
        inputs[1] = toKeyInput(recorded.p2.input);
        mask = 3;
    } else if (g_activeInputEnabled && g_activeInputFrames) {
        inputs[0] = g_activeInputs[0];
        inputs[1] = g_activeInputs[1];
        mask = g_activeInputMask;
    }
    if (!mask)
        return;

    SokuLib::CharacterManager *characters[2] = {
        &manager->leftCharacterManager, &manager->rightCharacterManager};
    for (int player = 0; player < 2; ++player) {
        if (!(mask & (1U << player)))
            continue;
        auto *character = characters[player];
        character->keyMap = inputs[player];
        if (character->keyManager && character->keyManager->keymapManager)
            character->keyManager->keymapManager->input = inputs[player];
        g_effectiveInputs[player] = toLogicalInput(inputs[player]);
    }
}

int callSimulationUpdate(SokuLib::BattleManager *manager)
{
    if (isPracticeGameplay() && SokuLib::practiceSettings)
        SokuLib::practiceSettings->state = SokuLib::DUMMY_STATE_2P_CONTROL;
    g_effectiveInputs[0] = {};
    g_effectiveInputs[1] = {};
    applySimulationInputs(manager);
    g_inSimulationUpdate = true;
    const auto result = (manager->*g_originalBattleManagerProcess)();
    g_inSimulationUpdate = false;
    if (!g_reconstructing && g_activeInputEnabled && g_activeInputFrames) {
        --g_activeInputFrames;
        store32(&g_control->inputFramesRemaining, g_activeInputFrames);
        if (!g_activeInputFrames) {
            g_activeInputEnabled = false;
            g_activeInputMask = 0;
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
        publishReconstructionFrame(initial);
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
    const bool gameplay = isSupportedGameplay();
    store32(&g_control->inGameplay, gameplay ? 1U : 0U);
    consumeCommand(gameplay);
    if (g_episodeResetRequested)
        return 0;
    if (!gameplay) {
        g_battleActive = false;
        invalidateCheckpoint(SokuRLBridge::ResultCode::CheckpointInvalidated);
        return (manager->*g_originalBattleManagerProcess)();
    }
    if (!g_battleActive) {
        g_battleActive = true;
        g_currentFrame = 0;
        g_stepsRemaining = 0;
        g_effectiveInputs[0] = {};
        g_effectiveInputs[1] = {};
        if (g_checkpointArmed && isPracticeGameplay() && SokuLib::practiceSettings)
            SokuLib::practiceSettings->state = SokuLib::DUMMY_STATE_2P_CONTROL;
        if (g_checkpointArmed && g_checkpointSeedRequested)
            SokuLib::gameParams.randomSeed = g_requestedCheckpointSeed;
        const auto initial = captureState(manager, 0);
        if (g_checkpointArmed) {
            g_checkpointArmed = false;
            g_checkpointSeedRequested = false;
            g_checkpoint = readIdentity();
            g_history.clear();
            g_history.push_back(initial);
            publishReconstructionFrame(initial);
            setCheckpointValid(true);
            g_paused = true;
            store32(&g_control->validationState,
                static_cast<std::uint32_t>(SokuRLBridge::ValidationState::Unknown));
            g_control->lastVerifiedFrame = SokuRLBridge::NO_FRAME;
            g_control->firstDivergentFrame = SokuRLBridge::NO_FRAME;
            publishLatest(initial);
            pushRing(initial);
            publishResult(SokuRLBridge::ResultCode::Complete);
            store32(&g_control->runState,
                static_cast<std::uint32_t>(SokuRLBridge::RunState::Paused));
            return 0;
        }
        publishLatest(initial);
        pushRing(initial);
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
        g_renderPending = true;
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

int __fastcall selectOnProcess(SokuLib::Select *select)
{
    if (g_checkpointArmed && SokuLib::practiceSettings)
        SokuLib::practiceSettings->state = SokuLib::DUMMY_STATE_2P_CONTROL;
    const auto result = (select->*g_originalSelectProcess)();
    if (g_checkpointArmed && g_checkpointSeedRequested)
        SokuLib::gameParams.randomSeed = g_requestedCheckpointSeed;
    return result;
}

int __fastcall titleOnProcess(SokuLib::Title *title)
{
    // Scene deletion is asynchronous. A new battle must not reuse global
    // resources while the previous battle is still destroying them.
    if (g_vsBootstrapArmed && !g_vsBootstrapComplete &&
        !SokuRLBridge::retiredBattleSceneDestroyed())
        return SokuLib::SCENE_TITLE;
    const auto result = (title->*g_originalTitleProcess)();
    if (!g_vsBootstrapArmed || g_vsBootstrapComplete)
        return result;

    *reinterpret_cast<signed char *>(INPUT_MANAGER_CLUSTER_DEVICE) = -1;
    SokuLib::setBattleMode(SokuLib::BATTLE_MODE_VSPLAYER,
        SokuLib::BATTLE_SUBMODE_PLAYING1);
    configureVsPlayer(SokuLib::leftPlayerInfo, false, g_vsP1Character,
        g_vsP1Palette, g_vsP1Deck);
    configureVsPlayer(SokuLib::rightPlayerInfo, true, g_vsP2Character,
        g_vsP2Palette, g_vsP2Deck);
    SokuLib::gameParams.stageId = static_cast<unsigned char>(g_vsStage);
    SokuLib::gameParams.musicId = static_cast<unsigned char>(g_vsMusic);
    if (g_vsSeedRequested)
        SokuLib::gameParams.randomSeed = g_vsSeed;
    if (g_vsPauseAtStart) {
        g_checkpointArmed = true;
        g_checkpointSeedRequested = g_vsSeedRequested;
        g_requestedCheckpointSeed = g_vsSeed;
        g_paused = true;
    }
    g_vsBootstrapComplete = true;
    return SokuLib::SCENE_LOADING;
}

int __fastcall battleOnProcess(SokuLib::Battle *battle)
{
    const auto result = (battle->*g_originalBattleProcess)();
    if (g_episodeResetRequested) {
        SokuRLBridge::retireBattleScene(battle);
        // Return through the engine's normal scene lifecycle. It destroys the
        // old battle asynchronously; titleOnProcess waits for that destruction
        // before loading the next one. The process and DLL stay alive.
        g_episodeResetRequested = false;
        g_vsBootstrapComplete = false;
        g_battleActive = false;
        g_checkpointArmed = false;
        g_checkpointSeedRequested = false;
        g_restartRequested = false;
        g_awaitingRestart = false;
        g_reconstructing = false;
        g_currentFrame = 0;
        ++g_segmentId;
        g_history.clear();
        g_effectiveInputs[0] = {};
        g_effectiveInputs[1] = {};
        g_neutralPending = false;
        g_renderPending = true;
        beginStatusWrite();
        g_control->inGameplay = 0;
        g_control->checkpointValid = 0;
        g_control->reconstructing = 0;
        g_control->currentFrame = 0;
        g_control->recordedFrames = 0;
        g_control->stepsRemaining = 0;
        g_control->ringReadSeq = 0;
        g_control->ringWriteSeq = 0;
        g_control->droppedFrames = 0;
        g_control->lastVerifiedFrame = SokuRLBridge::NO_FRAME;
        g_control->firstDivergentFrame = SokuRLBridge::NO_FRAME;
        endStatusWrite();
        SokuRLBridge::resetImageCapture();
        return SokuLib::SCENE_TITLE;
    }
    if (g_captureStateOnly && g_battleActive && g_control && isSupportedGameplay())
        SokuRLBridge::captureImage(g_currentFrame);
    if (!g_restartRequested)
        return result;
    g_restartRequested = false;
    g_awaitingRestart = true;
    SokuLib::gameParams.randomSeed = g_checkpoint.randomSeed;
    return SokuLib::SCENE_LOADING;
}

int __fastcall battleOnRender(SokuLib::Battle *battle)
{
    const auto result = (battle->*g_originalBattleRender)();
    if (g_battleActive && g_control && isSupportedGameplay()) {
        SokuRLBridge::captureImage(g_currentFrame);
        g_renderPending = false;
    }
    return result;
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
    SokuRLBridge::closeImageCapture();
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
    const bool headlessHookInstalled = installHeadlessRenderHook();
    const bool unlimitedHookInstalled = installUnlimitedPacingHook();
    DWORD ignored = 0;
    VirtualProtect(reinterpret_cast<void *>(TEXT_SECTION_OFFSET), TEXT_SECTION_SIZE,
        textProtection, &ignored);

    DWORD rdataProtection = 0;
    if (!VirtualProtect(reinterpret_cast<void *>(RDATA_SECTION_OFFSET), RDATA_SECTION_SIZE,
            PAGE_EXECUTE_WRITECOPY, &rdataProtection))
        return false;
    g_originalBattleManagerProcess = SokuLib::TamperDword(
        &SokuLib::VTable_BattleManager.onProcess, battleManagerOnProcess);
    g_originalSelectProcess = SokuLib::TamperDword(
        &SokuLib::VTable_Select.onProcess, selectOnProcess);
    g_originalTitleProcess = SokuLib::TamperDword(
        &SokuLib::VTable_Title.onProcess, titleOnProcess);
    const bool resetBarrierInstalled = SokuRLBridge::installSceneResetBarrier();
    if (g_captureImages)
        g_originalBattleRender = SokuLib::TamperDword(
            &SokuLib::VTable_Battle.onRender, battleOnRender);
    g_originalBattleProcess = SokuLib::TamperDword(
        &SokuLib::VTable_Battle.onProcess, battleOnProcess);
    VirtualProtect(reinterpret_cast<void *>(RDATA_SECTION_OFFSET), RDATA_SECTION_SIZE,
        rdataProtection, &ignored);
    FlushInstructionCache(GetCurrentProcess(), nullptr, 0);
    return g_originalSetInputs && headlessHookInstalled && unlimitedHookInstalled &&
        g_originalBattleManagerProcess && g_originalSelectProcess && g_originalTitleProcess &&
        (!g_captureImages || g_originalBattleRender) &&
        g_originalBattleProcess && resetBarrierInstalled;
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
    g_headlessRender = environmentValue(L"SOKURL_HEADLESS_RENDER", 0) == 1;
    if (environmentValue(L"SOKURL_MUTE_AUDIO", g_headlessRender ? 1U : 0U) == 1 &&
        !SokuRLBridge::installAudioMute()) {
        closeMapping();
        return false;
    }
    const auto captureMode = environmentValue(L"SOKURL_CAPTURE_IMAGES", 0);
    g_captureImages = captureMode != 0;
    g_captureStateOnly = captureMode == 2;
    if (g_captureImages) {
        // Images require the original renderer, even when the window is unattended.
        g_headlessRender = g_captureStateOnly;
        if (!SokuRLBridge::initializeImageCapture(captureMode == 1)) {
            closeMapping();
            return false;
        }
    }
    g_unlimitedPacing = environmentValue(L"SOKURL_UNLIMITED_PACING", 0) == 1;
    g_vsBootstrapArmed = environmentValue(L"SOKURL_VS_BOOTSTRAP", 0) == 1;
    if (g_vsBootstrapArmed) {
        g_vsP1Character = environmentValue(L"SOKURL_VS_P1_CHARACTER", 1);
        g_vsP2Character = environmentValue(L"SOKURL_VS_P2_CHARACTER", 0);
        g_vsP1Palette = environmentValue(L"SOKURL_VS_P1_PALETTE", 0);
        g_vsP2Palette = environmentValue(L"SOKURL_VS_P2_PALETTE", 0);
        g_vsP1Deck = environmentValue(L"SOKURL_VS_P1_DECK", 0);
        g_vsP2Deck = environmentValue(L"SOKURL_VS_P2_DECK", 0);
        g_vsStage = environmentValue(L"SOKURL_VS_STAGE", 0);
        g_vsMusic = environmentValue(L"SOKURL_VS_MUSIC", 0);
        g_vsPauseAtStart = environmentValue(L"SOKURL_VS_PAUSE_AT_START", 0) == 1;
        const auto seed = environmentValue(L"SOKURL_VS_SEED", 0xFFFFFFFFU);
        g_vsSeedRequested = seed != 0xFFFFFFFFU;
        g_vsSeed = seed;
    }
    const auto *commandLine = GetCommandLineW();
    if (commandLine && wcsstr(commandLine, L".rep")) {
        g_checkpointArmed = true;
        g_paused = true;
    }
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
