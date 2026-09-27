#include "ControlBlock.hpp"

#include <BattleMode.hpp>
#include <Hash.hpp>
#include <InputManager.hpp>
#include <SokuAddresses.hpp>
#include <Tamper.hpp>
#include <UnionCast.hpp>

#include <Windows.h>
#include <cstring>

namespace
{
using SetInputsMethod = void (SokuLib::KeymapManager::*)();

constexpr DWORD KEYMAP_SET_INPUTS_HOOK = 0x0040A45D;
constexpr DWORD PLAYER_ONE_MANAGER_POINTER = 0x008989A0;
constexpr int LOCAL_BATTLE_SCENE = 5;

HANDLE g_mapping = nullptr;
volatile SokuRLBridge::ControlBlock *g_control = nullptr;
SetInputsMethod g_originalSetInputs = nullptr;

std::uint32_t g_lastCommandSeq = 0;
SokuLib::KeyInput g_activeInput{};
std::uint32_t g_framesRemaining = 0;
bool g_active = false;
bool g_neutralPending = false;

std::uint32_t load32(const volatile std::uint32_t *value)
{
    return static_cast<std::uint32_t>(InterlockedCompareExchange(
        reinterpret_cast<volatile LONG *>(const_cast<volatile std::uint32_t *>(value)), 0, 0));
}

void store32(volatile std::uint32_t *target, std::uint32_t value)
{
    InterlockedExchange(reinterpret_cast<volatile LONG *>(target), static_cast<LONG>(value));
}

void publishResult(SokuRLBridge::ResultCode result)
{
    store32(&g_control->framesRemaining, g_framesRemaining);
    store32(&g_control->resultCode, static_cast<std::uint32_t>(result));
}

void clearControlledInput(SokuRLBridge::ResultCode result, bool injectNeutral)
{
    g_activeInput = {};
    g_framesRemaining = 0;
    g_active = false;
    g_neutralPending = injectNeutral;
    publishResult(result);
}

bool isBoolean(std::uint32_t value)
{
    return value <= 1;
}

bool isValidInput(
    std::int32_t horizontal,
    std::int32_t vertical,
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t c,
    std::uint32_t d,
    std::uint32_t duration)
{
    return horizontal >= -1 && horizontal <= 1 &&
        vertical >= -1 && vertical <= 1 &&
        isBoolean(a) && isBoolean(b) && isBoolean(c) && isBoolean(d) &&
        duration >= 1 && duration <= SokuRLBridge::MAX_DURATION_FRAMES;
}

bool isPracticeGameplay()
{
    const auto scene = *reinterpret_cast<const int *>(SokuLib::ADDR_SCENE_ID);
    return scene == LOCAL_BATTLE_SCENE &&
        SokuLib::mainMode == SokuLib::BATTLE_MODE_PRACTICE;
}

void acknowledge(std::uint32_t sequence)
{
    MemoryBarrier();
    store32(&g_control->ackSeq, sequence);
}

void consumeNewCommand(bool gameplay)
{
    const auto sequence = load32(&g_control->commandSeq);
    if (sequence == g_lastCommandSeq)
        return;

    MemoryBarrier();
    const auto commandType = static_cast<SokuRLBridge::CommandType>(load32(&g_control->commandType));
    const auto horizontal = g_control->horizontalAxis;
    const auto vertical = g_control->verticalAxis;
    const auto a = load32(&g_control->a);
    const auto b = load32(&g_control->b);
    const auto c = load32(&g_control->c);
    const auto d = load32(&g_control->d);
    const auto duration = load32(&g_control->durationFrames);
    g_lastCommandSeq = sequence;

    if (commandType == SokuRLBridge::CommandType::Release) {
        clearControlledInput(SokuRLBridge::ResultCode::Released, gameplay);
        acknowledge(sequence);
        return;
    }

    if (commandType != SokuRLBridge::CommandType::Input ||
        !isValidInput(horizontal, vertical, a, b, c, d, duration)) {
        clearControlledInput(SokuRLBridge::ResultCode::InvalidCommand, gameplay);
        acknowledge(sequence);
        return;
    }

    if (!gameplay) {
        clearControlledInput(SokuRLBridge::ResultCode::NotInGameplay, false);
        acknowledge(sequence);
        return;
    }

    g_activeInput = {
        horizontal,
        vertical,
        static_cast<int>(a),
        static_cast<int>(b),
        static_cast<int>(c),
        static_cast<int>(d),
        0,
        0,
    };
    g_framesRemaining = duration;
    g_active = true;
    g_neutralPending = false;
    publishResult(SokuRLBridge::ResultCode::Accepted);
    acknowledge(sequence);
}

void __fastcall keymapManagerSetInputs(SokuLib::KeymapManager *self)
{
    (self->*g_originalSetInputs)();
    if (!g_control)
        return;

    const auto playerOne = *reinterpret_cast<SokuLib::KeymapManager **>(PLAYER_ONE_MANAGER_POINTER);
    if (self != playerOne)
        return;

    const bool gameplay = isPracticeGameplay();
    store32(&g_control->inGameplay, gameplay ? 1U : 0U);

    if (gameplay) {
        InterlockedIncrement64(reinterpret_cast<volatile LONG64 *>(&g_control->gameFrame));
    } else if (g_active || g_neutralPending) {
        clearControlledInput(SokuRLBridge::ResultCode::NotInGameplay, false);
    }

    consumeNewCommand(gameplay);
    if (!gameplay)
        return;

    if (g_neutralPending) {
        self->input = {};
        g_neutralPending = false;
        return;
    }

    if (!g_active || g_framesRemaining == 0)
        return;

    self->input = g_activeInput;
    --g_framesRemaining;
    store32(&g_control->framesRemaining, g_framesRemaining);
    if (g_framesRemaining == 0) {
        g_active = false;
        g_neutralPending = true;
        store32(&g_control->resultCode, static_cast<std::uint32_t>(SokuRLBridge::ResultCode::Complete));
    }
}

bool createControlBlock()
{
    g_mapping = CreateFileMappingW(
        INVALID_HANDLE_VALUE,
        nullptr,
        PAGE_READWRITE,
        0,
        sizeof(SokuRLBridge::ControlBlock),
        SokuRLBridge::MAPPING_NAME);
    if (!g_mapping)
        return false;

    g_control = static_cast<volatile SokuRLBridge::ControlBlock *>(MapViewOfFile(
        g_mapping,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(SokuRLBridge::ControlBlock)));
    if (!g_control) {
        CloseHandle(g_mapping);
        g_mapping = nullptr;
        return false;
    }

    auto *control = const_cast<SokuRLBridge::ControlBlock *>(g_control);
    std::memset(control, 0, sizeof(*control));
    control->magic = SokuRLBridge::CONTROL_MAGIC;
    control->version = SokuRLBridge::CONTROL_VERSION;
    control->structSize = sizeof(*control);
    control->resultCode = static_cast<std::uint32_t>(SokuRLBridge::ResultCode::Idle);
    store32(&g_control->connected, 1);
    return true;
}

void closeControlBlock()
{
    if (g_control) {
        store32(&g_control->connected, 0);
        UnmapViewOfFile(const_cast<SokuRLBridge::ControlBlock *>(g_control));
        g_control = nullptr;
    }
    if (g_mapping) {
        CloseHandle(g_mapping);
        g_mapping = nullptr;
    }
}

bool installInputHook()
{
    DWORD oldProtection = 0;
    if (!VirtualProtect(
            reinterpret_cast<void *>(TEXT_SECTION_OFFSET),
            TEXT_SECTION_SIZE,
            PAGE_EXECUTE_WRITECOPY,
            &oldProtection))
        return false;

    g_originalSetInputs = SokuLib::union_cast<SetInputsMethod>(
        SokuLib::TamperNearJmpOpr(KEYMAP_SET_INPUTS_HOOK, keymapManagerSetInputs));

    DWORD ignored = 0;
    VirtualProtect(
        reinterpret_cast<void *>(TEXT_SECTION_OFFSET),
        TEXT_SECTION_SIZE,
        oldProtection,
        &ignored);
    FlushInstructionCache(GetCurrentProcess(), nullptr, 0);
    return g_originalSetInputs != nullptr;
}
}

extern "C" __declspec(dllexport) bool CheckVersion(const BYTE hash[16])
{
    return std::memcmp(hash, SokuLib::targetHash, sizeof(SokuLib::targetHash)) == 0;
}

extern "C" __declspec(dllexport) bool Initialize(HMODULE, HMODULE)
{
    static_assert(sizeof(void *) == 4, "SokuRLBridge must be built for Win32/x86");
    if (!createControlBlock())
        return false;
    if (!installInputHook()) {
        closeControlBlock();
        return false;
    }
    return true;
}

BOOL APIENTRY DllMain(HMODULE, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_DETACH)
        closeControlBlock();
    return TRUE;
}
