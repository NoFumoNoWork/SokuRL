#include "SceneReset.hpp"
#include "CrashReport.hpp"
#include <Scenes.hpp>
#include <Tamper.hpp>
#include <VTables.hpp>
#include <Windows.h>

namespace SokuRLBridge {
namespace {
using Destructor = SokuLib::Battle *(SokuLib::Battle::*)(char);
Destructor originalDestructor = nullptr;
PVOID volatile retiredBattle = nullptr;
bool waitForDestructor = false;
DWORD diagnosticDelay = 0;

SokuLib::Battle *__fastcall destroyBattle(SokuLib::Battle *battle, void *, char release)
{
    // The engine runs scene destructors on its cleanup thread. Do not touch the
    // object after calling its original destructor: release may free it.
    traceResetStage("battle_destructor_begin", battle);
    if (diagnosticDelay)
        Sleep(diagnosticDelay);
    auto *result = (battle->*originalDestructor)(release);
    traceResetStage("battle_destructor_end", battle);
    InterlockedCompareExchangePointer(&retiredBattle, nullptr, battle);
    return result;
}
}

bool installSceneResetBarrier(bool enabled, std::uint32_t diagnosticDelayMs)
{
    if (diagnosticDelayMs > 1000)
        return false;
    waitForDestructor = enabled;
    diagnosticDelay = diagnosticDelayMs;
    traceResetStage(enabled ? "barrier_enabled" : "barrier_disabled", nullptr);
    originalDestructor = SokuLib::TamperDword(&SokuLib::VTable_Battle.destructor, destroyBattle);
    return originalDestructor != nullptr;
}

void retireBattleScene(SokuLib::Battle *battle)
{
    if (waitForDestructor)
        InterlockedExchangePointer(&retiredBattle, battle);
}

bool retiredBattleSceneDestroyed()
{
    return InterlockedCompareExchangePointer(&retiredBattle, nullptr, nullptr) == nullptr;
}
}
