#include "SceneReset.hpp"
#include <Scenes.hpp>
#include <Tamper.hpp>
#include <VTables.hpp>
#include <Windows.h>

namespace SokuRLBridge {
namespace {
using Destructor = SokuLib::Battle *(SokuLib::Battle::*)(char);
Destructor originalDestructor = nullptr;
PVOID volatile retiredBattle = nullptr;

SokuLib::Battle *__fastcall destroyBattle(SokuLib::Battle *battle, void *, char release)
{
    // The engine runs scene destructors on its cleanup thread. Do not touch the
    // object after calling its original destructor: release may free it.
    auto *result = (battle->*originalDestructor)(release);
    InterlockedCompareExchangePointer(&retiredBattle, nullptr, battle);
    return result;
}
}

bool installSceneResetBarrier()
{
    originalDestructor = SokuLib::TamperDword(&SokuLib::VTable_Battle.destructor, destroyBattle);
    return originalDestructor != nullptr;
}

void retireBattleScene(SokuLib::Battle *battle)
{
    if (originalDestructor)
        InterlockedExchangePointer(&retiredBattle, battle);
}

bool retiredBattleSceneDestroyed()
{
    return InterlockedCompareExchangePointer(&retiredBattle, nullptr, nullptr) == nullptr;
}
}
