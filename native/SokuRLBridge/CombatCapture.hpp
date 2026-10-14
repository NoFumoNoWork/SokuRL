#pragma once
#include "ControlBlock.hpp"
#include <cstdint>
namespace SokuLib { struct BattleManager; }
namespace SokuRLBridge {
bool installCombatCapture();
bool combatCaptureEnabled();
struct CombatActionToken {
    std::uint32_t previous, reserved;
    std::int32_t inputs[8], buffered[8], released[6];
    std::uint32_t inputScope, command, movement;
};
std::uint32_t combatInputBegin(void *);
void combatInputEnd(void *, std::uint32_t);
struct CombatSpawnToken { std::uint32_t root, provenance; };
CombatActionToken combatActionBegin(void *);
void combatActionEnd(void *, unsigned, unsigned, unsigned, bool, CombatActionToken);
CombatSpawnToken combatSpawnBegin(void *, void *);
void combatSpawnEnd(void *, void *, void *, CombatSpawnToken);
void beginCombatFrame(SokuLib::BattleManager *, std::uint64_t, std::uint32_t);
void endCombatFrame();
void closeCombatCapture();
}
