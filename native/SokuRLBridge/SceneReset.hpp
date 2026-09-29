#pragma once
#include <cstdint>

namespace SokuLib { struct Battle; }

namespace SokuRLBridge {
// Install while the scene vtables are writable.
bool installSceneResetBarrier(bool enabled, std::uint32_t diagnosticDelayMs);
void retireBattleScene(SokuLib::Battle *battle);
bool retiredBattleSceneDestroyed();
}
