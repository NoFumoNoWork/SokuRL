#pragma once

namespace SokuLib { struct Battle; }

namespace SokuRLBridge {
// Install while the scene vtables are writable.
bool installSceneResetBarrier();
void retireBattleScene(SokuLib::Battle *battle);
bool retiredBattleSceneDestroyed();
}
