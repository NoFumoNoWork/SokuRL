#pragma once
#include <BattleManager.hpp>
namespace SokuRLBridge {
bool installFreshReset();
bool freshResetEnabled();
bool canStartFreshWorld(SokuLib::BattleManager *manager);
bool startFreshWorld(SokuLib::BattleManager *manager);
}
