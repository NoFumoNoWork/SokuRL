#include "FreshReset.hpp"
#include <Windows.h>
#include <Character.hpp>
#include <Tamper.hpp>
#include <UnionCast.hpp>
#include <VTables.hpp>
#include <CharacterManager.hpp>
#include <mutex>

namespace SokuRLBridge {
namespace {
using Create = void (__thiscall *)(void *, int, void *, int);
Create originalCreate = nullptr;
bool enabled = false;
thread_local bool reuseCharacters = false;
struct Settings {
    int character = -1;
    unsigned char side = 0, palette = 0, deck = 0;
    SokuLib::KeyManager **keys = nullptr;
};
Settings settings[2];
std::mutex settingsMutex;
Settings readSettings(const SokuLib::PlayerInfo &info)
{
    return {int(info.character), info.isRight, info.palette, info.deck, info.keyManager};
}
bool matches(const Settings &a, const Settings &b)
{
    return a.character == b.character && a.side == b.side && a.palette == b.palette &&
        a.deck == b.deck && a.keys == b.keys;
}
void __fastcall create(void *manager, void *, int index, void *info, int flags)
{
    if (!reuseCharacters) {
        originalCreate(manager, index, info, flags);
        if (index >= 0 && index < 2) {
            std::lock_guard<std::mutex> lock(settingsMutex);
            settings[index] = readSettings(*static_cast<SokuLib::PlayerInfo *>(info));
        }
    }
    // Original onArenaStart next invokes 479520, including each original
    // character's virtual reset and object-list cleanup. No frame snapshot.
}
}

bool freshResetEnabled() { return enabled; }

bool installFreshReset()
{
    wchar_t value[8]{};
    enabled = GetEnvironmentVariableW(L"SOKURL_FRESH_RESET", value, 8) == 1 && value[0] == L'1';
    if (!enabled) return true;
    if (GetEnvironmentVariableW(L"SOKURL_RESET_TIMING_DIR", nullptr, 0)) return false;
    const DWORD sites[] = {0x4818D4, 0x4818E7};
    for (auto site : sites) {
        auto *code = reinterpret_cast<const unsigned char *>(site);
        if (*code != 0xE8 || site + 5 + *reinterpret_cast<const DWORD *>(code + 1) != 0x46EAD0)
            return false;
    }
    for (auto site : sites) {
        DWORD protection = 0;
        if (!VirtualProtect(reinterpret_cast<void *>(site), 5, PAGE_EXECUTE_WRITECOPY, &protection)) return false;
        originalCreate = reinterpret_cast<Create>(SokuLib::TamperNearCall(site, create));
        DWORD ignored = 0;
        VirtualProtect(reinterpret_cast<void *>(site), 5, protection, &ignored);
    }
    return FlushInstructionCache(GetCurrentProcess(), nullptr, 0) != FALSE;
}

bool canStartFreshWorld(SokuLib::BattleManager *manager)
{
    std::lock_guard<std::mutex> lock(settingsMutex);
    return enabled && !reuseCharacters && manager &&
        unsigned(SokuLib::leftPlayerInfo.character) <= 1 && unsigned(SokuLib::rightPlayerInfo.character) <= 1 &&
        matches(settings[0], readSettings(SokuLib::leftPlayerInfo)) &&
        matches(settings[1], readSettings(SokuLib::rightPlayerInfo));
}

bool startFreshWorld(SokuLib::BattleManager *manager)
{
    if (!canStartFreshWorld(manager)) return false;
    (manager->*SokuLib::VTable_BattleManager.onExit)();
    // Constructor 46B891/46B897 initializes these counts to zero; the native
    // virtual reset does not clear them. Keep resource/box ownership intact.
    manager->leftCharacterManager.objectBase.hitBoxCount = 0;
    manager->leftCharacterManager.objectBase.hurtBoxCount = 0;
    manager->rightCharacterManager.objectBase.hitBoxCount = 0;
    manager->rightCharacterManager.objectBase.hurtBoxCount = 0;
    reuseCharacters = true;
    (manager->*SokuLib::VTable_BattleManager.onArenaStart)(&SokuLib::gameParams);
    reuseCharacters = false;
    return true;
}
}
