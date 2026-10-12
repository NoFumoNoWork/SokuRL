#include "ResetTiming.hpp"
#include <Windows.h>
#include <cstdio>
#include <cstring>
#include <UnionCast.hpp>
#include <Tamper.hpp>
#include <VTables.hpp>

namespace SokuRLBridge {
namespace {
using Function = void (__cdecl *)();
Function originalLoad = nullptr;
Function originalPrepare = nullptr;
Function originalBattleLoad = nullptr;
using ArenaMethod = void (SokuLib::BattleManager::*)(void *);
ArenaMethod originalArena = nullptr;
using Music = void (__cdecl *)(const char *);
using Character = void (__thiscall *)(void *, int, void *, int);
Music originalMusic = nullptr;
Character originalCharacter = nullptr;
wchar_t logPath[32768]{};
void record(const char *phase, LARGE_INTEGER start)
{
    LARGE_INTEGER end{}, frequency{};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&end);
    FILE *file = nullptr;
    if (_wfopen_s(&file, logPath, L"a") == 0 && file) {
        fprintf(file, "%s,%lu,%.9f\n", phase, GetCurrentThreadId(),
            double(end.QuadPart - start.QuadPart) / frequency.QuadPart);
        fclose(file);
    }
}
void __cdecl music(const char *path)
{
    LARGE_INTEGER start{};
    QueryPerformanceCounter(&start);
    originalMusic(path);
    record("music_43FE10", start);
}
void __fastcall character(void *manager, void *, int index, void *info, int flags)
{
    auto fingerprint = []() {
        unsigned long long hash = 14695981039346656037ULL;
        const auto *state = reinterpret_cast<const unsigned char *>(0x89B660);
        for (unsigned i = 0; i < 624 * 4; ++i) hash = (hash ^ state[i]) * 1099511628211ULL;
        return (hash ^ *reinterpret_cast<const unsigned *>(0x896B20)) * 1099511628211ULL;
    };
    const auto before = fingerprint();
    LARGE_INTEGER start{};
    QueryPerformanceCounter(&start);
    originalCharacter(manager, index, info, flags);
    const auto after = fingerprint();
    record(index ? "character_p2_46EAD0" : "character_p1_46EAD0", start);
    FILE *file = nullptr;
    if (_wfopen_s(&file, logPath, L"a") == 0 && file) {
        fprintf(file, "constructor_rng,%d,%llu,%llu\n", index, before, after);
        fclose(file);
    }
}

void timed(Function original, const char *phase)
{
    LARGE_INTEGER start{}, end{}, frequency{};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    original();
    QueryPerformanceCounter(&end);
    FILE *file = nullptr;
    if (_wfopen_s(&file, logPath, L"a") == 0 && file) {
        fprintf(file, "%s,%lu,%.9f\n", phase, GetCurrentThreadId(),
            double(end.QuadPart - start.QuadPart) / frequency.QuadPart);
        fclose(file);
    }
}

void __cdecl load() { timed(originalLoad, "load_43DE70"); }
void __cdecl prepare() { timed(originalPrepare, "prepare_43C660"); }
void __cdecl battleLoad() { timed(originalBattleLoad, "battle_load_4386F0"); }
void __fastcall arena(SokuLib::BattleManager *manager, void *, void *parameters)
{
    LARGE_INTEGER start{}, end{}, frequency{};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    (manager->*originalArena)(parameters);
    QueryPerformanceCounter(&end);
    FILE *file = nullptr;
    if (_wfopen_s(&file, logPath, L"a") == 0 && file) {
        fprintf(file, "arena_481810,%lu,%.9f\n", GetCurrentThreadId(),
            double(end.QuadPart - start.QuadPart) / frequency.QuadPart);
        fclose(file);
    }
}
}

bool installResetTiming()
{
    wchar_t directory[32000]{};
    const auto length = GetEnvironmentVariableW(L"SOKURL_RESET_TIMING_DIR", directory, 32000);
    if (!length) return true;
    if (length >= 32000) return false;
    swprintf_s(logPath, L"%s\\timing_%lu.csv", directory, GetCurrentProcessId());
    const DWORD extraSites[] = {0x4818BE, 0x4818D4, 0x4818E7};
    const DWORD extraTargets[] = {0x43FE10, 0x46EAD0, 0x46EAD0};
    for (unsigned i = 0; i < 3; ++i) {
        auto *code = reinterpret_cast<const unsigned char *>(extraSites[i]);
        if (*code != 0xE8 || extraSites[i] + 5 + *reinterpret_cast<const DWORD *>(code + 1) != extraTargets[i])
            return false;
    }
    for (unsigned i = 0; i < 3; ++i) {
        DWORD protection = 0;
        if (!VirtualProtect(reinterpret_cast<void *>(extraSites[i]), 5, PAGE_EXECUTE_WRITECOPY, &protection)) return false;
        if (i == 0) originalMusic = SokuLib::TamperNearCall(extraSites[i], music);
        else originalCharacter = reinterpret_cast<Character>(SokuLib::TamperNearCall(extraSites[i], character));
        DWORD ignored = 0;
        VirtualProtect(reinterpret_cast<void *>(extraSites[i]), 5, protection, &ignored);
    }
    const DWORD sites[] = {0x0043E623, 0x0043DE70, 0x0043E664};
    const DWORD targets[] = {0x0043DE70, 0x0043C660, 0x004386F0};
    Function hooks[] = {load, prepare, battleLoad};
    Function *originals[] = {&originalLoad, &originalPrepare, &originalBattleLoad};
    for (unsigned i = 0; i < 3; ++i) {
        auto *code = reinterpret_cast<const unsigned char *>(sites[i]);
        if (*code != 0xE8 || sites[i] + 5 + *reinterpret_cast<const DWORD *>(code + 1) != targets[i])
            return false;
    }
    DWORD protection = 0;
    auto *slot = &SokuLib::VTable_BattleManager.onArenaStart;
    if (!VirtualProtect(slot, sizeof(DWORD), PAGE_EXECUTE_WRITECOPY, &protection)) return false;
    originalArena = SokuLib::TamperDword(slot, arena);
    DWORD ignored = 0;
    VirtualProtect(slot, sizeof(DWORD), protection, &ignored);
    for (unsigned i = 0; i < 3; ++i) {
        DWORD protection = 0;
        if (!VirtualProtect(reinterpret_cast<void *>(sites[i]), 5, PAGE_EXECUTE_WRITECOPY, &protection))
            return false;
        *originals[i] = SokuLib::TamperNearCall(sites[i], hooks[i]);
        DWORD ignored = 0;
        VirtualProtect(reinterpret_cast<void *>(sites[i]), 5, protection, &ignored);
    }
    return FlushInstructionCache(GetCurrentProcess(), nullptr, 0) != FALSE;
}
}
