#include "CombatCapture.hpp"
#include <BattleManager.hpp>
#include <Tamper.hpp>
#include <intrin.h>
#include <cstring>
#include <cstdio>

namespace {
constexpr unsigned CAPACITY = 2048;
constexpr unsigned WORDS = 52;
struct Event { std::uint32_t value[WORDS]; };
struct Header {
    std::uint32_t magic, version, eventSize, capacity;
    volatile LONG write, read, dropped;
    std::uint32_t reserved;
};
struct Mapping { Header header; Event events[CAPACITY]; };
static_assert(sizeof(Header) == 32 && sizeof(Event) == 208, "combat ABI");
Mapping *g_map = nullptr;
HANDLE g_handle = nullptr;
void *g_players[2]{};
std::uint64_t g_frame = 0;
std::uint32_t g_segment = 0, g_order = 0;
bool g_active = false;
Event *g_contact = nullptr;
std::uint32_t g_generation[2]{}, g_nextGeneration=0, g_nextObject=0;
unsigned g_spawnDepth[2]{};
std::uint32_t g_inputScope[2]{}, g_nextInputScope=0;
struct ObjectIdentity { void *address; std::uint32_t serial, root, provenance; };
ObjectIdentity g_objects[2048]{};
ObjectIdentity *identity(void *address, bool create=false) {
    const auto key=(reinterpret_cast<std::uint32_t>(address)>>4)&2047;
    for (unsigned i=0;i<2048;++i) {
        auto &item=g_objects[(key+i)&2047];
        if (item.address==address) return &item;
        if (!item.address) { if (create) { item.address=address; return &item; } return nullptr; }
    }
    if (create && g_map) InterlockedIncrement(&g_map->header.dropped);
    return nullptr;
}
using Resolve = std::uint32_t (__thiscall *)(void *, void *, void *);
using Hit = void (__thiscall *)(void *, void *, void *);

template<typename T> T load(const void *base, unsigned offset) {
    T result;
    std::memcpy(&result, static_cast<const unsigned char *>(base)+offset, sizeof(result));
    return result;
}
int seat(void *player) {
    return player == g_players[0] ? 0 : player == g_players[1] ? 1 : -1;
}
void publish(const Event &event) {
    auto &h = g_map->header;
    const auto write = static_cast<std::uint32_t>(h.write);
    const auto read = static_cast<std::uint32_t>(h.read);
    if (write-read >= CAPACITY) { InterlockedIncrement(&h.dropped); return; }
    g_map->events[write%CAPACITY] = event;
    InterlockedExchange(&h.write, static_cast<LONG>(write+1));
}
bool before(Event &e, unsigned kind, void *source, void *defender) {
    if (!g_active || !g_map) return false;
    void *owner = load<void *>(source, 0x16C);
    if (seat(owner) < 0 || seat(defender) < 0) return false;
    e.value[0] = static_cast<std::uint32_t>(g_frame);
    e.value[1] = static_cast<std::uint32_t>(g_frame >> 32);
    e.value[2] = g_segment;
    e.value[3] = ++g_order;
    e.value[4] = kind;
    e.value[5] = static_cast<unsigned>(seat(owner));
    e.value[6] = static_cast<unsigned>(seat(defender));
    e.value[7] = reinterpret_cast<std::uint32_t>(source);
    e.value[8] = load<unsigned short>(source, 0x13C);
    e.value[9] = load<unsigned short>(source, 0x13E);
    e.value[10] = load<unsigned short>(defender, 0x13C);
    e.value[11] = load<unsigned short>(defender, 0x13E);
    void *pose = load<void *>(defender, 0x158);
    e.value[12] = pose ? load<std::uint32_t>(pose, 0x4C) : 0;
    e.value[13] = static_cast<std::uint32_t>(load<short>(defender, 0x184));
    e.value[15] = static_cast<std::uint32_t>(load<short>(defender, 0x49E));
    e.value[17] = load<std::uint32_t>(owner, 0x4B0);
    e.value[18] = 0xFFFFFFFF; // absent, not an invented Rate
    e.value[34] = 0xFFFFFFFF; // correction flags at the damage calculation
    e.value[20] = load<unsigned short>(owner, 0x4B4);
    e.value[22] = load<unsigned short>(owner, 0x4B8);
    e.value[24] = load<unsigned char>(owner, 0x4AD);
    if (source==owner) {
        e.value[30]=static_cast<unsigned>(seat(owner))+1;
        e.value[31]=g_generation[seat(owner)]; e.value[32]=1;
    } else if (auto *id=identity(source)) {
        e.value[30]=id->serial; e.value[31]=id->root; e.value[32]=id->provenance;
    }
    return true;
}
void after(Event &e, void *source, void *defender, unsigned result) {
    void *owner = load<void *>(source, 0x16C);
    e.value[14] = static_cast<std::uint32_t>(load<short>(defender, 0x184));
    e.value[16] = static_cast<std::uint32_t>(load<short>(defender, 0x49E));
    e.value[19] = load<std::uint32_t>(owner, 0x4B0);
    e.value[21] = load<unsigned short>(owner, 0x4B4);
    e.value[23] = load<unsigned short>(owner, 0x4B8);
    e.value[25] = load<unsigned char>(owner, 0x4AD);
    e.value[26] = load<unsigned short>(defender, 0x13C);
    e.value[27] = result;
    e.value[28] = load<std::uint32_t>(source, 0x190);
    publish(e);
}
std::uint32_t resolve(unsigned kind, std::uint32_t address, void *manager, void *source, void *defender) {
    Event e{};
    // Capturing reads only integer bits; avoid introducing x87 arithmetic.
    const bool observing = before(e, kind, source, defender);
    Event *previous = g_contact;
    if (observing) g_contact = &e;
    unsigned result = 0;
    if (kind == 1) reinterpret_cast<Hit>(address)(manager, source, defender);
    else result = reinterpret_cast<Resolve>(address)(manager, source, defender);
    g_contact = previous;
    if (observing) after(e, source, defender, result);
    return result;
}
void __fastcall hit(void *manager, void *, void *source, void *defender) {
    resolve(1, 0x47AA50, manager, source, defender);
}
unsigned __fastcall blockHigh(void *manager, void *, void *source, void *defender) {
    return resolve(2, 0x47BF90, manager, source, defender);
}
unsigned __fastcall blockLow(void *manager, void *, void *source, void *defender) {
    return resolve(3, 0x47C2E0, manager, source, defender);
}
void __cdecl captureRate(void *source) {
    if (g_contact && g_contact->value[7] == reinterpret_cast<std::uint32_t>(source)) {
        void *owner = load<void *>(source, 0x16C);
        const auto observed=load<std::uint32_t>(owner, 0x4B0);
        if (!g_contact->value[29]) g_contact->value[18]=observed;
        else if (g_contact->value[18]!=observed) g_contact->value[18]=0xFFFFFFFF;
        g_contact->value[33]=load<unsigned short>(owner,0x4B4);
        const auto flags=load<unsigned char>(owner,0x4AD);
        if (!g_contact->value[29]) g_contact->value[34]=flags;
        else if (g_contact->value[34]!=flags) g_contact->value[34]=0xFFFFFFFF;
        ++g_contact->value[29];
    }
}
unsigned g_factor = 0x463F50;
// Preserve entry GPR/EFLAGS, x87 and SIMD before tail-calling the original
// factor function. Its ST0 return is produced by original code, not C++.
__declspec(naked) void factor() {
    __asm {
        pushfd
        pushad
        sub esp, 528
        mov ebx, esp
        add ebx, 15
        and ebx, -16
        fxsave [ebx]
        push ecx
        call captureRate
        add esp, 4
        fxrstor [ebx]
        add esp, 528
        popad
        popfd
        jmp dword ptr [g_factor]
    }
}
struct Site { unsigned address, target; void *replacement; };
}
namespace SokuRLBridge {
bool installCombatCapture() {
    wchar_t enabled[8]{};
    if (!GetEnvironmentVariableW(L"SOKURL_COMBAT_EVENTS", enabled, 8) || enabled[0] != L'1') return true;
    Site sites[] = {
        {0x47C84D,0x47AA50,reinterpret_cast<void *>(hit)},
        {0x47C5C6,0x47BF90,reinterpret_cast<void *>(blockHigh)},
        {0x47C619,0x47BF90,reinterpret_cast<void *>(blockHigh)},
        {0x47C66C,0x47BF90,reinterpret_cast<void *>(blockHigh)},
        {0x47C711,0x47BF90,reinterpret_cast<void *>(blockHigh)},
        {0x47C6AE,0x47C2E0,reinterpret_cast<void *>(blockLow)},
        {0x47C6D6,0x47C2E0,reinterpret_cast<void *>(blockLow)},
        {0x464A84,0x463F50,reinterpret_cast<void *>(factor)},
    };
    // Validate every original CALL before touching any one of them.
    for (const auto &s : sites) {
        const auto *p = reinterpret_cast<const unsigned char *>(s.address);
        if (p[0] != 0xE8 || s.address+5+load<int>(const_cast<unsigned char *>(p),1) != s.target) return false;
    }
    wchar_t name[80];
    swprintf_s(name, L"Local\\SokuRLCombat_%lu", GetCurrentProcessId());
    g_handle = CreateFileMappingW(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,0,sizeof(Mapping),name);
    if (!g_handle || GetLastError() == ERROR_ALREADY_EXISTS) return false;
    g_map = static_cast<Mapping *>(MapViewOfFile(g_handle,FILE_MAP_ALL_ACCESS,0,0,sizeof(Mapping)));
    if (!g_map) return false;
    std::memset(g_map,0,sizeof(Mapping));
    g_map->header.magic=0x31564543; g_map->header.version=8;
    wchar_t counters[8]{};
    g_map->header.reserved=1U | ((GetEnvironmentVariableW(L"SOKURL_NATIVE_INPUT_COUNTERS",counters,8)
        && counters[0]==L'1') ? 2U : 0U);
    g_map->header.eventSize=sizeof(Event); g_map->header.capacity=CAPACITY;
    for (const auto &s : sites) SokuLib::TamperNearJmpOpr(s.address,s.replacement);
    return true;
}
void beginCombatFrame(SokuLib::BattleManager *manager, std::uint64_t frame, std::uint32_t segment) {
    if (!g_map) return;
    if (segment!=g_segment || frame<=g_frame) {
        std::memset(g_objects,0,sizeof(g_objects));
        g_generation[0]=g_generation[1]=0; g_nextGeneration=0; g_nextObject=2;
        g_spawnDepth[0]=g_spawnDepth[1]=0;
        g_inputScope[0]=g_inputScope[1]=0; g_nextInputScope=0;
    }
    g_players[0]=&manager->leftCharacterManager; g_players[1]=&manager->rightCharacterManager;
    g_frame=frame; g_segment=segment; g_order=0; g_active=true;
}
void endCombatFrame() {
    if (g_active && g_map) {
        for (unsigned index=0;index<2;++index) {
            auto *subject=g_players[index];
            Event e{};
            e.value[0]=static_cast<unsigned>(g_frame); e.value[1]=static_cast<unsigned>(g_frame>>32);
            e.value[2]=g_segment; e.value[3]=++g_order; e.value[4]=7;
            e.value[5]=e.value[6]=index;
            e.value[19]=load<unsigned>(subject,0x754); e.value[20]=load<unsigned>(subject,0x758);
            for (unsigned i=0;i<6;++i) e.value[13+i]=load<unsigned>(subject,0x75C+4*i);
            for (unsigned i=0;i<8;++i) e.value[36+i]=load<unsigned>(subject,0x774+4*i);
            publish(e);
        }
    }
    g_active=false;
}
bool combatCaptureEnabled() { return g_map!=nullptr; }
std::uint32_t combatInputBegin(void *subject) {
    const int index=seat(subject);
    if (!g_active || !g_map || index<0) return 0;
    const auto previous=g_inputScope[index];
    g_inputScope[index]=++g_nextInputScope;
    Event e{};
    e.value[0]=static_cast<unsigned>(g_frame); e.value[1]=static_cast<unsigned>(g_frame>>32);
    e.value[2]=g_segment; e.value[3]=++g_order; e.value[4]=6;
    e.value[5]=e.value[6]=index; e.value[35]=g_inputScope[index];
    const auto *keys=static_cast<unsigned char *>(subject)+0x754;
    e.value[19]=load<unsigned>(keys,0); e.value[20]=load<unsigned>(keys,4);
    for (unsigned i=0;i<6;++i) e.value[13+i]=load<unsigned>(keys,8+4*i);
    for (unsigned i=0;i<8;++i) e.value[36+i]=load<unsigned>(subject,0x774+4*i);
    publish(e);
    return previous;
}
void combatInputEnd(void *subject, std::uint32_t previous) {
    const int index=seat(subject);
    if (index>=0) g_inputScope[index]=previous;
}
CombatActionToken combatActionBegin(void *subject) {
    const int index=seat(subject);
    if (!g_active || !g_map || index<0) return {0,0,{}};
    CombatActionToken token{g_generation[index],++g_nextGeneration,{}};
    const auto &keys=static_cast<SokuLib::CharacterManager *>(subject)->keyMap;
    static_assert(sizeof(keys)==sizeof(token.inputs),"native key counter layout");
    std::memcpy(token.inputs,&keys,sizeof(token.inputs));
    std::memcpy(token.buffered,static_cast<unsigned char *>(subject)+0x774,sizeof(token.buffered));
    std::memcpy(token.released,static_cast<unsigned char *>(subject)+0x794,sizeof(token.released));
    token.command=load<unsigned>(subject,0x7C8);
    token.movement=load<unsigned>(subject,0x7C4);
    token.inputScope=g_inputScope[index];
    g_generation[index]=token.reserved;
    return token;
}
void combatActionEnd(void *subject, unsigned caller, unsigned oldAction, unsigned newAction,
                     bool accepted, CombatActionToken token) {
    const int index=seat(subject);
    if (!token.reserved || index<0) return;
    if (!accepted && g_generation[index]==token.reserved) g_generation[index]=token.previous;
    Event e{};
    e.value[0]=static_cast<unsigned>(g_frame); e.value[1]=static_cast<unsigned>(g_frame>>32);
    e.value[2]=g_segment; e.value[3]=++g_order; e.value[4]=4;
    e.value[5]=e.value[6]=index; e.value[7]=reinterpret_cast<unsigned>(subject);
    e.value[8]=newAction; e.value[10]=oldAction; e.value[26]=newAction;
    e.value[27]=accepted?1:0; e.value[30]=index+1; e.value[31]=token.reserved;
    e.value[19]=token.inputs[0]; e.value[20]=token.inputs[1];
    for (unsigned i=0;i<6;++i) e.value[13+i]=token.inputs[2+i];
    e.value[32]=(g_generation[index]==token.reserved && accepted)?1:0;
    e.value[33]=caller;
    e.value[35]=token.inputScope;
    for (unsigned i=0;i<8;++i) e.value[36+i]=token.buffered[i];
    for (unsigned i=0;i<6;++i) e.value[44+i]=token.released[i];
    e.value[50]=token.command; e.value[51]=token.movement;
    publish(e);
}
CombatSpawnToken combatSpawnBegin(void *parent, void *owner) {
    if (!g_active || !g_map || seat(owner)<0) return {0,0};
    const bool nested=g_spawnDepth[seat(owner)]++ != 0;
    if (!parent) return {g_generation[seat(owner)],1};
    if (nested) return {0,0}; // parent may be reusing a still-stale heap address
    if (auto *id=identity(parent)) return {id->root,id->provenance?2U:0U};
    // A child born during its parent's initializer precedes parent registration.
    // Keep this dependency unknown instead of attributing it to current action.
    return {0,0};
}
void combatSpawnEnd(void *object, void *parent, void *owner, CombatSpawnToken token) {
    if (!g_active || !g_map || seat(owner)<0) return;
    if (g_spawnDepth[seat(owner)]) --g_spawnDepth[seat(owner)];
    if (!object) return;
    auto *id=identity(object,true);
    if (!id) return;
    id->serial=++g_nextObject; id->root=token.root; id->provenance=token.provenance;
    Event e{};
    e.value[0]=static_cast<unsigned>(g_frame); e.value[1]=static_cast<unsigned>(g_frame>>32);
    e.value[2]=g_segment; e.value[3]=++g_order; e.value[4]=5;
    e.value[5]=e.value[6]=seat(owner); e.value[7]=reinterpret_cast<unsigned>(object);
    e.value[8]=load<unsigned short>(object,0x13C); e.value[9]=load<unsigned short>(object,0x13E);
    e.value[30]=id->serial; e.value[31]=id->root; e.value[32]=id->provenance;
    e.value[33]=reinterpret_cast<unsigned>(parent);
    publish(e);
}
void closeCombatCapture() {
    if (g_map) UnmapViewOfFile(g_map);
    if (g_handle) CloseHandle(g_handle);
    g_map=nullptr; g_handle=nullptr;
}
}
