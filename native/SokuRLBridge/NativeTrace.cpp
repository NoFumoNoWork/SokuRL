#include "NativeTrace.hpp"
#include "CombatCapture.hpp"

#include <BattleManager.hpp>
#include <CharacterManager.hpp>
#include <Tamper.hpp>
#include <intrin.h>

#include <algorithm>
#include <cstring>

namespace
{
constexpr std::uint32_t PLAYER_VTABLES[] = {0x0085BF4C, 0x0085C37C};
constexpr std::uint32_t OBJECT_VTABLES[] = {0x0085BEBC, 0x0085C2EC};
constexpr std::uint32_t OBJECT_LIST_VTABLES[] = {0x0085BF1C, 0x0085C34C};
constexpr std::size_t SET_ACTION_SLOT = 2;
constexpr std::size_t SET_SEQUENCE_SLOT = 3;
constexpr std::size_t LIST_CREATE_OBJECT_SLOT = 1;
constexpr std::uint32_t FLAG_DIRECT_SPAWN = 1;
constexpr std::uint32_t FLAG_CHILD_SPAWN = 2;

using SetActionFunction = bool (__thiscall *)(void *, short);
using SetSequenceFunction = void (__thiscall *)(void *, short);
using HandleInputsFunction = void (__thiscall *)(void *);
using CreateObjectFunction = void *(__thiscall *)(
    void *, void *, SokuLib::CharacterManager *, short, float, float,
    char, char, void *, unsigned int);

SokuRLBridge::BridgeMapping *g_mapping = nullptr;
const std::uint64_t *g_currentFrame = nullptr;
SokuRLBridge::TraceSink g_sink = nullptr;
SetActionFunction g_playerSetAction[2]{};
SetSequenceFunction g_playerSetSequence[2]{};
SetActionFunction g_objectSetAction[2]{};
SetSequenceFunction g_objectSetSequence[2]{};
CreateObjectFunction g_createObject[2]{};
HandleInputsFunction g_handleInputs[2]{};

std::uint32_t bits(float value)
{
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

__forceinline std::uint32_t returnAddress()
{
    return reinterpret_cast<std::uint32_t>(_ReturnAddress());
}

bool eventKindEnabled(SokuRLBridge::TraceEventKind kind)
{
    const auto mask = g_mapping->control.traceConfig.eventMask;
    const auto bit = 1U << (static_cast<std::uint32_t>(kind) - 1U);
    return !mask || (mask & bit) != 0;
}

bool actionFiltersMatch(void *subject, SokuRLBridge::TraceSubjectKind subjectKind)
{
    if (!g_mapping || !g_mapping->control.traceActive)
        return false;
    const auto &config = g_mapping->control.traceConfig;
    auto &battle = SokuLib::getBattleMgr();
    if (config.p1Action != SokuRLBridge::TRACE_ANY &&
        static_cast<std::uint32_t>(battle.leftCharacterManager.objectBase.action) != config.p1Action)
        return false;
    if (config.p2Action != SokuRLBridge::TRACE_ANY &&
        static_cast<std::uint32_t>(battle.rightCharacterManager.objectBase.action) != config.p2Action)
        return false;
    if (subjectKind == SokuRLBridge::TraceSubjectKind::Object &&
        config.objectAction != SokuRLBridge::TRACE_ANY) {
        const auto *object = static_cast<const SokuLib::ObjectManager *>(subject);
        if (static_cast<std::uint32_t>(object->action) != config.objectAction)
            return false;
    }
    return true;
}

void emitCall(std::uint32_t functionAddress, std::uint32_t ret, void *subject,
    SokuRLBridge::TraceSubjectKind subjectKind, std::uint32_t flags,
    const std::uint32_t (&args)[16])
{
    if (!g_sink || !eventKindEnabled(SokuRLBridge::TraceEventKind::FunctionCall) ||
        !actionFiltersMatch(subject, subjectKind))
        return;
    SokuRLBridge::TraceEvent event{};
    // NativeTrace is installed with &g_currentFrame. During the update that
    // produces captured frame N, g_currentFrame is still N-1 (it is incremented
    // only after the simulation update returns). Align the trace frame id with
    // the RawFrameState frame that will expose this transition.
    event.frameId = *g_currentFrame + 1;
    event.kind = static_cast<std::uint32_t>(SokuRLBridge::TraceEventKind::FunctionCall);
    event.flags = flags;
    // returnAddress is exact; the instruction/caller estimate below is NOT exact.
    // vtable/indirect CALL encodings are not uniformly five bytes, so treat these
    // fields as heuristic locality hints rather than instruction-level proof.
    event.instruction = ret >= 5 ? ret - 5 : 0;
    event.caller = event.instruction;
    event.returnAddress = ret;
    event.functionAddress = functionAddress;
    event.subject = reinterpret_cast<std::uint32_t>(subject);
    event.subjectKind = static_cast<std::uint32_t>(subjectKind);
    std::copy(std::begin(args), std::end(args), std::begin(event.args));
    g_sink(event);
}

template<std::size_t Index, bool Player>
bool __fastcall setActionHook(void *self, void *, short action)
{
    const auto ret = returnAddress();
    const auto oldAction = static_cast<std::uint32_t>(
        static_cast<SokuLib::ObjectManager *>(self)->action);
    auto function = Player ? g_playerSetAction[Index] : g_objectSetAction[Index];
    const auto token = Player ? SokuRLBridge::combatActionBegin(self) : SokuRLBridge::CombatActionToken{0,0};
    const auto result = function(self, action);
    const auto newAction = static_cast<std::uint32_t>(
        static_cast<SokuLib::ObjectManager *>(self)->action);
    if (Player) SokuRLBridge::combatActionEnd(self,ret,oldAction,newAction,result,token);
    const std::uint32_t args[16] = {
        oldAction, static_cast<std::uint16_t>(action), newAction,
        result ? 1U : 0U,
    };
    emitCall(reinterpret_cast<std::uint32_t>(function), ret, self,
        Player ? SokuRLBridge::TraceSubjectKind::Player : SokuRLBridge::TraceSubjectKind::Object,
        0, args);
    return result;
}

template<std::size_t Index, bool Player>
void __fastcall setSequenceHook(void *self, void *, short sequence)
{
    const auto ret = returnAddress();
    auto *object = static_cast<SokuLib::ObjectManager *>(self);
    const auto oldSequence = static_cast<std::uint32_t>(object->actionBlockId);
    auto function = Player ? g_playerSetSequence[Index] : g_objectSetSequence[Index];
    function(self, sequence);
    const std::uint32_t args[16] = {
        oldSequence, static_cast<std::uint16_t>(sequence),
        static_cast<std::uint32_t>(object->actionBlockId),
    };
    emitCall(reinterpret_cast<std::uint32_t>(function), ret, self,
        Player ? SokuRLBridge::TraceSubjectKind::Player : SokuRLBridge::TraceSubjectKind::Object,
        0, args);
}

template<std::size_t Index>
void *__fastcall createObjectHook(void *self, void *, void *parent,
    SokuLib::CharacterManager *owner, short action, float x, float y,
    char direction, char layer, void *customData, unsigned int customDataSize)
{
    const auto ret = returnAddress();
    auto function = g_createObject[Index];
    const auto token = SokuRLBridge::combatSpawnBegin(parent,owner);
    auto *result = function(
        self, parent, owner, action, x, y, direction, layer, customData, customDataSize);
    SokuRLBridge::combatSpawnEnd(result,parent,owner,token);
    std::uint32_t args[16] = {
        static_cast<std::uint16_t>(action), bits(x), bits(y),
        static_cast<std::uint8_t>(direction), static_cast<std::uint8_t>(layer),
        reinterpret_cast<std::uint32_t>(customData), customDataSize,
        reinterpret_cast<std::uint32_t>(result),
    };
    const auto *payload = static_cast<const float *>(customData);
    for (unsigned int index = 0; payload && index < std::min(customDataSize, 8U); ++index)
        args[8 + index] = bits(payload[index]);
    emitCall(reinterpret_cast<std::uint32_t>(function), ret, owner,
        SokuRLBridge::TraceSubjectKind::Player,
        parent ? FLAG_CHILD_SPAWN : FLAG_DIRECT_SPAWN, args);
    return result;
}

template<std::size_t Index>
void __fastcall handleInputsHook(void *self, void *)
{
    const auto previous=SokuRLBridge::combatInputBegin(self);
    g_handleInputs[Index](self);
    SokuRLBridge::combatInputEnd(self,previous);
}

template<std::size_t Index>
bool installCharacterHooks()
{
    auto **playerVtable = reinterpret_cast<void **>(PLAYER_VTABLES[Index]);
    auto **objectVtable = reinterpret_cast<void **>(OBJECT_VTABLES[Index]);
    auto **listVtable = reinterpret_cast<void **>(OBJECT_LIST_VTABLES[Index]);
    g_playerSetAction[Index] = reinterpret_cast<SetActionFunction>(SokuLib::TamperDword(
        &playerVtable[SET_ACTION_SLOT], setActionHook<Index, true>));
    g_playerSetSequence[Index] = reinterpret_cast<SetSequenceFunction>(SokuLib::TamperDword(
        &playerVtable[SET_SEQUENCE_SLOT], setSequenceHook<Index, true>));
    g_objectSetAction[Index] = reinterpret_cast<SetActionFunction>(SokuLib::TamperDword(
        &objectVtable[SET_ACTION_SLOT], setActionHook<Index, false>));
    g_objectSetSequence[Index] = reinterpret_cast<SetSequenceFunction>(SokuLib::TamperDword(
        &objectVtable[SET_SEQUENCE_SLOT], setSequenceHook<Index, false>));
    g_createObject[Index] = reinterpret_cast<CreateObjectFunction>(SokuLib::TamperDword(
        &listVtable[LIST_CREATE_OBJECT_SLOT], createObjectHook<Index>));
    if (SokuRLBridge::combatCaptureEnabled()) {
        g_handleInputs[Index] = reinterpret_cast<HandleInputsFunction>(SokuLib::TamperDword(
            &playerVtable[20],handleInputsHook<Index>));
        if (!g_handleInputs[Index]) return false;
    }
    return g_playerSetAction[Index] && g_playerSetSequence[Index] &&
        g_objectSetAction[Index] && g_objectSetSequence[Index] && g_createObject[Index];
}
}

namespace SokuRLBridge
{
bool installNativeTraceHooks(
    BridgeMapping *mapping,
    const std::uint64_t *currentFrame,
    TraceSink sink)
{
    g_mapping = mapping;
    g_currentFrame = currentFrame;
    g_sink = sink;
    return installCharacterHooks<0>() && installCharacterHooks<1>();
}
}
