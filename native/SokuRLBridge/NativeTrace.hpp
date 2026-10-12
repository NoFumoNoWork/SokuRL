#pragma once

#include "ControlBlock.hpp"

namespace SokuRLBridge
{
using TraceSink = void (*)(const TraceEvent &event);

bool installNativeTraceHooks(
    BridgeMapping *mapping,
    const std::uint64_t *currentFrame,
    TraceSink sink);
}
