#include "CrashReport.hpp"
#include <Windows.h>
#include <cstdio>

namespace SokuRLBridge {
namespace {
PVOID handler = nullptr;
volatile LONG reported = 0;

LONG CALLBACK report(EXCEPTION_POINTERS *exception)
{
    const auto *record = exception->ExceptionRecord;
    const auto address = reinterpret_cast<DWORD>(record->ExceptionAddress);
    // Reset faults have occurred in several game resource containers.
    // Keep Wine and system DLL exceptions outside this temporary probe.
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION ||
        (address < 0x00401000 || address >= 0x00858000) ||
        InterlockedCompareExchange(&reported, 1, 0))
        return EXCEPTION_CONTINUE_SEARCH;
    const auto &context = *exception->ContextRecord;
    char buffer[8192]{};
    int used = sprintf_s(buffer, sizeof(buffer),
        "[DEBUG-reset-crash] pid=%lu tid=%lu ip=%08lX access=%lu target=%08lX "
        "eax=%08lX ebx=%08lX ecx=%08lX edx=%08lX esi=%08lX edi=%08lX "
        "esp=%08lX ebp=%08lX flags=%08lX\n",
        GetCurrentProcessId(), GetCurrentThreadId(), context.Eip,
        record->ExceptionInformation[0], record->ExceptionInformation[1],
        context.Eax, context.Ebx, context.Ecx, context.Edx, context.Esi,
        context.Edi, context.Esp, context.Ebp, context.EFlags);
    __try {
        const auto *stack = reinterpret_cast<const DWORD *>(context.Esp);
        for (unsigned i = 0; i < 128; i += 4)
            used += sprintf_s(buffer + used, sizeof(buffer) - used,
                "[DEBUG-reset-crash] %08lX: %08lX %08lX %08lX %08lX\n",
                context.Esp + i * sizeof(DWORD), stack[i], stack[i + 1], stack[i + 2], stack[i + 3]);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        used += sprintf_s(buffer + used, sizeof(buffer) - used, "[DEBUG-reset-crash] unreadable stack boundary\n");
    }
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), buffer, static_cast<DWORD>(used), &written, nullptr);
    return EXCEPTION_CONTINUE_SEARCH;
}
}

bool installCrashReport()
{
    handler = AddVectoredExceptionHandler(0, report);
    return handler != nullptr;
}

void closeCrashReport()
{
    if (handler)
        RemoveVectoredExceptionHandler(handler);
    handler = nullptr;
}
}
