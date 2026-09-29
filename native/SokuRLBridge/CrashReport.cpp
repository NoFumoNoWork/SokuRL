#include "CrashReport.hpp"
#include <Windows.h>
#include <cstdio>

namespace SokuRLBridge {
namespace {
PVOID handler = nullptr;
HANDLE logFile = INVALID_HANDLE_VALUE;
volatile LONG reported = 0;

LONG CALLBACK report(EXCEPTION_POINTERS *exception)
{
    const auto *record = exception->ExceptionRecord;
    // Corrupt game resources can also fault inside a runtime DLL. Record a
    // bounded number of first-chance faults without changing their handling.
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION ||
        InterlockedIncrement(&reported) > 4)
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
    WriteFile(logFile, buffer, static_cast<DWORD>(used), &written, nullptr);
    FlushFileBuffers(logFile);
    return EXCEPTION_CONTINUE_SEARCH;
}
}

bool installCrashReport()
{
    // The game is a GUI process. Python can close inherited Win32 standard
    // handles even though Wine still prints its own messages on Unix stderr.
    wchar_t path[128]{};
    swprintf_s(path, L"modules\\SokuRLBridge\\crash-%lu.log", GetCurrentProcessId());
    logFile = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (logFile == INVALID_HANDLE_VALUE)
        return false;
    handler = AddVectoredExceptionHandler(0, report);
    traceResetStage("installed", nullptr);
    return handler != nullptr;
}

void traceResetStage(const char *stage, const void *scene)
{
    if (logFile == INVALID_HANDLE_VALUE)
        return;
    char buffer[256]{};
    const auto length = sprintf_s(buffer,
        "[DEBUG-reset-crash] pid=%lu tid=%lu stage=%s object=%p live_scene=%lu next_scene=%lu cleanup_count=%lu\n",
        GetCurrentProcessId(), GetCurrentThreadId(), stage, scene,
        *reinterpret_cast<volatile DWORD *>(0x008A0044),
        *reinterpret_cast<volatile DWORD *>(0x008A0040),
        *reinterpret_cast<volatile DWORD *>(0x008A001C));
    DWORD written = 0;
    WriteFile(logFile, buffer, static_cast<DWORD>(length), &written, nullptr);
}

void closeCrashReport()
{
    if (handler)
        RemoveVectoredExceptionHandler(handler);
    handler = nullptr;
    if (logFile != INVALID_HANDLE_VALUE)
        CloseHandle(logFile);
    logFile = INVALID_HANDLE_VALUE;
}
}
