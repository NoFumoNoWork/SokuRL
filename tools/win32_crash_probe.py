from __future__ import annotations

import ctypes
import json
import struct
import time
from ctypes import wintypes
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GAME_DIR = ROOT / "th123_jp"
GAME_EXE = GAME_DIR / "th123.exe"
DEBUG_ONLY_THIS_PROCESS = 0x00000002
DBG_CONTINUE = 0x00010002
DBG_EXCEPTION_NOT_HANDLED = 0x80010001
EXCEPTION_INVALID_PARAMETER = 0xC000000D
EXCEPTION_BREAKPOINT = 0x80000003
INVALID_PARAMETER_WRAPPER = 0x0081FF70
WOW64_CONTEXT_FULL = 0x00010007
THREAD_ALL_FOR_CONTEXT = 0x0008 | 0x0040 | 0x0800


class STARTUPINFO(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
        ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD),
    ]


class WOW64_FLOATING_SAVE_AREA(ctypes.Structure):
    _fields_ = [
        ("ControlWord", wintypes.DWORD), ("StatusWord", wintypes.DWORD),
        ("TagWord", wintypes.DWORD), ("ErrorOffset", wintypes.DWORD),
        ("ErrorSelector", wintypes.DWORD), ("DataOffset", wintypes.DWORD),
        ("DataSelector", wintypes.DWORD), ("RegisterArea", ctypes.c_ubyte * 80),
        ("Cr0NpxState", wintypes.DWORD),
    ]


class WOW64_CONTEXT(ctypes.Structure):
    _fields_ = [
        ("ContextFlags", wintypes.DWORD),
        ("Dr0", wintypes.DWORD), ("Dr1", wintypes.DWORD),
        ("Dr2", wintypes.DWORD), ("Dr3", wintypes.DWORD),
        ("Dr6", wintypes.DWORD), ("Dr7", wintypes.DWORD),
        ("FloatSave", WOW64_FLOATING_SAVE_AREA),
        ("SegGs", wintypes.DWORD), ("SegFs", wintypes.DWORD),
        ("SegEs", wintypes.DWORD), ("SegDs", wintypes.DWORD),
        ("Edi", wintypes.DWORD), ("Esi", wintypes.DWORD),
        ("Ebx", wintypes.DWORD), ("Edx", wintypes.DWORD),
        ("Ecx", wintypes.DWORD), ("Eax", wintypes.DWORD),
        ("Ebp", wintypes.DWORD), ("Eip", wintypes.DWORD),
        ("SegCs", wintypes.DWORD), ("EFlags", wintypes.DWORD),
        ("Esp", wintypes.DWORD), ("SegSs", wintypes.DWORD),
        ("ExtendedRegisters", ctypes.c_ubyte * 512),
    ]


def probe(timeout: float = 30.0) -> dict[str, object]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    startup = STARTUPINFO()
    startup.cb = ctypes.sizeof(startup)
    process = PROCESS_INFORMATION()
    command = ctypes.create_unicode_buffer(f'"{GAME_EXE}"')
    if not kernel32.CreateProcessW(
        str(GAME_EXE), command, None, None, False, DEBUG_ONLY_THIS_PROCESS,
        None, str(GAME_DIR), ctypes.byref(startup), ctypes.byref(process),
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    event = (ctypes.c_ubyte * 176)()
    deadline = time.monotonic() + timeout
    result: dict[str, object] = {"pid": process.dwProcessId}

    def capture_context(tid: int) -> tuple[WOW64_CONTEXT, list[int]]:
        thread = kernel32.OpenThread(THREAD_ALL_FOR_CONTEXT, False, tid)
        if not thread:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            context = WOW64_CONTEXT()
            context.ContextFlags = WOW64_CONTEXT_FULL
            if not kernel32.Wow64GetThreadContext(thread, ctypes.byref(context)):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel32.CloseHandle(thread)
        stack = (ctypes.c_ubyte * 1024)()
        read = ctypes.c_size_t()
        success = kernel32.ReadProcessMemory(
            process.hProcess, ctypes.c_void_p(context.Esp), stack,
            len(stack), ctypes.byref(read),
        )
        if not success and not read.value:
            raise ctypes.WinError(ctypes.get_last_error())
        count = read.value // 4
        return context, list(struct.unpack_from(f"<{count}I", bytes(stack), 0))

    def save_context(kind: str, tid: int, context: WOW64_CONTEXT, words: list[int]) -> None:
        result.update({
            "stop": kind,
            "thread_id": tid,
            "registers": {
                name.lower(): f"{getattr(context, name):08X}"
                for name in ("Eip", "Esp", "Ebp", "Eax", "Ebx", "Ecx", "Edx", "Esi", "Edi")
            },
            "stack_words": [f"{value:08X}" for value in words],
            "probable_th123_returns": [
                f"{value:08X}" for value in words if 0x00401000 <= value < 0x00860000
            ],
        })

    try:
        while time.monotonic() < deadline:
            if not kernel32.WaitForDebugEvent(ctypes.byref(event), 500):
                if ctypes.get_last_error() == 121:
                    continue
                raise ctypes.WinError(ctypes.get_last_error())
            code, pid, tid = struct.unpack_from("<III", bytes(event), 0)
            continuation = DBG_CONTINUE
            if code == 3:
                original = (ctypes.c_ubyte * 1)()
                read = ctypes.c_size_t()
                if not kernel32.ReadProcessMemory(
                    process.hProcess, ctypes.c_void_p(INVALID_PARAMETER_WRAPPER),
                    original, 1, ctypes.byref(read),
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                breakpoint = (ctypes.c_ubyte * 1)(0xCC)
                written = ctypes.c_size_t()
                if not kernel32.WriteProcessMemory(
                    process.hProcess, ctypes.c_void_p(INVALID_PARAMETER_WRAPPER),
                    breakpoint, 1, ctypes.byref(written),
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                kernel32.FlushInstructionCache(
                    process.hProcess, ctypes.c_void_p(INVALID_PARAMETER_WRAPPER), 1
                )
                result["breakpoint_original"] = f"{original[0]:02X}"
            elif code == 1:
                exception = struct.unpack_from("<I", bytes(event), 16)[0]
                context, words = capture_context(tid)
                if exception == EXCEPTION_BREAKPOINT and context.Eip == INVALID_PARAMETER_WRAPPER + 1:
                    save_context("invalid_parameter_wrapper", tid, context, words)
                    result["exception"] = f"{exception:08X}"
                    kernel32.ContinueDebugEvent(pid, tid, DBG_CONTINUE)
                    break
                if exception == EXCEPTION_INVALID_PARAMETER:
                    save_context("invalid_parameter_exception", tid, context, words)
                    result["exception"] = f"{exception:08X}"
                    continuation = DBG_EXCEPTION_NOT_HANDLED
                    kernel32.ContinueDebugEvent(pid, tid, continuation)
                    break
            elif code == 5:
                result["exited_before_exception"] = True
                result["exit_code"] = f"{struct.unpack_from('<I', bytes(event), 16)[0]:08X}"
                break
            kernel32.ContinueDebugEvent(pid, tid, continuation)
        else:
            result["timeout"] = True
    finally:
        kernel32.TerminateProcess(process.hProcess, 1)
        kernel32.CloseHandle(process.hThread)
        kernel32.CloseHandle(process.hProcess)
    return result


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
