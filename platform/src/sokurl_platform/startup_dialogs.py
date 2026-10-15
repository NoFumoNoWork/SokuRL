"""Read blocking Windows dialogs from an explicitly owned set of game PIDs."""
import ctypes
from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)
WINDOW_CALLBACK = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [WINDOW_CALLBACK, wintypes.LPARAM]
user32.EnumChildWindows.argtypes = [wintypes.HWND, WINDOW_CALLBACK, wintypes.LPARAM]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL


def blocking_dialogs(pids):
    if not pids or any(type(pid) is not int or pid <= 0 for pid in pids):
        raise ValueError("owned positive process IDs are required")
    dialogs = []

    @WINDOW_CALLBACK
    def window(hwnd, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value not in pids or not user32.IsWindowVisible(hwnd):
            return True
        kind = ctypes.create_unicode_buffer(128)
        user32.GetClassNameW(hwnd, kind, len(kind))
        if kind.value != "#32770":
            return True
        title = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, title, len(title))
        messages = []

        @WINDOW_CALLBACK
        def child(control, _):
            text = ctypes.create_unicode_buffer(2048)
            user32.GetWindowTextW(control, text, len(text))
            if text.value:
                messages.append(text.value)
            return True

        user32.EnumChildWindows(hwnd, child, 0)
        dialogs.append({"pid": pid.value, "title": title.value, "messages": messages})
        return True

    if not user32.EnumWindows(window, 0):
        raise OSError(ctypes.get_last_error(), "cannot inspect game startup dialogs")
    return dialogs
