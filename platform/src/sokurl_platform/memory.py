"""Validated x86 process reads; no observation or reward policy."""
import ctypes
from ctypes import wintypes

class ProcessMemory:
    def __init__(self, pid):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.ReadProcessMemory.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.OpenProcess(0x0010, False, pid)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "cannot open owned game for observation")
        self.pages = {}

    def begin_frame(self):
        self.pages.clear()

    def read(self, address, size):
        if type(address) is not int or not 0 < address < 2**32 or not 0 < size <= 65536:
            raise ValueError(f"invalid 32-bit game memory read: address={address!r}, size={size}")
        result = bytearray()
        while size:
            page = address & ~4095
            if page not in self.pages:
                self.pages[page] = self.read_page(page)
            offset = address - page
            count = min(size, 4096 - offset)
            result.extend(self.pages[page][offset:offset + count])
            address += count
            size -= count
        return bytes(result)

    def read_page(self, address):
        size = 4096
        buffer = ctypes.create_string_buffer(size)
        count = ctypes.c_size_t()
        if (not self.kernel.ReadProcessMemory(self.handle, address, buffer, size, ctypes.byref(count))
                or count.value != size):
            raise OSError(ctypes.get_last_error(), f"game memory read failed at {address:#x}")
        return buffer.raw

    def close(self):
        self.kernel.CloseHandle(self.handle)
