"""Read an exact simulation frame from the native RGB image mapping."""
import ctypes
import struct
import time

from soku_rl.pixels import RGBFrame
from soku_rl.render_state import CapturedScene, RenderSnapshot, RENDER_STATE_SIZE

from bridge_shared import _kernel32


HEADER = struct.Struct("<IIiiQIIII")
WIDTH, HEIGHT = 320, 240
PIXEL_OFFSET = HEADER.size + RENDER_STATE_SIZE
MAPPING_SIZE = PIXEL_OFFSET + WIDTH * HEIGHT * 3


class ImageClient:
    def __init__(self, pid):
        self.handle = _kernel32.OpenFileMappingW(4, False, rf"Local\SokuRLImage_{pid}")
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "image mapping is unavailable")
        self.view = _kernel32.MapViewOfFile(self.handle, 4, 0, 0, MAPPING_SIZE)
        if not self.view:
            error = ctypes.get_last_error()
            _kernel32.CloseHandle(self.handle)
            self.handle = None
            raise OSError(error, "cannot map native images")

    def read(self, frame, timeout):
        if type(frame) is not int or frame < 0 or timeout <= 0 or not self.view:
            raise ValueError("read requires an open mapping, frame, and positive timeout")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            first = ctypes.c_int32.from_address(self.view + 8).value
            if first > 0 and not first & 1:
                data = ctypes.string_at(self.view, MAPPING_SIZE)
                last = ctypes.c_int32.from_address(self.view + 8).value
                magic, version, sequence, result, captured, width, height, _, _ = HEADER.unpack_from(data)
                if first == last == sequence:
                    if magic != 0x474D4953 or version != 2 or (width, height) != (WIDTH, HEIGHT):
                        raise RuntimeError("unsupported native image mapping")
                    if captured > frame:
                        raise RuntimeError("image advanced beyond the requested simulation frame")
                    if captured == frame:
                        if result < 0:
                            raise RuntimeError(f"native image capture failed: HRESULT {result & 0xFFFFFFFF:08X}")
                        rgb = RGBFrame(int(captured), width, height, data[PIXEL_OFFSET:])
                        render = RenderSnapshot.decode(data[HEADER.size:PIXEL_OFFSET])
                        return CapturedScene(rgb, render)
            time.sleep(0.001)
        raise TimeoutError(f"no rendered image for simulation frame {frame}")

    def close(self):
        if self.view:
            _kernel32.UnmapViewOfFile(self.view)
            self.view = None
        if self.handle:
            _kernel32.CloseHandle(self.handle)
            self.handle = None
