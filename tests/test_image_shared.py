"""Exercise the real image reader across a native reset sentinel."""
import ctypes
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from image_shared import HEADER, ImageClient, PIXEL_OFFSET


def test_reset_pending_image_is_not_a_future_frame(monkeypatch):
    buffer = ctypes.create_string_buffer(PIXEL_OFFSET)
    client = ImageClient.__new__(ImageClient)
    client.view = ctypes.addressof(buffer)

    def publish(sequence, result, frame):
        data = HEADER.pack(0x474D4953, 2, sequence, result, frame, 320, 240, 640, 480)
        ctypes.memmove(client.view, data, len(data))

    publish(2, -2147483638, 2**64 - 1)
    monkeypatch.setattr("image_shared.time.sleep", lambda duration: publish(4, 0, 0))
    result = client._read(0, 1., PIXEL_OFFSET)
    assert HEADER.unpack_from(result)[4] == 0
    publish(6, 0, 1)
    with pytest.raises(RuntimeError, match="advanced"):
        client._read(0, 1., PIXEL_OFFSET)
