"""Read-only, seqlock-protected health samples for live ABI 12 workers."""
from __future__ import annotations
import argparse
import ctypes
import json
import time
from pathlib import Path
import psutil
import bridge_shared as b


def sample(pid):
    handle = b._kernel32.OpenFileMappingW(4, False, b.MAPPING_NAME_FORMAT.format(pid))
    if not handle:
        raise RuntimeError(f"PID {pid}: mapping unavailable")
    view = b._kernel32.MapViewOfFile(handle, 4, 0, 0, ctypes.sizeof(b.ControlBlock))
    if not view:
        b._kernel32.CloseHandle(handle)
        raise RuntimeError(f"PID {pid}: prefix mapping failed")
    try:
        ptr = ctypes.cast(view, ctypes.POINTER(b.ControlBlock))
        for _ in range(10000):
            sequence = ptr.contents.statusSeq
            if sequence & 1:
                continue
            block = b.ControlBlock.from_buffer_copy(bytes(ptr.contents))
            if sequence == ptr.contents.statusSeq:
                break
        else:
            raise RuntimeError(f"PID {pid}: status writer never stabilized")
        if (block.magic, block.version, block.structSize) != (
                b.CONTROL_MAGIC, 12, ctypes.sizeof(b.ControlBlock)):
            raise RuntimeError("read-only inspector requires matching ABI12 control prefix")
        cpu = psutil.Process(pid).cpu_times()
        return {"pid": pid, "connected": block.connected, "in_gameplay": block.inGameplay,
                "frame": block.currentFrame, "latest_frame": block.latest.frameId,
                "command_seq": block.commandSeq, "ack_seq": block.ackSeq,
                "ring_write": block.ringWriteSeq, "ring_read": block.ringReadSeq,
                "dropped": block.droppedFrames, "trace_capacity": block.traceCapacity,
                "cpu_seconds": cpu.user + cpu.system,
                "players": [{"action": p.actionId, "hp": p.hp, "spirit": p.spirit,
                             "x": p.x, "y": p.y, "objects": p.objectCount}
                            for p in (block.latest.p1, block.latest.p2)]}
    finally:
        b._kernel32.UnmapViewOfFile(view)
        b._kernel32.CloseHandle(handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    pids = [p.pid for p in psutil.process_iter(["name"])
            if (p.info["name"] or "").lower() == "th123.exe"]
    before = [sample(pid) for pid in pids]
    time.sleep(args.interval)
    after = [sample(pid) for pid in pids]
    report = {"status": "NO_WORKERS" if not pids else "OBSERVED",
              "before": before, "after": after,
              "deltas": [{"pid": a["pid"], "frames": a["frame"] - z["frame"],
                          "commands": a["ack_seq"] - z["ack_seq"],
                          "ring_produced": a["ring_write"] - z["ring_write"],
                          "ring_consumed": a["ring_read"] - z["ring_read"],
                          "cpu_seconds": a["cpu_seconds"] - z["cpu_seconds"]}
                         for z, a in zip(before, after, strict=True)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
