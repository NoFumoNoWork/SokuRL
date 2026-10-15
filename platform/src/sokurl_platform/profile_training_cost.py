"""Controlled host microbenchmarks plus owned-worker pause/reset phase timings."""
from __future__ import annotations
import argparse
import ctypes
import csv
import io
import json
import os
import random
import statistics
import time
from pathlib import Path
import psutil
from sokurl_platform import bridge_shared as b
from sokurl_platform.headless_validation import wait_for_frame_zero
from sokurl_platform.unlimited_benchmark import _drain_fast
from sokurl_platform import sokurl
from sokurl_platform import raw_recorder

ROOT = Path(__file__).resolve().parents[2]


def cpu(process):
    t = process.cpu_times()
    return t.user + t.system


def timed(fn, repeats):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter()-start)/repeats


def micro():
    dll = ctypes.CDLL(str(ROOT / "native/BridgeTools/build/HostBench.dll"))
    dll.fnv64.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    dll.fnv64.restype = ctypes.c_uint64
    dll.copy_bytes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    dll.copy_bytes.restype = None
    # Plain deterministic bytes, not a game-mechanics measurement.
    payload = random.Random(41).randbytes(12424)
    src = ctypes.create_string_buffer(payload)
    def python_hash():
        v = 14695981039346656037
        for byte in payload:
            v = ((v ^ byte) * 1099511628211) & 0xffffffffffffffff
        return v
    assert dll.fnv64(src, len(payload)) == python_hash()
    hashes = []
    for _ in range(3):
        py = timed(python_hash, 300)
        native = timed(lambda: dll.fnv64(src, len(payload)), 20000)
        hashes.append({"python_seconds": py, "cpp_seconds": native, "speedup": py/native})
    n = 256
    records = (b.RawFrameState * n)()
    target = ctypes.create_string_buffer(ctypes.sizeof(records))
    per_frame = lambda: [b._copy_struct(row, b.RawFrameState) for row in records]
    bulk = lambda: ctypes.memmove(target, records, ctypes.sizeof(records))
    cpp = lambda: dll.copy_bytes(target, records, ctypes.sizeof(records))
    copies = [{"typed_frames_seconds": timed(per_frame, 100),
               "ctypes_bulk_seconds": timed(bulk, 100),
               "cpp_bulk_seconds": timed(cpp, 100)} for _ in range(3)]
    assert bytes(target)[:ctypes.sizeof(records)] == bytes(records)
    state = b.RawFrameState()
    assert b.calculate_state_hash(state, backend="cpp") == b.calculate_state_hash(state, backend="python")
    end_to_end = []
    for _ in range(3):
        py = timed(lambda: b.calculate_state_hash(state, backend="python"), 100)
        cpp_time = timed(lambda: b.calculate_state_hash(state, backend="cpp"), 1000)
        end_to_end.append({"python_seconds": py, "cpp_seconds": cpp_time, "speedup": py/cpp_time})
    return {"hash": hashes, "end_to_end_hash": end_to_end, "copy_256_frames": copies,
            "note": "algorithm-only / memory-only; no claim about complete PPO speedup"}


def profile_read(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenFileMappingW.argtypes = [ctypes.c_uint32,ctypes.c_int,ctypes.c_wchar_p]
    kernel.OpenFileMappingW.restype = ctypes.c_void_p
    kernel.MapViewOfFile.argtypes = [ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_size_t]
    kernel.MapViewOfFile.restype = ctypes.c_void_p
    kernel.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenFileMappingW(4,False,f"Local\\SokuRLBridgePerf_{pid}")
    if not handle:
        return None
    address = kernel.MapViewOfFile(handle,4,0,0,48)
    try:
        if not address:
            raise ctypes.WinError(ctypes.get_last_error())
        names = ["frequency","frames","simulation","capture","hash","recording"]
        return dict(zip(names,(ctypes.c_uint64*6).from_address(address)))
    finally:
        if address:
            kernel.UnmapViewOfFile(address)
        kernel.CloseHandle(handle)


def recording_micro(root):
    frames = (b.RawFrameState*256)()
    for i, frame in enumerate(frames):
        frame.frameId = i
    rows = []
    host = psutil.Process()
    for repetition in range(3):
        text = io.StringIO(newline="")
        inputs = io.StringIO(newline="")
        frame_writer = csv.DictWriter(text,fieldnames=raw_recorder.FRAME_FIELDS)
        input_writer = csv.DictWriter(inputs,fieldnames=raw_recorder.INPUT_FIELDS)
        start, before = time.perf_counter(), cpu(host)
        for frame in frames:
            frame_writer.writerow(raw_recorder.frame_row(frame))
            input_writer.writerow(raw_recorder.input_row(frame))
        format_wall, format_cpu = time.perf_counter()-start, cpu(host)-before
        writer = raw_recorder.RawSessionWriter(root,{"synthetic_performance_sample": True})
        start, before = time.perf_counter(),cpu(host)
        assert writer.write(frames) == 256
        write_wall, write_cpu = time.perf_counter()-start,cpu(host)-before
        writer.close(dropped_frames=0,validation="SYNTHETIC PERFORMANCE ONLY")
        payload = bytes(frames)
        start = time.perf_counter()
        path = root / f"binary-{repetition}.raw"
        with path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
        binary_wall = time.perf_counter()-start
        assert path.read_bytes() == payload
        rows.append({"csv_format_wall_seconds":format_wall,"csv_format_cpu_seconds":format_cpu,
                     "csv_write_flush_wall_seconds":write_wall,"csv_write_cpu_seconds":write_cpu,
                     "binary_write_flush_wall_seconds":binary_wall,"binary_bytes":len(payload)})
    return {"rows":rows,"frames_per_batch":256,
            "boundary":"synthetic states; OS cached writes, no fsync or compression; CSV and raw schemas differ"}


def live(resets):
    process = sokurl._launch_vs_from_title(35,headless=True,unlimited=True,
                                          seed=0x534f4b55,pause_at_start=True)
    client = b.BridgeClient(process.pid)
    host = psutil.Process()
    try:
        wait_for_frame_zero(client,process.pid)
        before = client.snapshot()
        start, game, py = time.perf_counter(),cpu(process),cpu(host)
        time.sleep(3)
        paused = {"seconds": time.perf_counter()-start,
                  "game_cpu_seconds": cpu(process)-game,"python_cpu_seconds":cpu(host)-py,
                  "start_frame":before.game_frame,"end_frame":client.snapshot().game_frame}
        buffer = (ctypes.c_ubyte * (b.FRAME_RING_CAPACITY*ctypes.sizeof(b.RawFrameState)))()
        rows = []
        for i in range(resets):
            current = client.snapshot()
            seed = 1001+i
            start, game, py = time.perf_counter(),cpu(process),cpu(host)
            sequence = client.reset_episode(seed)
            sent = time.perf_counter()
            client.wait_for_reset(sequence,current.latest.segmentId+1,seed,35)
            ready = time.perf_counter()
            reset_cost = {"wall_seconds":ready-start,"send_seconds":sent-start,
                          "game_cpu_seconds":cpu(process)-game,"python_cpu_seconds":cpu(host)-py}
            start, game, py = time.perf_counter(),cpu(process),cpu(host)
            sequence = client.step(256)
            while True:
                block = client.block
                if block.ackSeq == sequence and block.currentFrame == 256 and block.runState == 1:
                    break
                if time.perf_counter()-start>10:
                    raise TimeoutError("step did not finish")
                time.sleep(.0002)
            advance_end = time.perf_counter()
            consume_start = time.perf_counter()
            count = _drain_fast(client,buffer)
            snapshot = client.snapshot()
            consume_end = time.perf_counter()
            assert snapshot.latest.stateHash == b.calculate_state_hash(snapshot.latest)
            rows.append({"reset": reset_cost,"advance_256_wall_seconds":advance_end-start,
                         "advance_game_cpu_seconds":cpu(process)-game,
                         "advance_host_cpu_seconds":cpu(host)-py,
                         "consume_wall_seconds":consume_end-consume_start,
                         "consumed":count,"frame":snapshot.game_frame,"seed":seed,
                         "state_hash":int(snapshot.latest.stateHash)})
        return {"paused":paused,"cycles":rows,"stages":profile_read(process.pid),
                "idle_wait":os.environ.get("SOKURL_IDLE_WAIT","0"),
                "profile_stages":os.environ.get("SOKURL_PROFILE_STAGES","0")}
    finally:
        client.close()
        if process.is_running():
            sokurl.shutdown(5,process.pid)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resets",type=int,default=5)
    parser.add_argument("--idle-wait",action="store_true")
    parser.add_argument("--profile-stages",action="store_true")
    parser.add_argument("--host-only",action="store_true")
    args=parser.parse_args()
    os.environ["SOKURL_IDLE_WAIT"] = "1" if args.idle_wait else "0"
    os.environ["SOKURL_PROFILE_STAGES"] = "1" if args.profile_stages else "0"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result={"micro":micro(),"recording":recording_micro(args.output.parent / "recording_samples")}
    args.output.write_text(json.dumps(result,indent=2))
    if not args.host_only:
        result["live"]=live(args.resets)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
