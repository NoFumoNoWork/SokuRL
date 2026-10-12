"""Compare experimental pacing/timing against unchanged exported state."""
import argparse
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path

import bridge_shared as b
import sokurl
from headless_validation import fixed_trace, wait_for_frame_zero, step_group, _coverage_update
from profile_training_cost import profile_read


def run(frames):
    results = []
    reference = None
    for idle, profile in [(False, False), (True, False), (True, True)]:
        os.environ["SOKURL_IDLE_WAIT"] = str(int(idle))
        os.environ["SOKURL_PROFILE_STAGES"] = str(int(profile))
        process = sokurl._launch_vs_from_title(35,headless=True,unlimited=True,
                                              seed=0x534f4b55,pause_at_start=True)
        client = b.BridgeClient(process.pid)
        hashes = []
        coverage = dict(max_p1_objects=0,max_p2_objects=0,min_p1_hp=10000,
                        min_p2_hp=10000,max_hitstop=0,max_untech=0,airborne_seen=False,
                        min_x=10000.,max_x=0.,actions=set())
        latencies = []
        try:
            wait_for_frame_zero(client,process.pid)
            start = time.perf_counter()
            for frame, (p1,p2) in enumerate(fixed_trace(frames),1):
                sent = time.perf_counter()
                state, = step_group([client],p1,p2,frame)
                latencies.append(time.perf_counter()-sent)
                assert state.stateHash == b.calculate_state_hash(state)
                hashes.append(int(state.stateHash))
                _coverage_update(coverage,state)
                payload = client.drain_frame_bytes()
                assert len(payload) == ctypes.sizeof(b.RawFrameState)
            if reference is None:
                reference = hashes
            if hashes != reference:
                first = next(i for i,(a,z) in enumerate(zip(hashes,reference)) if a != z)
                raise AssertionError(f"state divergence at frame {first+1}")
            coverage["actions"] = sorted(coverage["actions"])
            ordered = sorted(latencies)
            results.append({"idle_wait":idle,"profile":profile,"frames":frames,
                            "seconds":time.perf_counter()-start,"latency_mean":sum(latencies)/frames,
                            "latency_p95":ordered[int(.95*(frames-1))],
                            "coverage":coverage,"dropped":int(client.block.droppedFrames),
                            "stages":profile_read(process.pid),
                            "hash_sequence_sha256":hashlib.sha256(json.dumps(hashes).encode()).hexdigest(),
                            "state_hashes":hashes})
        finally:
            client.close()
            if process.is_running():
                sokurl.shutdown(5,process.pid)
    return {"status":"PASS","scope":"exported ABI12 projection, not complete hidden state","runs":results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames",type=int,default=2000)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("frames must be positive")
    try:
        result = run(args.frames)
    except Exception as error:
        result = {"status":"FAIL","error":repr(error)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({key:value for key,value in result.items() if key != "runs"}))
    raise SystemExit(result["status"] != "PASS")
