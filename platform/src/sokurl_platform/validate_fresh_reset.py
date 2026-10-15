"""Compare snapshot-free per-instance fresh resets against original new games."""
import argparse
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path

from sokurl_platform import bridge_shared as b
from sokurl_platform import sokurl
from sokurl_platform.headless_validation import fixed_trace, step_group, wait_for_frame_zero
from sokurl_platform.frame_validation import state_diff


def rng_hash(pid):
    handle = sokurl.kernel32.OpenProcess(sokurl.PROCESS_QUERY_INFORMATION | sokurl.PROCESS_VM_READ, False, pid)
    if not handle: raise ctypes.WinError(ctypes.get_last_error())
    digest = hashlib.sha256()
    try:
        for address,size in [(0x89B660,624*4),(0x896B20,4)]:
            buffer = (ctypes.c_ubyte*size)()
            count = ctypes.c_size_t()
            if not sokurl.kernel32.ReadProcessMemory(handle,ctypes.c_void_p(address),buffer,size,ctypes.byref(count)) or count.value != size:
                raise ctypes.WinError(ctypes.get_last_error())
            digest.update(bytes(buffer))
        return digest.hexdigest()
    finally:
        sokurl.kernel32.CloseHandle(handle)


def run(workers, frames, cycles, output, active_reset=False, immediate_input=False, characters=None, terminal_reset=False):
    report = {"status":"RUNNING","workers":workers,"frames":frames,"resets":[]}
    def save(): output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    def launch(seeds, fast):
        os.environ.pop("SOKURL_FAST_RESET_PROVIDER",None)
        os.environ.pop("SOKURL_RESET_TIMING_DIR",None)
        os.environ["SOKURL_FRESH_RESET"] = "1" if fast else "0"
        processes = sokurl._launch_vs_group_from_title(workers,40,headless=True,unlimited=True,
            seeds=seeds,pause_at_start=True,characters=characters)
        clients = [b.BridgeClient(p.pid) for p in processes]
        initial = [wait_for_frame_zero(c,p.pid,40) for c,p in zip(clients,processes)]
        return processes,clients,initial
    def close(processes,clients):
        for c in clients: c.close()
        for p in processes:
            if p.is_running(): sokurl.shutdown(5,p.pid)
    def trace(count):
        if immediate_input:
            first = [(b.ACTION_INPUTS["C"],b.ACTION_INPUTS["B"])]*min(8,count)
            return first+fixed_trace(count-len(first))
        return fixed_trace(count)
    def trajectory(clients, count=None):
        states = [[] for _ in clients]
        for frame,(p1,p2) in enumerate(trace(frames if count is None else count),1):
            for i,state in enumerate(step_group(clients,p1,p2,frame)):
                assert state.stateHash == b.calculate_state_hash(state)
                states[i].append(state)
                clients[i].drain_frame_bytes()
        return states
    references = {}
    reference_rng = {}
    seed_sets = [tuple(100+j*100+i for i in range(workers)) for j in range(2)]
    save()
    for seeds in seed_sets:
        processes,clients,initial = launch(seeds,False)
        try:
            reference_rng[seeds] = [rng_hash(p.pid) for p in processes]
            references[seeds] = initial,trajectory(clients)
        finally:
            close(processes,clients)
    processes,clients,initial = launch(seed_sets[0],True)
    try:
        trajectory(clients)
        for cycle in range(cycles):
            seeds = seed_sets[cycle%2]
            start = time.perf_counter()
            segments = [int(c.snapshot().latest.segmentId)+1 for c in clients]
            commands = [c.reset_episode(seed) for c,seed in zip(clients,seeds)]
            actual_initial = []
            row = {"cycle":cycle,"seeds":seeds,"pids":[p.pid for p in processes],"seconds":[]}
            for i,(c,seq,seed,segment) in enumerate(zip(clients,commands,seeds,segments)):
                raw = c.wait_for_reset(seq,segment,seed,10)
                row["seconds"].append(time.perf_counter()-start)
                actual_initial.append(raw)
                c.drain_frame_bytes()
            expected_initial,expected = references[seeds]
            assert [rng_hash(p.pid) for p in processes] == reference_rng[seeds], "native RNG differs after fresh reset"
            row["native_rng"] = "PASS"
            for i,(actual,expected_state) in enumerate(zip(actual_initial,expected_initial)):
                if actual.stateHash != expected_state.stateHash:
                    report["first_difference"] = {"slot":i,"frame":0,"fields":state_diff(expected_state,actual)[:30]}
                    save()
                    raise RuntimeError("fresh reset initial state differs")
            actual = trajectory(clients)
            for i,(left,right) in enumerate(zip(actual,expected)):
                for frame,(a,e) in enumerate(zip(left,right),1):
                    if a.stateHash != e.stateHash:
                        report["first_difference"] = {"slot":i,"frame":frame,"fields":state_diff(e,a)[:30]}
                        save()
                        raise RuntimeError("fresh reset trajectory differs")
            row["status"] = "PASS"
            row["private_bytes"] = [getattr(p.memory_info(),"private",p.memory_info().vms) for p in processes]
            report["resets"].append(row)
            save()
            if active_reset:
                segments = [int(c.snapshot().latest.segmentId)+1 for c in clients]
                commands = [c.reset_episode(seed) for c,seed in zip(clients,seeds)]
                for c,seq,segment,seed in zip(clients,commands,segments,seeds):
                    c.wait_for_reset(seq,segment,seed,10)
                    c.drain_frame_bytes()
                count = min(frames,[400,600,1050,1750][cycle%4])
                prepared = trajectory(clients,count)
                row["next_reset_from"] = [{"frame":count,"objects":[int(s[-1].p1ObjectCount),int(s[-1].p2ObjectCount)],
                    "actions":[int(s[-1].p1.actionId),int(s[-1].p2.actionId)],"hp":[int(s[-1].p1.hp),int(s[-1].p2.hp)]} for s in prepared]
                for left,right in zip(prepared,expected):
                    assert [s.stateHash for s in left] == [s.stateHash for s in right[:count]]
                save()
        if workers > 1:
            neighbors = [c.snapshot().latest for c in clients[1:]]
            segment = int(clients[0].snapshot().latest.segmentId)+1
            sequence = clients[0].reset_episode(seed_sets[0][0])
            clients[0].wait_for_reset(sequence,segment,seed_sets[0][0],10)
            for c,old in zip(clients[1:],neighbors):
                current = c.snapshot().latest
                assert (current.stateHash,current.segmentId,current.frameId) == (old.stateHash,old.segmentId,old.frameId)
            report["unselected_instances_unchanged"] = "PASS"
        if terminal_reset:
            c = clients[0]
            frame = int(c.snapshot().game_frame)
            for tick in range(6000):
                phase = tick%80
                keys = b.ACTION_INPUTS["RIGHT"] if phase < 30 else b.ACTION_INPUTS["A"] if phase < 33 else b.ACTION_INPUTS["NEUTRAL"]
                state, = step_group([c],keys,b.ACTION_INPUTS["NEUTRAL"],frame+tick+1)
                c.drain_frame_bytes()
                if state.p1.hp <= 0 or state.p2.hp <= 0:
                    report["terminal_frame"] = int(state.frameId)
                    segment = int(state.segmentId)+1
                    seq = c.reset_episode(seed_sets[0][0])
                    fresh = c.wait_for_reset(seq,segment,seed_sets[0][0],10)
                    assert fresh.stateHash == references[seed_sets[0]][0][0].stateHash
                    assert rng_hash(processes[0].pid) == reference_rng[seed_sets[0]][0]
                    resumed = trajectory([c])[0]
                    assert [s.stateHash for s in resumed] == [s.stateHash for s in references[seed_sets[0]][1][0]]
                    report["terminal_then_new_world"] = "PASS"
                    break
            else:
                report["terminal_then_new_world"] = "NOT TESTED: no KO in bounded input sequence"
        report["status"] = "PASS"
        save()
        return report
    finally:
        close(processes,clients)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--frames",type=int,default=1000)
    parser.add_argument("--cycles",type=int,default=4)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--active-reset",action="store_true")
    parser.add_argument("--immediate-input",action="store_true")
    parser.add_argument("--characters",type=int,nargs=2,choices=(0,1))
    parser.add_argument("--terminal-reset",action="store_true")
    args = parser.parse_args()
    if min(args.workers,args.frames,args.cycles) < 1: parser.error("positive values required")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try: report = run(args.workers,args.frames,args.cycles,args.output,args.active_reset,args.immediate_input,args.characters,args.terminal_reset)
    except Exception as error:
        report = json.loads(args.output.read_text()) if args.output.exists() else {}
        report.update(status="FAIL",error=repr(error))
        args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))
    raise SystemExit(report["status"] != "PASS")
