"""Measure original fresh-scene reset transitions, without checkpoint provider."""
import argparse
import json
import os
import time
from pathlib import Path

import bridge_shared as b
import sokurl
from headless_validation import wait_for_frame_zero


def run(workers, cycles):
    os.environ.pop("SOKURL_FAST_RESET_PROVIDER", None)
    processes = sokurl._launch_vs_group_from_title(workers, 40, headless=True,
        unlimited=True, seeds=tuple(1732+i for i in range(workers)), pause_at_start=True)
    clients = []
    reports = []
    try:
        for process in processes:
            client = b.BridgeClient(process.pid)
            clients.append(client)
            wait_for_frame_zero(client, process.pid, 40)
        for cycle in range(cycles):
            start = time.perf_counter()
            commands = [client.reset_episode(100+cycle*workers+i) for i,client in enumerate(clients)]
            pending = set(range(workers))
            rows = [{"pid":p.pid,"transitions":[]} for p in processes]
            last = {}
            while pending and time.perf_counter()-start < 40:
                for i in list(pending):
                    snap = clients[i].snapshot()
                    scene,mode,*_ = sokurl._read_process_values(processes[i].pid)
                    key = (scene,mode,snap.in_gameplay,snap.result_code)
                    if key != last.get(i):
                        rows[i]["transitions"].append({"seconds":time.perf_counter()-start,"state":key})
                        last[i] = key
                    if snap.ack_seq == commands[i] and snap.in_gameplay and snap.result_code == 2 and snap.latest.randomSeed == 100+cycle*workers+i and snap.game_frame == 0:
                        rows[i]["seconds"] = time.perf_counter()-start
                        pending.remove(i)
                time.sleep(.005)
            if pending:
                raise TimeoutError(f"fresh reset timeout slots={pending}")
            reports.append(rows)
        return {"status":"PASS","workers":workers,"cycles":reports}
    finally:
        for client in clients:
            client.close()
        for process in processes:
            if process.is_running():
                sokurl.shutdown(5, process.pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--cycles",type=int,default=3)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    try:
        report = run(args.workers,args.cycles)
    except Exception as error:
        report = {"status":"FAIL","error":repr(error)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))
    raise SystemExit(report["status"] != "PASS")
