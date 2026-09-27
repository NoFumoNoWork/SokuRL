from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from bridge_shared import ACTION_INPUTS, BridgeClient, BridgeUnavailable
from raw_recorder import RawSessionWriter

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


class DebugPanel:
    POLL_MS = 50

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SokuRL Debug Panel")
        self.client: BridgeClient | None = None
        self.writer: RawSessionWriter | None = None
        self.vars = {name: tk.StringVar(value=value) for name, value in {
            "connection": "Disconnected", "scene": "Other scene", "run": "-",
            "frame": "0", "checkpoint": "Invalid", "recorded": "0", "dropped": "0",
            "output": "-", "deterministic": "UNKNOWN", "verified": "-", "divergent": "-",
            "p1": "-", "p2": "-", "global": "-", "result": "-",
        }.items()}
        self.goto_var = tk.StringVar(value="0")
        self.action_var = tk.StringVar(value="RIGHT")
        self.action_frames_var = tk.StringVar(value="120")
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(0, self.poll)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.grid(sticky="nsew")
        status = ttk.LabelFrame(outer, text="Connection", padding=8)
        status.grid(row=0, column=0, columnspan=2, sticky="ew")
        for row, (label, name) in enumerate((
            ("Status", "connection"), ("Scene", "scene"), ("Result", "result"),
        )):
            ttk.Label(status, text=label).grid(row=row, column=0, sticky="w")
            ttk.Label(status, textvariable=self.vars[name], width=44).grid(row=row, column=1, sticky="w")

        simulation = ttk.LabelFrame(outer, text="Simulation", padding=8)
        simulation.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        ttk.Label(simulation, text="State").grid(row=0, column=0, sticky="w")
        ttk.Label(simulation, textvariable=self.vars["run"]).grid(row=0, column=1, sticky="w")
        ttk.Label(simulation, text="Current frame").grid(row=1, column=0, sticky="w")
        ttk.Label(simulation, textvariable=self.vars["frame"]).grid(row=1, column=1, sticky="w")
        commands = (("Run", self.run), ("Pause", self.pause), ("+1F", lambda: self.step(1)),
                    ("+10F", lambda: self.step(10)), ("+60F", lambda: self.step(60)),
                    ("-1F", self.step_back))
        for index, (label, command) in enumerate(commands):
            ttk.Button(simulation, text=label, command=command, width=9).grid(
                row=2 + index // 3, column=index % 3, padx=2, pady=2)

        checkpoint = ttk.LabelFrame(outer, text="Checkpoint / Navigation", padding=8)
        checkpoint.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(8, 0))
        ttk.Label(checkpoint, text="Checkpoint").grid(row=0, column=0, sticky="w")
        ttk.Label(checkpoint, textvariable=self.vars["checkpoint"]).grid(row=0, column=1, sticky="w")
        ttk.Button(checkpoint, text="Establish / Reset", command=self.establish).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Label(checkpoint, text="Goto frame").grid(row=2, column=0, sticky="w")
        ttk.Entry(checkpoint, textvariable=self.goto_var, width=14).grid(row=2, column=1, sticky="ew")
        ttk.Button(checkpoint, text="Goto and Freeze", command=self.goto).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=3)

        recording = ttk.LabelFrame(outer, text="Recording / Validation", padding=8)
        recording.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for row, (label, name) in enumerate((
            ("Recorded frames", "recorded"), ("Dropped frames", "dropped"),
            ("Output session", "output"), ("Deterministic", "deterministic"),
            ("Last verified", "verified"), ("First divergent", "divergent"),
        )):
            ttk.Label(recording, text=label).grid(row=row, column=0, sticky="w")
            ttk.Label(recording, textvariable=self.vars[name], width=58).grid(row=row, column=1, sticky="w")
        ttk.Button(recording, text="Start", command=self.start_recording).grid(row=0, column=2, padx=3)
        ttk.Button(recording, text="Stop", command=self.stop_recording).grid(row=1, column=2, padx=3)

        live = ttk.LabelFrame(outer, text="Live State", padding=8)
        live.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for row, name in enumerate(("global", "p1", "p2")):
            ttk.Label(live, text=name.upper()).grid(row=row, column=0, sticky="nw")
            ttk.Label(live, textvariable=self.vars[name], width=86).grid(row=row, column=1, sticky="w")

        inputs = ttk.LabelFrame(outer, text="Logical Input", padding=8)
        inputs.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Combobox(inputs, textvariable=self.action_var, values=list(ACTION_INPUTS),
                     state="readonly", width=18).grid(row=0, column=0)
        ttk.Spinbox(inputs, from_=1, to=10000, textvariable=self.action_frames_var,
                    width=8).grid(row=0, column=1, padx=4)
        ttk.Button(inputs, text="Send", command=self.send_input).grid(row=0, column=2)
        ttk.Button(inputs, text="Release / Neutral", command=self.release).grid(row=0, column=3, padx=4)

    def _call(self, callback) -> None:
        if self.client is None:
            return
        try:
            callback()
        except (BridgeUnavailable, ValueError) as error:
            messagebox.showerror("Bridge command failed", str(error))

    def run(self) -> None:
        self._call(self.client.run if self.client else lambda: None)

    def pause(self) -> None:
        self._call(self.client.pause if self.client else lambda: None)

    def step(self, count: int) -> None:
        self._call(lambda: self.client.step(count))

    def step_back(self) -> None:
        self._call(lambda: self.client.step_back())

    def establish(self) -> None:
        self._call(self.client.establish_checkpoint if self.client else lambda: None)

    def goto(self) -> None:
        try:
            target = int(self.goto_var.get())
        except ValueError:
            messagebox.showerror("Invalid frame", "Goto frame must be a non-negative integer.")
            return
        self._call(lambda: self.client.goto_frame(target))

    def send_input(self) -> None:
        try:
            frames = int(self.action_frames_var.get())
        except ValueError:
            messagebox.showerror("Invalid duration", "Input duration must be an integer.")
            return
        self._call(lambda: self.client.send_action(self.action_var.get(), frames))

    def release(self) -> None:
        self._call(self.client.release if self.client else lambda: None)

    def start_recording(self) -> None:
        if self.client is None or self.writer is not None:
            return
        self.client.reset_ring()
        self.writer = RawSessionWriter(DATA_ROOT, {"bridge_abi": 2, "game": "th123 1.10a x86"})
        self.vars["output"].set(str(self.writer.path))

    def stop_recording(self) -> None:
        if self.writer is None:
            return
        snapshot = self.client.snapshot() if self.client else None
        self.writer.close(
            dropped_frames=snapshot.dropped_frames if snapshot else -1,
            validation=snapshot.deterministic_name if snapshot else "UNKNOWN",
        )
        self.writer = None

    def _disconnect(self) -> None:
        if self.writer is not None:
            self.writer.close(dropped_frames=-1, validation="UNKNOWN")
            self.writer = None
        if self.client is not None:
            self.client.close()
            self.client = None
        self.vars["connection"].set("Disconnected")
        self.vars["scene"].set("Other scene")

    def poll(self) -> None:
        try:
            if self.client is None:
                self.client = BridgeClient()
            snapshot = self.client.snapshot()
            frames = self.client.drain_frames()
            if self.writer is not None:
                self.writer.write(frames)
            state = snapshot.latest
            self.vars["connection"].set("Connected" if snapshot.connected else "Disconnected")
            self.vars["scene"].set("Practice Battle" if snapshot.in_gameplay else "Other scene")
            self.vars["run"].set(snapshot.run_state_name)
            self.vars["frame"].set(str(snapshot.game_frame))
            self.vars["checkpoint"].set("Valid (frame 0)" if snapshot.checkpoint_valid else "Invalid")
            self.vars["recorded"].set(str(self.writer.frame_count if self.writer else snapshot.recorded_frames))
            self.vars["dropped"].set(str(snapshot.dropped_frames))
            self.vars["deterministic"].set(snapshot.deterministic_name)
            self.vars["verified"].set("-" if snapshot.last_verified_frame is None else str(snapshot.last_verified_frame))
            self.vars["divergent"].set("-" if snapshot.first_divergent_frame is None else str(snapshot.first_divergent_frame))
            self.vars["result"].set(f"{snapshot.ack_seq}/{snapshot.command_seq} {snapshot.result_name}")
            self.vars["global"].set(
                f"scene={state.sceneId} mode={state.battleMode} stage={state.stageId} "
                f"round={state.roundId} weather={state.activeWeather}:{state.weatherCounter} hash={state.stateHash:016X}")
            for name, player in (("p1", state.p1), ("p2", state.p2)):
                self.vars[name].set(
                    f"char={player.characterId} pos=({player.x:.2f},{player.y:.2f}) "
                    f"vel=({player.speedX:.2f},{player.speedY:.2f}) hp={player.hp} sp={player.spirit} "
                    f"action={player.actionId}/{player.sequenceId}/{player.subsequenceId} frame={player.elapsedInSubsequence}")
        except (BridgeUnavailable, OSError, ValueError):
            self._disconnect()
        self.root.after(self.POLL_MS, self.poll)

    def close(self) -> None:
        self.stop_recording()
        if self.client is not None:
            self.client.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    DebugPanel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
