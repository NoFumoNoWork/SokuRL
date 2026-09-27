from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from bridge_shared import (
    ACTION_INPUTS,
    MAX_DURATION_FRAMES,
    BridgeClient,
    BridgeUnavailable,
)


class ControlPanel:
    POLL_MS = 100

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SokuRL Control Panel")
        self.root.resizable(False, False)
        self.client: BridgeClient | None = None
        self.last_submitted = "None"

        self.connection_var = tk.StringVar(value="Disconnected")
        self.scene_var = tk.StringVar(value="Not in Practice battle")
        self.frame_var = tk.StringVar(value="0")
        self.remaining_var = tk.StringVar(value="0")
        self.ack_var = tk.StringVar(value="-")
        self.last_var = tk.StringVar(value=self.last_submitted)
        self.action_var = tk.StringVar(value="RIGHT")
        self.frames_var = tk.StringVar(value="120")

        outer = ttk.Frame(root, padding=14)
        outer.grid(sticky="nsew")

        status = ttk.LabelFrame(outer, text="Bridge status", padding=10)
        status.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._status_row(status, 0, "Connection", self.connection_var)
        self._status_row(status, 1, "Scene", self.scene_var)
        self._status_row(status, 2, "Game frame", self.frame_var)
        self._status_row(status, 3, "Remaining", self.remaining_var)
        self._status_row(status, 4, "Acknowledgement", self.ack_var)
        self._status_row(status, 5, "Last submitted", self.last_var)

        command = ttk.LabelFrame(outer, text="Command", padding=10)
        command.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Label(command, text="Action").grid(row=0, column=0, sticky="w")
        self.action_box = ttk.Combobox(
            command,
            textvariable=self.action_var,
            values=list(ACTION_INPUTS),
            state="readonly",
            width=20,
        )
        self.action_box.grid(row=0, column=1, padx=(12, 0), sticky="ew")

        ttk.Label(command, text="Frames").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.frames_spinbox = ttk.Spinbox(
            command,
            from_=1,
            to=MAX_DURATION_FRAMES,
            textvariable=self.frames_var,
            width=12,
        )
        self.frames_spinbox.grid(row=1, column=1, padx=(12, 0), pady=(8, 0), sticky="w")

        buttons = ttk.Frame(outer)
        buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.send_button = ttk.Button(buttons, text="Send", command=self.send)
        self.send_button.grid(row=0, column=0, sticky="ew")
        self.release_button = ttk.Button(buttons, text="Release / Neutral", command=self.release)
        self.release_button.grid(row=0, column=1, padx=(8, 0), sticky="ew")
        buttons.columnconfigure((0, 1), weight=1)

        self._set_controls_enabled(False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(0, self.poll)

    @staticmethod
    def _status_row(parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, textvariable=variable, width=34).grid(row=row, column=1, padx=(12, 0), sticky="w")

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.send_button.configure(state=state)
        self.release_button.configure(state=state)
        self.frames_spinbox.configure(state=state)
        self.action_box.configure(state="readonly" if enabled else "disabled")

    def _disconnect(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        self.connection_var.set("Disconnected")
        self.scene_var.set("Not in Practice battle")
        self._set_controls_enabled(False)

    def poll(self) -> None:
        if self.client is None:
            try:
                self.client = BridgeClient()
            except BridgeUnavailable:
                self.root.after(self.POLL_MS, self.poll)
                return

        try:
            snapshot = self.client.snapshot()
            if not snapshot.connected:
                self._disconnect()
            else:
                self.connection_var.set("Connected")
                self.scene_var.set("Practice battle" if snapshot.in_gameplay else "Not in Practice battle")
                self.frame_var.set(str(snapshot.game_frame))
                self.remaining_var.set(str(snapshot.frames_remaining))
                self.ack_var.set(
                    f"{snapshot.ack_seq}/{snapshot.command_seq} ({snapshot.result_name})"
                )
                self._set_controls_enabled(True)
        except (BridgeUnavailable, OSError, ValueError):
            self._disconnect()
        self.root.after(self.POLL_MS, self.poll)

    def send(self) -> None:
        if self.client is None:
            return
        try:
            frames = int(self.frames_var.get())
            if not 1 <= frames <= MAX_DURATION_FRAMES:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid frames", f"Frames must be an integer from 1 to {MAX_DURATION_FRAMES}.")
            return

        action = self.action_var.get()
        try:
            sequence = self.client.send_action(action, frames)
        except (BridgeUnavailable, ValueError) as error:
            messagebox.showerror("Command failed", str(error))
            self._disconnect()
            return
        self.last_submitted = f"#{sequence}: {action} for {frames} frames"
        self.last_var.set(self.last_submitted)

    def release(self) -> None:
        if self.client is None:
            return
        try:
            sequence = self.client.release()
        except BridgeUnavailable as error:
            messagebox.showerror("Release failed", str(error))
            self._disconnect()
            return
        self.last_submitted = f"#{sequence}: RELEASE"
        self.last_var.set(self.last_submitted)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ControlPanel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
