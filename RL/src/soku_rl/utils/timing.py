"""Frame pacing helpers."""


def frame_duration(fps: float = 60.0) -> float:
    """Return the duration of one frame in seconds."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return 1.0 / fps
