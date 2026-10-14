"""Explicit supervision masks preserve complete episode prefixes for recurrent memory."""
import numpy as np


def supervision_mask(rows):
    flags = []
    for row in rows:
        if len(row) not in (4, 5) or (len(row) == 5 and not isinstance(row[4], (bool, np.bool_))):
            raise ValueError("demonstration rows require four legacy fields and an optional boolean supervision flag")
        flags.append(True if len(row) == 4 else bool(row[4]))
    return np.asarray(flags, dtype=np.bool_)


def supervised_samples(rows):
    return [row for row, valid in zip(rows, supervision_mask(rows), strict=True) if valid]
