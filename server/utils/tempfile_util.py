"""Collision-safe temporary path allocation.

`tempfile.mktemp` is deprecated and unsafe: it only *proposes* a name, so two
concurrent requests can be handed the same path and one overwrites the other's
output. Audelle runs heavy jobs on a thread pool
(`server/services/jobs.py`), so that race is reachable in normal use.

`temp_output_path` reserves a genuinely unique name via `mkstemp`, then closes
and removes the placeholder file so callers can hand the path straight to an
audio writer (librosa/soundfile/ffmpeg all overwrite happily).
"""

from __future__ import annotations

import os
import tempfile


def temp_output_path(suffix: str = ".wav") -> str:
    """Return an unused absolute path ending in ``suffix``.

    The path does not exist when this returns; it is reserved, not created, so
    concurrent calls never collide.
    """
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        os.remove(path)
    except OSError:
        # Another process claimed it between close and unlink. The name is still
        # effectively ours because mkstemp guarantees exclusivity, so keep it.
        pass
    return path
