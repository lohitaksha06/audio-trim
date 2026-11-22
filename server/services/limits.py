"""Cheap in-process rate limiting and request timing.

No Redis, no accounts, no billing. One process keeps a sliding window per
client key. That is enough to stop a single client from hammering the heavy
upload/process endpoints; a multi-instance deployment should move this to the
proxy or Redis. `AUDIDELLE_RATE_LIMIT=0` disables it (used by tests).
"""

from __future__ import annotations

import os
import threading
import time

_LOCK = threading.Lock()
_HITS: dict[str, list[float]] = {}
WINDOW_SECONDS = 60.0


def limit_per_minute() -> int:
    try:
        return max(0, int(os.environ.get("AUDIDELLE_RATE_LIMIT", "120")))
    except ValueError:
        return 120


def client_key(headers, client) -> str:
    forwarded = headers.get("x-forwarded-for") if headers else None
    if forwarded:
        return forwarded.split(",")[0].strip()
    host = getattr(client, "host", None) if client else None
    return host or "unknown"


def check(key: str) -> tuple[bool, int]:
    """Return ``(allowed, retry_after_seconds)``."""
    limit = limit_per_minute()
    if limit <= 0:
        return True, 0
    now = time.monotonic()
    with _LOCK:
        hits = [t for t in _HITS.get(key, []) if now - t < WINDOW_SECONDS]
        if len(hits) >= limit:
            retry = max(1, int(WINDOW_SECONDS - (now - hits[0])) + 1)
            _HITS[key] = hits
            return False, retry
        hits.append(now)
        _HITS[key] = hits
        return True, 0


def reset() -> None:
    """Clear all windows. Used by tests."""
    with _LOCK:
        _HITS.clear()
