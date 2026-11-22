"""Upload cap, rate limiting, storage cleanup and health."""

import os
import time

from fastapi.testclient import TestClient

from server.main import app
from server.services import limits, storage


def test_upload_rejects_files_over_the_cap(monkeypatch):
    monkeypatch.setenv("AUDIDELLE_MAX_UPLOAD_MB", "0.0001")  # ~104 bytes
    client = TestClient(app)
    payload = b"x" * 1000
    response = client.post("/api/upload", files={"file": ("big.wav", payload, "audio/wav")})
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.setenv("AUDIDELLE_RATE_LIMIT", "2")
    limits.reset()
    client = TestClient(app)
    codes = [
        client.post("/api/suggest", json={"prompt": "make it louder"}).status_code
        for _ in range(3)
    ]
    assert codes[:2] == [200, 200]
    assert codes[2] == 429
    limits.reset()


def test_cleanup_removes_only_old_files(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "STORAGE_DIR", str(tmp_path))
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    old = uploads / "old.wav"
    new = uploads / "new.wav"
    old.write_bytes(b"a" * 10)
    new.write_bytes(b"b" * 10)
    past = time.time() - 3 * 3600
    os.utime(old, (past, past))

    result = storage.cleanup(1.0)
    assert result["removed"] == 1
    assert not old.exists()
    assert new.exists()


def test_cleanup_disabled_at_zero_hours(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_DIR", str(tmp_path))
    (tmp_path / "uploads").mkdir()
    old = tmp_path / "uploads" / "old.wav"
    old.write_bytes(b"a")
    past = time.time() - 99 * 3600
    os.utime(old, (past, past))
    assert storage.cleanup(0)["removed"] == 0
    assert old.exists()


def test_health_reports_storage_and_limits(monkeypatch):
    monkeypatch.setenv("AUDIDELLE_RATE_LIMIT", "60")
    client = TestClient(app)
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "storage" in body
    assert body["rate_limit_per_min"] == 60
    assert body["uptime_seconds"] >= 0
