"""Unit tests for Phase 2 services: storage, exporter, jobs."""

import tempfile
import zipfile

import server.services.storage as storage
from server.services.exporter import export_edl, export_fcpxml, export_zip
from server.services.jobs import job_manager


class TestStorage:
    def test_roundtrip(self):
        key = storage.save_upload("x.bin", b"payload")
        path = storage.resolve(key)
        assert open(path, "rb").read() == b"payload"

    def test_delete(self):
        key = storage.save_upload("y.bin", b"data")
        assert storage.delete(key) is True


class TestExporter:
    def test_stems_zip(self):
        tf = tempfile.mktemp(suffix=".wav")
        open(tf, "wb").write(b"stem-audio")
        res = export_zip([tf])
        assert res["files"] == [tf.split("\\")[-1]]
        with zipfile.ZipFile(res["path"]) as zf:
            assert len(zf.namelist()) == 1

    def test_fcpxml(self):
        res = export_fcpxml(
            [{"start": 0, "end": 2, "label": "Intro"}, {"start": 2, "end": 5, "label": "Chorus"}],
            5.0,
        )
        xml = open(res["path"], encoding="utf-8").read()
        assert "<fcpxml version=\"1.8\">" in xml
        assert "Intro" in xml and "Chorus" in xml

    def test_edl(self):
        res = export_edl([{"start": 0, "end": 2, "label": "Intro"}], 5.0)
        edl = open(res["path"], encoding="utf-8").read()
        assert edl.startswith("TITLE: Audelle Export")
        assert "FROM CLIP NAME: Intro" in edl


class TestJobs:
    def test_run_and_complete(self):
        job_id = job_manager.submit(lambda a, b: a + b, 2, 3)
        job = job_manager.get(job_id)
        assert job is not None
        # poll briefly
        for _ in range(100):
            job = job_manager.get(job_id)
            if job.status in {"completed", "failed"}:
                break
            import time
            time.sleep(0.01)
        assert job.status == "completed"
        assert job.result == 5

    def test_missing_job(self):
        assert job_manager.get("does-not-exist") is None