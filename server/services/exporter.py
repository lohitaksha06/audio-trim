"""Export pipeline: stems ZIP, FCPXML (Final Cut), EDL (Premiere/DaVinci).

All export functions write to storage and return a storage key; callers can
serve those keys through the download endpoint.
"""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import server.services.storage as storage


def export_zip(paths: list[str], prefix: str = "stems") -> dict[str, Any]:
    """Bundle a list of file paths into a ZIP in storage."""
    if not paths:
        raise ValueError("No files to zip")

    zip_path = Path(storage.STORAGE_DIR) / "exports" / f"{prefix}-{uuid4_hex()}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            name = Path(p).name
            zf.write(p, arcname=name)

    key = storage.store_local_file(str(zip_path), ext=".zip")
    return {"key": key, "path": str(zip_path), "files": [Path(p).name for p in paths]}


def uuid4_hex() -> str:
    import uuid
    return uuid.uuid4().hex


def _fmt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    f = int(round((t % 1) * 24))
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def _fmt_time_edl(t: float, fps: int = 30) -> str:
    total_frames = int(round(t * fps))
    h = total_frames // (3600 * fps)
    m = (total_frames // (60 * fps)) % 60
    s = (total_frames // fps) % 60
    f = total_frames % fps
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def export_fcpxml(segments: list[dict], duration: float, project_name: str = "Audelle Export") -> dict[str, Any]:
    """Generate a Final Cut Pro XML (FCPXML 1.8) document from labeled segments."""
    xml_esc = _xml_escape

    def asset_clip(seg: dict, idx: int) -> str:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", duration))
        label = xml_esc(str(seg.get("label", f"segment {idx}")))
        return (
            f'<asset-clip name="{label}" start="{start}s" duration="{end - start}s" '
            f'offset="{start}s" format="r1" enabled="1"/>'
        )

    timeline = "\n".join(asset_clip(s, i) for i, s in enumerate(segments))
    doc = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<fcpxml version="1.8">\n'
        f'  <resources>\n'
        f'    <format id="r1" name="FFVideoFormatRate30fps" frameDuration="1/30s" width="1920" height="1080"/>\n'
        f'    <asset id="a1" name="{xml_esc(project_name)}" start="0s" duration="{duration}s" '
        f'format="r1" hasVideo="1" hasAudio="1"/>\n'
        f'  </resources>\n'
        f'  <library>\n'
        f'    <event name="Audelle">\n'
        f'      <project name="{xml_esc(project_name)}">\n'
        f'        <sequence format="r1" duration="{duration}s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">\n'
        f'          <spine>\n{timeline}\n          </spine>\n'
        f'        </sequence>\n'
        f'      </project>\n'
        f'    </event>\n'
        f'  </library>\n'
        f'</fcpxml>\n'
    )
    return _write_xml(doc, "audelle-export.fcpxml")


def export_edl(segments: list[dict], duration: float, fps: int = 30) -> dict[str, Any]:
    """Generate a CMX3600 EDL (importable by Premiere / DaVinci Resolve)."""
    lines = ["TITLE: Audelle Export", "FCM: NON-DROP FRAME", ""]
    for i, seg in enumerate(segments, start=1):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", duration))
        name = str(seg.get("label", f"segment {i - 1}"))[:63]
        lines.append(f"{i:03d}  AX       V     C        {_fmt_time_edl(start, fps)} {_fmt_time_edl(end, fps)} {_fmt_time_edl(start, fps)} {_fmt_time_edl(end, fps)}")
        lines.append(f"* FROM CLIP NAME: {name}")
        lines.append("")
    doc = "\n".join(lines)
    return _write_text(doc, "audelle-export.edl")


def _xml_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _write_text(text: str, name: str) -> dict[str, Any]:
    path = Path(storage.STORAGE_DIR) / "exports" / f"{uuid4_hex()}-{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    key = storage.store_local_file(str(path), ext=path.suffix)
    return {"key": key, "path": str(path)}


def _write_xml(xml: str, name: str) -> dict[str, Any]:
    return _write_text(xml, name)