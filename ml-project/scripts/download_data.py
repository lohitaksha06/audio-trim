#!/usr/bin/env python3
"""Download the small Audelle audio subsets. Standard library only.

Nothing is downloaded unless you ask for it explicitly:

    python scripts/download_data.py                      # help + known sizes
    python scripts/download_data.py --source demand --check
    python scripts/download_data.py --source demand --file OHALLWAY_16k
    python scripts/download_data.py --source voicebank --file test-16k

Files land in ``ml-project/data/<source>/``. Sizes are exact bytes taken from
the providers' metadata APIs on 2026-11-28 (see ../datasets.md, "Download
sources"). Every download is size-checked and hashed.

A safety cap (``--max-bytes``, default 250 MB) stops an accidental multi-GB
pull. Pass ``--allow-large`` only when you really mean it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DATASHARE = "https://datashare.ed.ac.uk"
HF_VB16K = "https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k/resolve/main"
ZENODO = "https://zenodo.org/records/1227121/files"

DEFAULT_MAX_BYTES = 250 * 1024 * 1024
CHUNK = 256 * 1024
UA = "audelle-download/1.0 (+https://github.com/audelle)"


def ds_bitstream(uuid: str, name: str, size: int, note: str) -> dict:
    """A University of Edinburgh DataShare bitstream."""
    return {
        "file": name,
        "url": f"{DATASHARE}/bitstreams/{uuid}/download",
        "bytes": size,
        "note": note,
    }


SOURCES: dict[str, dict] = {
    "voicebank": {
        "title": "VoiceBank-DEMAND clean/noisy speech",
        "home": "https://datashare.ed.ac.uk/handle/10283/2791",
        "license": "CC BY 4.0",
        "default": "test-16k",
        "notes": [
            "The DataShare record is 48 kHz only; the 16 kHz files are a "
            "community resample hosted on Hugging Face.",
            "DataShare publishes no sha256, only sizes. Zenodo publishes md5.",
        ],
        "files": {
            # --- 16 kHz (recommended) -------------------------------------
            "test-16k": {
                "file": "data/test-00000-of-00001.parquet",
                "url": f"{HF_VB16K}/data/test-00000-of-00001.parquet",
                "bytes": 132_200_447,
                "note": "SMALLEST usable 16 kHz subset: 824 pairs, 2 unseen "
                        "speakers, clean+noisy audio columns.",
            },
            "train-16k-shard-4": {
                "file": "data/train-00004-of-00005.parquet",
                "url": f"{HF_VB16K}/data/train-00004-of-00005.parquet",
                "bytes": 415_510_617,
                "note": "Smallest train shard (~2,314 pairs). Needs "
                        "--allow-large.",
            },
            "train-16k-shard-0": {
                "file": "data/train-00000-of-00005.parquet",
                "url": f"{HF_VB16K}/data/train-00000-of-00005.parquet",
                "bytes": 444_302_320,
                "note": "Train shard 0/5 (~2,314 pairs).",
            },
            "train-16k-shard-1": {
                "file": "data/train-00001-of-00005.parquet",
                "url": f"{HF_VB16K}/data/train-00001-of-00005.parquet",
                "bytes": 416_366_498,
                "note": "Train shard 1/5 (~2,314 pairs).",
            },
            "train-16k-shard-2": {
                "file": "data/train-00002-of-00005.parquet",
                "url": f"{HF_VB16K}/data/train-00002-of-00005.parquet",
                "bytes": 430_855_381,
                "note": "Train shard 2/5 (~2,314 pairs).",
            },
            "train-16k-shard-3": {
                "file": "data/train-00003-of-00005.parquet",
                "url": f"{HF_VB16K}/data/train-00003-of-00005.parquet",
                "bytes": 444_005_802,
                "note": "Train shard 3/5 (~2,314 pairs).",
            },
            # --- 48 kHz originals (only if you want the pristine source) ---
            "clean-testset-48k": ds_bitstream(
                "dec213d3-bf57-4777-9663-c24bdce92d5e",
                "clean_testset_wav.zip", 154_332_064,
                "48 kHz clean test set, 824 utterances.",
            ),
            "clean-trainset-28spk-48k": ds_bitstream(
                "245452b6-6235-44b6-a6f9-e7eb19797769",
                "clean_trainset_28spk_wav.zip", 2_486_057_279,
                "48 kHz clean train set, 28 speakers. Needs --allow-large.",
            ),
        },
    },
    "demand": {
        "title": "DEMAND environmental noise",
        "home": "https://zenodo.org/records/1227121",
        "license": "CC BY 4.0",
        "default": "OHALLWAY_16k",
        "notes": [
            "17 of the 18 environments have a 16 kHz build; SCAFE is 48 kHz "
            "only.",
            "Zenodo publishes md5 per file, not sha256.",
        ],
        "files": {
            "OHALLWAY_16k": {"file": "OHALLWAY_16k.zip",
                             "url": f"{ZENODO}/OHALLWAY_16k.zip?download=1",
                             "bytes": 77_941_854,
                             "note": "SMALLEST 16 kHz noise. Quiet hall."},
            "DLIVING_16k": {"file": "DLIVING_16k.zip",
                            "url": f"{ZENODO}/DLIVING_16k.zip?download=1",
                            "bytes": 80_170_627, "note": "Domestic living room."},
            "OMEETING_16k": {"file": "OMEETING_16k.zip",
                             "url": f"{ZENODO}/OMEETING_16k.zip?download=1",
                             "bytes": 82_703_355, "note": "Office meeting room."},
            "NPARK_16k": {"file": "NPARK_16k.zip",
                          "url": f"{ZENODO}/NPARK_16k.zip?download=1",
                          "bytes": 86_029_332, "note": "Public park."},
            "NFIELD_16k": {"file": "NFIELD_16k.zip",
                           "url": f"{ZENODO}/NFIELD_16k.zip?download=1",
                           "bytes": 86_536_782, "note": "Nature field."},
            "OOFFICE_16k": {"file": "OOFFICE_16k.zip",
                            "url": f"{ZENODO}/OOFFICE_16k.zip?download=1",
                            "bytes": 88_995_191, "note": "Office, quiet."},
            "NRIVER_16k": {"file": "NRIVER_16k.zip",
                           "url": f"{ZENODO}/NRIVER_16k.zip?download=1",
                           "bytes": 98_689_565, "note": "Riverbank."},
            "DWASHING_16k": {"file": "DWASHING_16k.zip",
                             "url": f"{ZENODO}/DWASHING_16k.zip?download=1",
                             "bytes": 102_343_250, "note": "Washing machine."},
            "PCAFETER_16k": {"file": "PCAFETER_16k.zip",
                             "url": f"{ZENODO}/PCAFETER_16k.zip?download=1",
                             "bytes": 107_431_494, "note": "Cafeteria."},
            "DKITCHEN_16k": {"file": "DKITCHEN_16k.zip",
                             "url": f"{ZENODO}/DKITCHEN_16k.zip?download=1",
                             "bytes": 110_501_049, "note": "Kitchen."},
            "SPSQUARE_16k": {"file": "SPSQUARE_16k.zip",
                             "url": f"{ZENODO}/SPSQUARE_16k.zip?download=1",
                             "bytes": 110_928_528, "note": "Public square."},
            "PRESTO_16k": {"file": "PRESTO_16k.zip",
                           "url": f"{ZENODO}/PRESTO_16k.zip?download=1",
                           "bytes": 111_292_815, "note": "Restaurant."},
            "STRAFFIC_16k": {"file": "STRAFFIC_16k.zip",
                             "url": f"{ZENODO}/STRAFFIC_16k.zip?download=1",
                             "bytes": 118_572_691, "note": "Street traffic."},
            "PSTATION_16k": {"file": "PSTATION_16k.zip",
                             "url": f"{ZENODO}/PSTATION_16k.zip?download=1",
                             "bytes": 119_441_094, "note": "Public station."},
            "TMETRO_16k": {"file": "TMETRO_16k.zip",
                           "url": f"{ZENODO}/TMETRO_16k.zip?download=1",
                           "bytes": 126_589_090, "note": "Metro."},
            "TBUS_16k": {"file": "TBUS_16k.zip",
                         "url": f"{ZENODO}/TBUS_16k.zip?download=1",
                         "bytes": 128_916_709, "note": "Bus interior."},
            "TCAR_16k": {"file": "TCAR_16k.zip",
                         "url": f"{ZENODO}/TCAR_16k.zip?download=1",
                         "bytes": 130_045_975, "note": "Car interior."},
            "DEMAND_pdf": {"file": "DEMAND.pdf",
                           "url": f"{ZENODO}/DEMAND.pdf?download=1",
                           "bytes": 87_837, "note": "Dataset paper."},
        },
    },
}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def human(n: int) -> str:
    """Render a byte count in decimal units, matching the providers' labels."""
    if n < 1000:
        return f"{n} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n_f = n / 1000.0
        if n_f < 1000 or unit == "TB":
            return f"{n_f:.2f} {unit}" if unit != "KB" else f"{n_f:.1f} {unit}"
        n = int(n_f)
    return f"{n} B"


def print_help() -> None:
    """Print usage plus the known sizes for every source."""
    print(__doc__.strip())
    print()
    print("Known sources (exact bytes, verified 2026-11-28):")
    for key, src in SOURCES.items():
        total = sum(f["bytes"] for f in src["files"].values())
        print(f"\n  {key}  -  {src['title']}")
        print(f"    license : {src['license']}")
        print(f"    home    : {src['home']}")
        print(f"    catalog : {len(src['files'])} files, "
              f"{human(total)} ({total:,} bytes) if you took them all")
        print(f"    default : {src['default']} "
              f"({human(src['files'][src['default']]['bytes'])})")
        for name, f in src["files"].items():
            star = "*" if name == src["default"] else " "
            print(f"      {star} {name:<24} {human(f['bytes']):>9}  "
                  f"{f['file']}")
        for note in src["notes"]:
            print(f"    note: {note}")
    print("\nNothing was downloaded. Add --source <name> to fetch a file.")


def print_list(source: str) -> None:
    src = SOURCES[source]
    print(f"{source}: {src['title']}  [{src['license']}]")
    for name, f in src["files"].items():
        print(f"  {name:<24} {human(f['bytes']):>9}  {f['file']}")
        print(f"  {'':<24} {f['url']}")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(
        url, method=method,
        headers={"User-Agent": UA, "Accept": "*/*"},
    )


def head_size(url: str) -> tuple[int | None, str]:
    """Return (content_length, status_text). HEAD only, never a full GET.

    Some hosts ignore HEAD or answer with a redirect stub, so fall back to a
    one-byte ranged GET and read ``Content-Range``.
    """
    try:
        with urllib.request.urlopen(_request(url, "HEAD"), timeout=60) as r:
            length = r.headers.get("Content-Length")
            ctype = r.headers.get("Content-Type", "")
            if length and int(length) > 8192:
                return int(length), f"{r.status} {ctype.split(';')[0]}"
    except urllib.error.HTTPError as e:
        if e.code not in (405, 501, 403):
            return None, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001 - report, never crash a --check
        return None, type(e).__name__

    # Fallback: ask for one byte and read the total from Content-Range.
    req = _request(url)
    req.add_header("Range", "bytes=0-0")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                return int(cr.rsplit("/", 1)[1]), f"{r.status} ranged"
            return None, "no length exposed"
    except Exception as e:  # noqa: BLE001
        return None, f"unreachable ({type(e).__name__})"


def check_source(source: str, names: list[str]) -> int:
    """HEAD every selected file and compare against the catalog."""
    src = SOURCES[source]
    print(f"checking {source} ({src['license']}) - HEAD requests only\n")
    bad = 0
    for name in names:
        f = src["files"][name]
        print(f"  {name:<24} {f['file']}")
        print(f"    url      {f['url']}")
        print(f"    catalog  {f['bytes']:,} bytes ({human(f['bytes'])})")
        size, status = head_size(f["url"])
        if size is None:
            print(f"    server   {status} - size not confirmed")
            bad += 1
        elif size == f["bytes"]:
            print(f"    server   {size:,} bytes - matches catalog")
        else:
            print(f"    server   {size:,} bytes - MISMATCH "
                  f"({size - f['bytes']:+,} bytes)")
            bad += 1
    print()
    print("Nothing was downloaded." if bad == 0
          else f"{bad} file(s) could not be confirmed; nothing was downloaded.")
    return 0 if bad == 0 else 1


def download(source: str, name: str, *, force: bool,
             max_bytes: int) -> int:
    """Download one catalog file into data/<source>/, size-checked."""
    src = SOURCES[source]
    f = src["files"][name]
    size = f["bytes"]
    if size > max_bytes:
        print(f"refusing: {f['file']} is {human(size)}, above the "
              f"{human(max_bytes)} cap. Pass --allow-large or raise "
              f"--max-bytes.", file=sys.stderr)
        return 2

    dest_dir = DATA_DIR / source
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(f["file"]).name
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and not force:
        got = dest.stat().st_size
        print(f"already present: {dest} ({human(got)}) - use --force to "
              f"re-download.")
        return 0
    if part.exists() and not force:
        print(f"removing stale partial file {part.name}")
        part.unlink()

    print(f"downloading {f['file']}")
    print(f"  from  {f['url']}")
    print(f"  to    {dest}")
    print(f"  size  {human(size)} ({size:,} bytes)")

    digest = hashlib.sha256()
    done = 0
    next_mark = 0
    try:
        with urllib.request.urlopen(_request(f["url"]), timeout=120) as r, \
                open(part, "wb") as out:
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if size and done * 100 // size >= next_mark:
                    pct = done * 100 / size
                    print(f"    {pct:5.1f}%  {human(done)} / {human(size)}")
                    next_mark = int(pct) + 20
    except Exception as e:  # noqa: BLE001
        print(f"download failed: {type(e).__name__}: {e}", file=sys.stderr)
        if part.exists():
            print(f"  kept {part.name} for inspection", file=sys.stderr)
        return 1

    if done != size:
        print(f"size mismatch: got {done:,} bytes, catalog says {size:,}. "
              f"Leaving {part.name} in place.", file=sys.stderr)
        return 1

    part.replace(dest)
    print(f"  done  {human(done)}")
    print(f"  sha256 {digest.hexdigest()}")
    print(f"  saved {dest}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_data.py",
        description="Download small Audelle audio subsets (stdlib only). "
                    "With no arguments it prints help and known sizes.",
        epilog="Run with no arguments for the size table.",
    )
    p.add_argument("--source", choices=sorted(SOURCES),
                   help="which dataset to fetch")
    p.add_argument("--file", action="append", metavar="NAME",
                   help="catalog entry to fetch (repeatable); default is the "
                        "smallest usable subset")
    p.add_argument("--check", action="store_true",
                   help="HEAD only: print sizes, download nothing")
    p.add_argument("--list", action="store_true",
                   help="print the catalog for --source")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing file")
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                   metavar="N", help=f"safety cap (default {DEFAULT_MAX_BYTES:,})")
    p.add_argument("--allow-large", action="store_true",
                   help="lift the size cap for this run")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not argv:
        print_help()
        return 0

    if not args.source:
        print_help()
        return 0

    src = SOURCES[args.source]
    names = args.file or [src["default"]]
    unknown = [n for n in names if n not in src["files"]]
    if unknown:
        print(f"unknown file(s) for {args.source}: {', '.join(unknown)}\n",
              file=sys.stderr)
        print_list(args.source)
        return 2

    if args.list:
        print_list(args.source)
        return 0

    if args.check:
        return check_source(args.source, names)

    max_bytes = sys.maxsize if args.allow_large else args.max_bytes
    rc = 0
    for name in names:
        rc |= download(args.source, name, force=args.force,
                       max_bytes=max_bytes)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
