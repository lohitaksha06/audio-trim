#!/usr/bin/env python3
"""Template for downloading small evaluation subsets of the Audelle datasets.

This script is a TEMPLATE ONLY. It does not download anything.
Running it prints a warning and the TODOs below. Fill in the download logic
before using it for real.

Target: Python 3.12, standard library only.
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Dataset sources
# ---------------------------------------------------------------------------
# Full datasets are large. We only ever want a small evaluation subset.
MUSDB18_HQ_URL = "https://sigsep.github.io/datasets/musdb.html"  # ~30 GB total
DNS_CHALLENGE_URL = "https://github.com/microsoft/DNS-Challenge"  # hundreds of GB
NSYNTH_URL = "https://magenta.tensorflow.org/datasets/nsynth"  # ~30 GB total

# Suggested local layout (see ../datasets.md):
#   data/musdb/    small set of test tracks
#   data/dns/      a few clean/noisy pairs
#   data/nsynth/   a few hundred labelled notes


def download_musdb_subset() -> None:
    """Download a small MUSDB18-HQ evaluation subset.

    TODO: implement. Suggested approach:
      - fetch only the 7-second test clips, or 2-3 full test tracks
      - verify the checksum
      - unpack into data/musdb/
    """
    print("TODO: download a small subset here")


def download_dns_subset() -> None:
    """Download a small DNS Challenge evaluation subset.

    TODO: implement. Suggested approach:
      - fetch only the dev_testset clean/noisy pairs (tens of MB)
      - verify the checksum
      - unpack into data/dns/
    """
    print("TODO: download a small subset here")


def download_nsynth_subset() -> None:
    """Download a small NSynth evaluation subset.

    TODO: implement. Suggested approach:
      - fetch only a few hundred test notes
      - verify the checksum
      - unpack into data/nsynth/
    """
    print("TODO: download a small subset here")


def main() -> int:
    """Print the warning and the TODO placeholders. No network access."""
    print("WARNING: the full datasets are very large.")
    print("  MUSDB18-HQ      ~30 GB")
    print("  DNS Challenge   hundreds of GB")
    print("  NSynth          ~30 GB")
    print("This template does not download anything. Edit the functions below.")
    print()

    # These calls only print; they never touch the network.
    download_musdb_subset()
    download_dns_subset()
    download_nsynth_subset()

    print()
    print("Done (nothing was downloaded).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
