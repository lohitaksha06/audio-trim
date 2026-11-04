"""Shared paths for scripts. Adds ml/src to sys.path so the package runs
without an editable install (handy on a locked-down school laptop)."""

from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC = ML_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA = ML_ROOT / "data"
SPLITS = DATA / "splits"
REPORTS = ML_ROOT / "reports"
MODELS = ML_ROOT / "models"
