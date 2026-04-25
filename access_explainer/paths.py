"""Resolve project roots and extend sys.path for HyConEx imports."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HYCONEX_ROOT = _PROJECT_ROOT / "HyConEx"
_VENDOR_HYPERLOGIC_CODE = (
    _PROJECT_ROOT / "vendor" / "hyperlogic_official" / "code"
)


def project_root() -> Path:
    return _PROJECT_ROOT


def ensure_hyconex_path() -> None:
    p = str(_HYCONEX_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def ensure_vendor_hyperlogic_code_path() -> None:
    p = str(_VENDOR_HYPERLOGIC_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)
