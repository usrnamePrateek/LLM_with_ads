from __future__ import annotations

import os
import sys
from pathlib import Path


def prepend_venv_bin_to_path() -> None:
    """vLLM/flashinfer JIT looks up the `ninja` binary on PATH, not via pip."""
    venv_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
