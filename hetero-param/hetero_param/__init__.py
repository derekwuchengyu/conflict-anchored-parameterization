"""hetero_param — configurable heterogeneous scenario parameterization + fidelity analysis.

Import side effect: make the wrapped existing repos importable.
"""
import sys
from pathlib import Path

# Ensure paths.py (one level up) and the existing repos are importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths as paths  # noqa: E402  (project path config)
paths.add_import_paths()

__all__ = ["paths"]
