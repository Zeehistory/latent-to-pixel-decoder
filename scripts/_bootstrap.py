"""Make ``src`` importable when running scripts directly (``python scripts/foo.py``).

Installed usage (``pip install -e .``) does not need this, but it keeps the scripts runnable from a
fresh checkout without installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
