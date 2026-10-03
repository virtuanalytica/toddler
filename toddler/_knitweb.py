"""Single place that locates knitweb (review issue #1, blocker 1).

Order: an installed `knitweb` package; the TODDLER_KNITWEB_SRC environment variable; a sibling
checkout next to this repository (`../knitweb/src`, meaningful only for a source checkout of
toddler; for an installed toddler the environment variable is the way). No machine-specific path
is hard-coded and nothing touches sys.path until load() is called.

load() raises ModuleNotFoundError (not a plain ImportError) so pytest.importorskip skips cleanly.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

_SIBLING = Path(__file__).resolve().parents[2] / "knitweb" / "src"


def load() -> None:
    """Make `import knitweb` work, or raise ModuleNotFoundError explaining how to provide it."""
    try:
        importlib.import_module("knitweb")
        return
    except ImportError:
        pass
    for candidate in (os.environ.get("TODDLER_KNITWEB_SRC", ""), str(_SIBLING)):
        if candidate and (Path(candidate) / "knitweb").is_dir():
            if candidate not in sys.path:
                sys.path.append(candidate)
            importlib.import_module("knitweb")
            return
    raise ModuleNotFoundError("knitweb not found: install it, set TODDLER_KNITWEB_SRC to its src/ directory, "
                      "or check it out next to this repository as ../knitweb")
