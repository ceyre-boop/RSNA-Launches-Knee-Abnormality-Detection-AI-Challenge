"""Kaggle notebook entry point: train a single fold.

Paste into a notebook cell:

    !python /kaggle/working/repo/kaggle/train_fold.py --fold 0 --epochs 10

Any argument may instead be supplied as an environment variable using the
upper-cased flag name prefixed with ``RSNA_`` (e.g. ``RSNA_FOLD=1``), which is
convenient when driving several notebook versions from the same cell.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rsna_knee.imaging import train as train_module  # noqa: E402

ENV_PREFIX = "RSNA_"


def argv_from_env(argv: list[str]) -> list[str]:
    """Append ``--flag value`` pairs for RSNA_* variables not already on argv."""
    parser = train_module.build_parser()
    known = {
        action.dest: action.option_strings[0]
        for action in parser._actions  # noqa: SLF001 - argparse has no public accessor
        if action.option_strings
    }
    extended = list(argv)
    for dest, flag in known.items():
        if flag in extended:
            continue
        value = os.environ.get(f"{ENV_PREFIX}{dest.upper()}")
        if value is None:
            continue
        if value.lower() in {"true", "false"}:
            if value.lower() == "true":
                extended.append(flag)
            continue
        extended.extend([flag, value])
    return extended


def main() -> int:
    return train_module.main(argv_from_env(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
