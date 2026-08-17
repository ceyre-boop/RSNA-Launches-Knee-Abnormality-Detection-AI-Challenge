"""Noisy-student self-training loop with a statistically sound referee.

Four pieces, mapped to the round structure in ``Plans/greedy-shimmying-mochi.md``:

- ``gold_split``   deterministic working-40 / locked-18 split of the gold studies
- ``referee``      CV-primary gate, advisory bootstrap gold check, final check
- ``corrections``  per-label Confident Learning with caps and co-drift guards
- ``round_runner`` CLI that runs one local round end to end
"""

from __future__ import annotations

__all__ = ["gold_split", "referee", "corrections", "round_runner"]
