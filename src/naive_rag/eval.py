"""ragas-based quantitative evaluation.

Phase 2 — not implemented in this plan. The interface below declares the
intended entry point so callers can wire up dataset generation later.
"""

from typing import Any


def run_ragas_eval(*args: Any, **kwargs: Any) -> dict:
    raise NotImplementedError(
        "Ragas evaluation is a Phase 2 deliverable. See docs/design.md section 7."
    )
