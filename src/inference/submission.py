"""ARC-AGI submission helpers."""

from __future__ import annotations

import json
from pathlib import Path


def two_attempt_record(attempts: list[list[list[int]]]) -> dict:
    """Format top-2 predictions according to ARC-AGI submission conventions."""

    if not attempts:
        raise ValueError("at least one attempt is required")
    first = attempts[0]
    second = attempts[1] if len(attempts) > 1 else attempts[0]
    return {"attempt_1": first, "attempt_2": second}


def write_submission(predictions: dict[str, list[list[list[int]]]], path: str | Path) -> Path:
    """Write a JSON submission from ``task_id -> [attempt1, attempt2]`` predictions."""

    payload = {task_id: [two_attempt_record(attempts)] for task_id, attempts in predictions.items()}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out
