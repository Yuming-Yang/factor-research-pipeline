"""An append-only record of every configuration that was evaluated.

A Sharpe ratio means little without the number of things that were tried before it was
found. The ledger makes that number a by-product of running the pipeline: evaluation
functions take a ledger and write to it before they compute anything, and the deflated
Sharpe ratio reads the count back.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from factorpipe.panel import Panel


def fingerprint(panel: Panel) -> str:
    """A short hash of the returns grid, so trials on different data stay distinct."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(panel.returns.to_numpy(dtype=float)).tobytes())
    digest.update(str(panel.dates[0]).encode())
    digest.update(str(panel.dates[-1]).encode())
    digest.update("|".join(map(str, panel.assets)).encode())
    return digest.hexdigest()[:12]


@dataclass(frozen=True)
class TrialLedger:
    """A JSON-lines file with one line per distinct trial.

    A trial is identified by its method, its parameters and the data it ran on. Running
    the same trial again does not add a line, so re-running a script does not inflate
    the count, while changing any parameter does.
    """

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def record(self, method: str, params: Mapping[str, Any], data: str) -> str:
        """Record a trial and return its identifier."""
        entry = {"method": method, "params": _plain(params), "data": data}
        canonical = json.dumps(entry, sort_keys=True)
        trial_id = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        if trial_id not in {t["id"] for t in self.trials()}:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": trial_id, **entry}, sort_keys=True) + "\n")
        return trial_id

    def record_result(self, trial_id: str, **metrics: float) -> None:
        """Attach the outcome of a trial, for example its Sharpe ratio.

        Results are appended as their own lines, so the trial lines never change. If a
        trial is run again, the latest result is the one that is read back.
        """
        if trial_id not in {t["id"] for t in self.trials()}:
            raise KeyError(f"no trial {trial_id!r} in the ledger")
        entry = {"result_for": trial_id, "metrics": _plain(metrics)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def trials(self) -> list[dict[str, Any]]:
        return [line for line in self._lines() if "method" in line]

    def results(self) -> dict[str, dict[str, Any]]:
        """Latest recorded metrics by trial identifier."""
        return {
            line["result_for"]: line["metrics"] for line in self._lines() if "result_for" in line
        }

    def _lines(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @property
    def n_trials(self) -> int:
        return len(self.trials())


def _plain(value: Any) -> Any:
    """Convert parameters to JSON types with a stable form."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"parameter of type {type(value).__name__} cannot be recorded")
