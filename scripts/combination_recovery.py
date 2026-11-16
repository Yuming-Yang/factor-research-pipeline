"""Composite IC of each combination method against the best possible, over five seeds.

    uv run python scripts/combination_recovery.py

Reproduces the table in DECISIONS.md (milestone 3). No delistings, so the planted ICs
are exact. Takes about 30 seconds.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from factorpipe.combine import METHODS, combine
from factorpipe.evaluation import information_coefficient
from factorpipe.ledger import TrialLedger
from factorpipe.synthetic import SyntheticConfig, make_panel

BURN_IN = 60


def main() -> None:
    results: dict[str, list[float]] = {method: [] for method in METHODS}
    with tempfile.TemporaryDirectory() as folder:
        ledger = TrialLedger(Path(folder) / "trials.jsonl")
        for seed in range(1, 6):
            config = SyntheticConfig(
                delist_prob=0.0, late_listing_share=0.0, reported_factor=None, seed=seed
            )
            synthetic = make_panel(config)
            panel = synthetic.panel
            for method in METHODS:
                scores = combine(panel, method, ledger=ledger).scores
                scored = panel.with_characteristic("composite", scores)
                ic = information_coefficient(scored, "composite", method="pearson")
                results[method].append(float(ic.iloc[BURN_IN:].mean()))
    truth = synthetic.truth
    print(f"best possible composite IC   {truth.best_ic:.3f}")
    equal = truth.equal_weight_ic(list(panel.characteristics))
    print(f"plain average of all factors {equal:.3f}\n")
    for method in sorted(METHODS, key=lambda m: np.mean(results[m])):
        print(f"  {method:<10} {np.mean(results[method]):.3f}")


if __name__ == "__main__":
    main()
