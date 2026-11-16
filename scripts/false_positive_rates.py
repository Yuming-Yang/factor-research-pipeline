"""Measure how often pure-noise factors are flagged, before and after adjustment.

Reproduces the table in DECISIONS.md (milestone 2):

    uv run python scripts/false_positive_rates.py

Each panel has one planted factor and 25 noise factors. Only the noise factors are
tested, so every rejection is a false positive.
"""

from __future__ import annotations

import argparse

import numpy as np

from factorpipe.evaluation import information_coefficient
from factorpipe.stats import benjamini_yekutieli, holm, newey_west_mean
from factorpipe.synthetic import PlantedFactor, SyntheticConfig, make_panel

ALPHA = 0.05


def measure(n_months: int, n_panels: int, first_seed: int) -> dict[str, float]:
    raw = raw_no_lags = tests = 0
    holm_default = holm_no_lags = by_default = 0
    for seed in range(first_seed, first_seed + n_panels):
        cfg = SyntheticConfig(
            n_assets=150,
            n_months=n_months,
            planted=(PlantedFactor("p", ic=0.04, persistence=0.9),),
            n_noise=25,
            reported_factor=None,
            delist_prob=0.0,
            late_listing_share=0.0,
            seed=seed,
        )
        synthetic = make_panel(cfg)
        series = [information_coefficient(synthetic.panel, name) for name in synthetic.truth.noise]
        p = np.array([newey_west_mean(x).p_value for x in series])
        p0 = np.array([newey_west_mean(x, lags=0).p_value for x in series])
        tests += len(p)
        raw += int((p < ALPHA).sum())
        raw_no_lags += int((p0 < ALPHA).sum())
        holm_default += int((holm(p) < ALPHA).any())
        holm_no_lags += int((holm(p0) < ALPHA).any())
        by_default += int((benjamini_yekutieli(p) < ALPHA).any())
    return {
        "unadjusted rejection rate, default lags": raw / tests,
        "unadjusted rejection rate, zero lags": raw_no_lags / tests,
        "panels with any Holm rejection, default lags": holm_default / n_panels,
        "panels with any Holm rejection, zero lags": holm_no_lags / n_panels,
        "panels with any Benjamini-Yekutieli rejection": by_default / n_panels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="run a tenth of the panels")
    args = parser.parse_args()
    scale = 10 if args.quick else 1
    for n_months, n_panels, first_seed in ((240, 400, 9000), (120, 200, 5000)):
        n_panels //= scale
        print(f"\n{n_panels} panels of 150 assets x {n_months} months, nominal {ALPHA:.0%}")
        for label, value in measure(n_months, n_panels, first_seed).items():
            print(f"  {label:<50} {value:6.1%}")


if __name__ == "__main__":
    main()
