"""Pick the best of 40 single-factor strategies, with and without a real signal.

    uv run python scripts/overfitting_demo.py

Two panels of the same size. In the first, all 40 factors are noise. In the second, two
of the 40 predict returns. Each factor is traded as a long-short quantile portfolio, the
best one is picked, and its Sharpe ratio is judged three ways.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from factorpipe.experiments import annualize, select_best_factor
from factorpipe.ledger import TrialLedger
from factorpipe.synthetic import PlantedFactor, SyntheticConfig, make_panel

SHARED = {"n_assets": 300, "n_months": 240, "reported_factor": None, "seed": 2026}
PANELS = {
    "40 noise factors": SyntheticConfig(planted=(), n_noise=40, **SHARED),
    "2 real + 38 noise": SyntheticConfig(
        planted=(
            PlantedFactor("planted_strong", ic=0.05, persistence=0.95),
            PlantedFactor("planted_weak", ic=0.02, persistence=0.80),
        ),
        n_noise=38,
        **SHARED,
    ),
}


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        for label, config in PANELS.items():
            ledger = TrialLedger(Path(folder) / f"{label}.jsonl")
            result = select_best_factor(make_panel(config).panel, ledger)
            d, ci = result.deflated, result.interval
            print(f"\n{label}")
            print(f"  best factor                          {result.best}")
            print(f"  its annualized Sharpe ratio          {result.best_annual_sharpe:6.2f}")
            print(
                f"  95% bootstrap interval               "
                f"[{annualize(ci.lower):.2f}, {annualize(ci.upper):.2f}]"
            )
            print(f"  P(true Sharpe > 0), no adjustment    {d.psr:6.3f}")
            print(f"  backtests in the ledger              {d.n_trials:6d}")
            print(f"  Sharpe expected from the best of them{annualize(d.hurdle):6.2f}")
            print(f"  deflated Sharpe ratio                {d.dsr:6.3f}")
            print(f"  probability of backtest overfitting  {result.pbo:6.3f}")


if __name__ == "__main__":
    main()
