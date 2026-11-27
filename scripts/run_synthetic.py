"""Run the whole pipeline on the default synthetic panel and print every table.

    uv run python scripts/run_synthetic.py

500 assets, 240 months, 2 planted factors, 10 noise factors, and a quarterly reported
version of the strong factor. Takes about 20 seconds.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pandas as pd

from factorpipe.bootstrap import sharpe_interval
from factorpipe.combine import METHODS, combine
from factorpipe.evaluation import information_coefficient, screen_factors
from factorpipe.ledger import TrialLedger, fingerprint
from factorpipe.pbo import probability_of_backtest_overfitting
from factorpipe.portfolio import backtest_strategy
from factorpipe.sharpe import deflated_sharpe_from_ledger, sharpe_ratio
from factorpipe.synthetic import SyntheticConfig, make_panel

SEED = 2026
COST_RATE = 0.001  # 10 basis points per trade
BURN_IN = 60

# Printed at the precision the README quotes, so its numbers can be copied from the output
# rather than rounded a second time by hand.
FORMATS = {
    "true_ic": "{:.2f}",
    "rank_ic": "{:.3f}",
    "t_stat": "{:.1f}",
    "p_holm": "{:.3f}",
    "composite_ic": "{:.3f}",
    "net_sharpe": "{:.2f}",
    "ci_low": "{:.2f}",
    "ci_high": "{:.2f}",
    "monthly_turnover": "{:.2f}",
}


def show(table: pd.DataFrame) -> str:
    formatters = {c: FORMATS[c].format for c in table.columns if c in FORMATS}
    return table.to_string(formatters=formatters)


def main() -> None:
    synthetic = make_panel(SyntheticConfig(seed=SEED))
    panel, truth = synthetic.panel, synthetic.truth

    print("1. Factor screen (rank IC, Newey-West t, Holm-adjusted p)\n")
    screen = screen_factors(panel)
    screen.insert(0, "true_ic", [truth.ic.get(name, float("nan")) for name in screen.index])
    print(show(screen[["true_ic", "rank_ic", "t_stat", "p_holm", "significant_holm"]]))

    print(f"\n2. Combination and portfolio, net of {COST_RATE:.1%} per trade")
    print(f"   best possible composite IC: {truth.best_ic:.3f}\n")
    with tempfile.TemporaryDirectory() as folder:
        ledger = TrialLedger(Path(folder) / "trials.jsonl")
        rows, returns = [], {}
        for method in METHODS:
            combination = combine(panel, method, ledger=ledger)
            scored = panel.with_characteristic("composite", combination.scores)
            ic = information_coefficient(scored, "composite", method="pearson").iloc[BURN_IN:]
            result = backtest_strategy(panel, combination, ledger=ledger, cost_rate=COST_RATE)
            active = result.run.active_returns
            returns[method] = active
            interval = sharpe_interval(active)
            turnover = result.run.turnover
            rows.append(
                {
                    "method": method,
                    "composite_ic": ic.mean(),
                    "net_sharpe": sharpe_ratio(active),
                    "ci_low": interval.lower * math.sqrt(12),
                    "ci_high": interval.upper * math.sqrt(12),
                    "monthly_turnover": turnover[turnover > 0].mean(),
                    "months": len(active),
                }
            )
        table = pd.DataFrame(rows).set_index("method")
        print(show(table))

        best = str(table["net_sharpe"].idxmax())
        deflated = deflated_sharpe_from_ledger(returns[best], ledger, fingerprint(panel))
        pbo = probability_of_backtest_overfitting(pd.DataFrame(returns), n_blocks=10)
        print(f"\n3. Judging the best method ({best})\n")
        print(f"   backtests in the ledger                 {deflated.n_trials}")
        print(f"   all trials in the ledger                {ledger.n_trials}")
        hurdle = deflated.hurdle * math.sqrt(12)
        print(f"   Sharpe expected from the best, no skill {hurdle:.2f}")
        print(f"   deflated Sharpe ratio                   {deflated.dsr:.3f}")
        print(f"   probability of backtest overfitting     {pbo.pbo:.3f}")


if __name__ == "__main__":
    main()
