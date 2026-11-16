"""Run the pipeline on CRSP and Compustat.

In sample (July 1963 to December 2020), every method:

    uv run python scripts/run_real.py

Holdout (January 2021 to December 2025), the one frozen configuration, once:

    uv run python scripts/run_real.py --holdout

Needs ``data/wrds`` from ``scripts/wrds_download.py``. The in-sample run never loads a
month after December 2020. The holdout run refuses to start if its result file already
exists, so it cannot be repeated by accident. Results are written to ``results/``; the
trial ledger is ``results/trials.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from factorpipe.bootstrap import sharpe_interval
from factorpipe.combine import METHODS, combine
from factorpipe.evaluation import (
    information_coefficient,
    long_short,
    quantile_returns,
    screen_factors,
)
from factorpipe.french import load_french_factors
from factorpipe.ledger import TrialLedger, fingerprint
from factorpipe.pbo import probability_of_backtest_overfitting
from factorpipe.portfolio import backtest_strategy
from factorpipe.sharpe import deflated_sharpe_from_ledger, sharpe_ratio
from factorpipe.stats import hac_regression, newey_west_mean
from factorpipe.wrds_data import (
    HOLDOUT_STARTS_AFTER,
    SAMPLE_START,
    build_panel,
    load_raw,
    restrict_to_universe,
)

RESULTS = Path("results")
LEDGER = RESULTS / "trials.jsonl"
COST_RATE = 0.001  # 10 basis points of traded value, fixed before any run
FRENCH = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

# Fixed after the in-sample run and before the holdout run. See DECISIONS.md.
FROZEN_METHOD: str | None = "trees"


def load_panel(end: str | None):
    raw = load_raw("data/wrds", end=end)
    return restrict_to_universe(build_panel(raw, start=SAMPLE_START))


def annual(sharpe_per_month: float) -> float:
    return sharpe_per_month * math.sqrt(12)


def summarize(returns: pd.Series) -> dict[str, float]:
    interval = sharpe_interval(returns)
    wealth = (1.0 + returns).cumprod()
    return {
        "months": len(returns),
        "mean_annual": float(returns.mean() * 12),
        "vol_annual": float(returns.std() * math.sqrt(12)),
        "sharpe": float(sharpe_ratio(returns)),
        "sharpe_low": annual(interval.lower),
        "sharpe_high": annual(interval.upper),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
    }


def attribution(returns: pd.Series) -> pd.DataFrame:
    factors = load_french_factors("data/french")[FRENCH].reindex(returns.index)
    return hac_regression(returns, factors)


def in_sample() -> None:
    panel = load_panel(HOLDOUT_STARTS_AFTER)
    ledger = TrialLedger(LEDGER)
    months, securities = panel.returns.shape
    print(f"panel: {months} months to {panel.dates[-1].date()}, {securities:,} securities")
    print(f"average universe size: {panel.universe.sum(axis=1).mean():.0f}\n")

    screen = screen_factors(panel)
    extra = []
    for name in screen.index:
        pearson = newey_west_mean(information_coefficient(panel, name, method="pearson"))
        spread = newey_west_mean(long_short(quantile_returns(panel, name)))
        extra.append((pearson.mean, pearson.t_stat, spread.mean * 100, spread.t_stat))
    screen[["pearson_ic", "pearson_t", "spread_pct", "spread_t"]] = extra
    columns = ["rank_ic", "t_stat", "p_holm", "pearson_ic", "pearson_t", "spread_pct", "spread_t"]
    print("1. Factor screen\n")
    print(screen[columns].round(4).to_string())

    runs, turnover, composite = {}, {}, {}
    for method in METHODS:
        combination = combine(panel, method, ledger=ledger)
        scored = panel.with_characteristic("composite", combination.scores)
        composite[method] = information_coefficient(scored, "composite")
        result = backtest_strategy(panel, combination, ledger=ledger, cost_rate=COST_RATE)
        runs[method] = result.run.active_returns
        traded = result.run.turnover
        turnover[method] = float(traded[traded > 0].mean())
        levels = result.levels.value_counts().to_dict()
        print(f"   ran {method}: {len(runs[method])} months, solver levels {levels}", flush=True)

    # Compare on the months every method traded, so no method gets an easier window.
    common = pd.DataFrame(runs).dropna()
    rows = []
    for method in METHODS:
        stats = summarize(common[method])
        stats["composite_rank_ic"] = float(composite[method].reindex(common.index).mean())
        stats["turnover"] = turnover[method]
        rows.append({"method": method, **stats})
    table = pd.DataFrame(rows).set_index("method")
    print(f"\n2. Methods, net of {COST_RATE:.1%} per trade, {len(common)} common months")
    print(f"   {common.index[0].date()} to {common.index[-1].date()}\n")
    print(table.round(3).to_string())

    best = str(table["sharpe"].idxmax())
    deflated = deflated_sharpe_from_ledger(common[best], ledger, fingerprint(panel))
    pbo = probability_of_backtest_overfitting(common, n_blocks=16)
    regression = attribution(common[best])
    print(f"\n3. Best in sample: {best}\n")
    print(f"   backtests in the ledger for this data   {deflated.n_trials}")
    print(f"   Sharpe expected from the best, no skill {annual(deflated.hurdle):.2f}")
    print(f"   deflated Sharpe ratio                   {deflated.dsr:.3f}")
    print(f"   probability of backtest overfitting     {pbo.pbo:.3f}")
    print("\n4. Regression of its returns on Fama-French five factors and momentum\n")
    print(regression.round(4).to_string())
    print(f"   R-squared {regression.attrs['r_squared']:.3f}")

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "sample": [str(common.index[0].date()), str(common.index[-1].date())],
        "cost_rate": COST_RATE,
        "screen": screen[columns].round(6).to_dict("index"),
        "methods": table.round(6).to_dict("index"),
        "best": best,
        "deflated_sharpe": deflated.dsr,
        "hurdle_annual": annual(deflated.hurdle),
        "n_backtests": deflated.n_trials,
        "pbo": pbo.pbo,
        "attribution": regression.round(6).to_dict("index"),
        "r_squared": regression.attrs["r_squared"],
    }
    (RESULTS / "in_sample.json").write_text(json.dumps(payload, indent=2) + "\n")


def holdout() -> None:
    target = RESULTS / "holdout.json"
    if FROZEN_METHOD is None:
        raise SystemExit("no configuration is frozen; set FROZEN_METHOD first")
    if target.exists():
        raise SystemExit(f"{target} exists: the holdout has already been evaluated")
    panel = load_panel(None)
    ledger = TrialLedger(LEDGER)
    combination = combine(panel, FROZEN_METHOD, ledger=ledger)
    result = backtest_strategy(panel, combination, ledger=ledger, cost_rate=COST_RATE)
    returns = result.run.active_returns
    window = returns[returns.index > pd.Timestamp(HOLDOUT_STARTS_AFTER)]
    scored = panel.with_characteristic("composite", combination.scores)
    ic = information_coefficient(scored, "composite")
    stats = summarize(window)
    stats["composite_rank_ic"] = float(ic[ic.index > pd.Timestamp(HOLDOUT_STARTS_AFTER)].mean())
    regression = attribution(window)
    print(f"holdout: {FROZEN_METHOD}, {window.index[0].date()} to {window.index[-1].date()}\n")
    for key, value in stats.items():
        print(f"   {key:<18} {value:.3f}")
    print("\nRegression on Fama-French five factors and momentum\n")
    print(regression.round(4).to_string())
    payload = {
        "method": FROZEN_METHOD,
        "window": [str(window.index[0].date()), str(window.index[-1].date())],
        "cost_rate": COST_RATE,
        **stats,
        "attribution": regression.round(6).to_dict("index"),
        "r_squared": regression.attrs["r_squared"],
    }
    target.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--holdout", action="store_true", help="evaluate the frozen method once")
    args = parser.parse_args()
    holdout() if args.holdout else in_sample()


if __name__ == "__main__":
    main()
