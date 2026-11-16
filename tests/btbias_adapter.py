"""Plugs the pipeline's backtest engine into backtest-bias-suite."""

import pandas as pd
from btbias import BacktestResult, ExecutionConvention, Scenario, Strategy
from btbias.protocol import View

from factorpipe.backtest import Market, Snapshot, run_backtest


class PipelineEngineAdapter:
    name = "factorpipe"

    def run(self, scenario: Scenario, strategy: Strategy) -> BacktestResult:
        actions = scenario.corporate_actions.assign(
            ex_date=lambda d: pd.to_datetime(d["ex_date"]), value=lambda d: d["value"].astype(float)
        )
        by_kind = {
            kind: rows.pivot(index="ex_date", columns="asset", values="value")
            for kind, rows in actions.groupby("kind")
        }
        market = Market(
            open=scenario.open,
            close=scenario.close,
            last_bar=pd.to_datetime(scenario.listings["delist_date"]).dropna(),
            delisting_return=scenario.listings["delisting_return"].astype(float),
            splits=by_kind.get("split"),
            dividends=by_kind.get("dividend"),
            cost_rate=scenario.cost_model.rate,
            borrow_rate=scenario.borrow.daily_rate,
            no_short=scenario.borrow.hard_to_borrow,
            fill_at="open" if scenario.convention is ExecutionConvention.NEXT_OPEN else "close",
            initial_cash=scenario.initial_cash,
        )

        def decide(t: pd.Timestamp, snapshot: Snapshot) -> pd.Series:
            return strategy.target_weights(self.view(scenario, snapshot))

        run = run_backtest(market, decide)
        return BacktestResult(nav=run.nav, trades=run.trades, weights=run.weights)

    @staticmethod
    def view(scenario: Scenario, snapshot: Snapshot) -> View:
        """Prices come from the engine's snapshot. The engine has no notion of index
        membership or fundamentals, so those two are sliced here."""
        t = snapshot.as_of
        fundamentals = scenario.fundamentals
        if not fundamentals.empty:
            fundamentals = fundamentals[pd.to_datetime(fundamentals["publish_date"]) <= t]
        members = scenario.membership
        if members.empty:
            universe = tuple(snapshot.close.columns[snapshot.close.iloc[-1].notna()])
        else:
            start, end = pd.to_datetime(members["start"]), pd.to_datetime(members["end"])
            current = set(members.loc[(start <= t) & (end.isna() | (t < end)), "asset"])
            universe = tuple(a for a in scenario.assets if a in current)
        held = snapshot.weights
        return View(
            as_of=t,
            open=snapshot.open,
            close=snapshot.close,
            universe=universe,
            fundamentals=fundamentals,
            weights=held[held != 0.0],
        )
