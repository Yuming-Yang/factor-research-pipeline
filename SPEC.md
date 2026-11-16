# SPEC: Factor Research Pipeline

Status: v1 scope and decisions D1–D7 fixed on 2026-11-15 (see Decisions at the end). In progress.

## Problem

A factor backtest is easy to make look good. The factor is chosen on the full sample, the horizon and decay window are tuned on the same data that is then reported, only the best of many variants is shown, and the Sharpe ratio comes with no standard error. None of this raises an error.

This project is a research pipeline from factor to portfolio in which every step carries its own evidence against overfitting. It is checked first on synthetic data where the true signal is known, then run on U.S. equities.

It sits between two finished projects: [`backtest-bias-suite`](https://github.com/Yuming-Yang/backtest-bias-suite) checks that the backtest engine is unbiased, and [`robust-mvo`](https://github.com/Yuming-Yang/robust-mvo) builds the portfolio. This project supplies the research in the middle and uses both.

## Approach

### 1. Data layer

One panel interface, two sources.

- **Synthetic panel (always available).** A seeded generator of monthly returns and characteristics. Some characteristics are *planted factors* with a configurable true information coefficient and decay; the rest are noise with a true IC of zero. It also produces delistings with a delisting return and fundamentals with a publish lag. Every correctness test and the CI run use this source only.
- **WRDS loader (optional, needs a subscription).** CRSP monthly stock file and Compustat fundamentals through the `wrds` Python package, linked with the CCM table. Fundamentals are lagged to their public availability date. Delisting returns are included. No WRDS data is stored in the repository; the loader writes to a local cache that is git-ignored.

CRSP now ships in the CIZ format only. The legacy SIZ format ended with the December 2024 data, and delisting returns moved from a separate table into the security table. The loader targets the CIZ tables and does not reuse legacy field names.

### 2. Factor evaluation

For each factor, on each rebalance date:

- IC and rank IC (Spearman: both the factor and the forward return are ranked) at horizons of 1, 3, 6 and 12 months, and the decay curve across horizons.
- Quantile portfolio returns and the long-short spread.
- Factor turnover.
- Newey–West t-statistics on the IC series and on the spread return, with the lag set from the horizon overlap.
- A multiple-testing adjustment across all factors evaluated (Holm and Benjamini–Hochberg–Yekutieli), reported next to the unadjusted values. The t > 3 hurdle from Harvey, Liu and Zhu (2016) is shown as a reference line.

`alphalens-reloaded` is used in tests as an independent check on IC and quantile returns. It is a development dependency only.

### 3. Factor combination

Six methods, all on the same walk-forward splits:

| Method | What it does |
| --- | --- |
| Equal weight | Average of standardized factors. The baseline |
| ICIR weight | Weights proportional to trailing IC mean over IC standard deviation |
| Factor momentum | Weights from trailing factor returns |
| Factor MVO | Mean-variance weights on factor returns, with a shrunk covariance from `robust-mvo` |
| Ridge | Cross-sectional ridge regression of forward returns on factors |
| Gradient-boosted trees | A non-linear comparison, with scikit-learn's histogram gradient boosting |

Rules that apply to every method:

- **Walk-forward only.** Parameters and hyperparameters are fitted on data strictly before the evaluation date.
- **Purge and embargo.** Training rows whose forward-return window overlaps the evaluation period are dropped (purge), and a gap is left after each evaluation block (embargo), following López de Prado (2018).
- **Trial ledger.** Every configuration that is evaluated is appended to a log file. The number of trials is an input to the deflated Sharpe ratio, so it cannot be left out.

### 4. Portfolio construction

Scores are converted to expected returns with the Grinold rule (alpha = IC × volatility × score), so alpha has units of return. The optimization is a call to `robust-mvo`: factor risk model, turnover and cost terms, and its degradation chain. This project adds no optimizer of its own.

### 5. Backtest

A small monthly engine with an adapter for `backtest-bias-suite`. The engine must pass all eight traps in CI before any result is reported.

### 6. Statistics

- Deflated Sharpe ratio (Bailey and López de Prado, 2014), using the trial count from the ledger.
- Probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey, Borwein, López de Prado and Zhu, 2017).
- Stationary bootstrap confidence intervals for Sharpe and for the difference between two methods (Politis and Romano, 1994).
- Regression of strategy returns on the Fama–French five factors plus momentum, from the Ken French data library.

### 7. Holdout

The last years of the real-data sample are set aside when the project starts and are not read until every design choice is frozen. They are evaluated once, and that single result is reported whatever it is.

## Factor list (v1, proposed)

A small set from the published literature. Each definition will cite its paper.

| Factor | Reference |
| --- | --- |
| Size | Banz (1981) |
| Book-to-market | Fama and French (1992) |
| Earnings yield | Basu (1977) |
| Momentum (12–2) | Jegadeesh and Titman (1993) |
| Short-term reversal | Jegadeesh (1990) |
| Gross profitability | Novy-Marx (2013) |
| Asset growth | Cooper, Gulen and Schill (2008) |
| Accruals | Sloan (1996) |
| Net share issuance | Pontiff and Woodgate (2008) |
| Idiosyncratic volatility | Ang, Hodrick, Xing and Zhang (2006) |
| Market beta | Frazzini and Pedersen (2014) |

## Non-goals

- No new factors and no claim that any strategy is profitable.
- No LLM or agent layer. Factor proposal by agents belongs to a separate project.
- No daily or intraday data in v1.
- No distribution of CRSP or Compustat data, raw or reconstructable.
- No code, factor names, fee tables or data from any earlier private work. Definitions come from the cited papers.

## Acceptance criteria

- [x] On the synthetic panel, every planted factor is detected, and its estimated IC lies inside the stated confidence interval around the true value.
- [x] On the synthetic panel, the share of noise factors flagged as significant is measured over repeated seeds, before and after adjustment, and reported next to the nominal rate. (Reworded on 2026-11-15: the first version required the adjusted rate to be at or below nominal. Holm measures above it; see DECISIONS.md.)
- [x] IC and quantile returns agree with `alphalens-reloaded` on a shared test panel, within a stated tolerance.
- [x] A test shows that replacing every input after a date changes no fitted weight and no score on or before that date, for every combination method.
- [x] The backtest engine passes all eight `backtest-bias-suite` traps in CI.
- [ ] The deflated Sharpe ratio uses the trial count read from the ledger, and a test fails if a configuration is evaluated without being logged.
- [ ] The full pipeline runs on the synthetic panel from a clean clone with one command, without WRDS access.
- [ ] Every real-data number in the README is produced by one documented command for a reader with WRDS access, and is labelled as such.
- [ ] The holdout period is evaluated once, and the README reports it.
- [ ] `uv run pytest` and ruff pass in CI.
- [ ] README: one-sentence problem, quick-start demo, results with numbers, design trade-offs, Limitations.

## Milestones

1. **Scaffold, data layer, synthetic panel.** Panel interface, generator with planted and noise factors, point-in-time lagging. Done.
2. **Factor evaluation.** IC, decay, quantiles, turnover, Newey–West, multiple-testing adjustment, cross-check against `alphalens-reloaded`. Done.
3. **Factor combination.** Walk-forward splitter with purge and embargo, six methods, trial ledger. Done.
4. **Portfolio and backtest.** Alpha scaling, `robust-mvo` call, engine, `backtest-bias-suite` adapter. Done.
5. **Statistics.** Deflated Sharpe, PBO, bootstrap, factor attribution.
6. **WRDS loader, real-data run, holdout, README.** The loader is written here, where a live account can check its table and column names.

Each milestone is one branch and one PR.

## Dependencies (checked 2026-11-15)

Versions are the latest on PyPI on the check date. None was installed or run for this draft.

| Package | Version | Role | Licence |
| --- | --- | --- | --- |
| `robust-mvo` | git | Portfolio construction | MIT |
| `backtest-bias-suite` | git | Engine tests (dev) | MIT |
| `wrds` | 3.5.0 | WRDS access (optional extra) | BSD-3-Clause |
| `alphalens-reloaded` | 0.5.1 | Test oracle (dev) | Apache-2.0 |
| `statsmodels` | 0.15.0 | Newey–West, regressions | BSD-3-Clause |
| `scikit-learn` | ≥ 1.9 | Ridge, gradient boosting | BSD-3-Clause |
| `skfolio` | 1.12.16 | Reference for combinatorial purged CV; not a dependency | BSD-3-Clause |
| `openassetpricing` | 0.1.0 | Reference for factor definitions; not a dependency | GPLv2 |

**Data terms.**

- CRSP requires an attribution line on any research result made from its data. Raw CRSP data may not be made available to non-subscribers.
- A WRDS guide on data sharing advises sharing only transformed or aggregated results that cannot be reverse-engineered. This is guidance; the controlling document is the university's subscription agreement.
- The CRSP site states that Morningstar completed its acquisition of CRSP on 2026-02-02, so terms may change.
- Not yet confirmed: whether the author's account covers CRSP monthly, Compustat and the CCM link table; and whether aggregate statistics and charts may be published in a public repository. Both must be confirmed before milestone 6.
- The Ken French data library is public. Its access path from Python was not tested.

## Cost

No paid APIs and no cloud compute. WRDS access is through the university.

## Related work

Descriptions are what the projects say about themselves; none was run.

- `alphalens-reloaded`: factor tear sheets. It evaluates one factor at a time and does not address selection across many factors.
- `skfolio`: portfolio optimization with combinatorial purged cross-validation.
- Open Source Asset Pricing (Chen and Zimmermann): code and data for published predictors. Its latest listed data update is October 2025.
- Qlib and RD-Agent: factor mining frameworks, including LLM-driven ones.

The difference here is the combination: a synthetic ground truth for the research code itself, a trial ledger tied to the deflated Sharpe ratio, an engine tested for bias, and a single-look holdout.

## Decisions (2026-11-15)

| ID | Decision | Choice |
| --- | --- | --- |
| D1 | Frequency | Monthly for v1. It matches the literature the factors come from and keeps the data small. Daily is a later extension |
| D2 | Names | Repository `factor-research-pipeline`, import name `factorpipe` |
| D3 | Factor list | The eleven above |
| D4 | Universe | Common stocks on NYSE, AMEX and NASDAQ, excluding names below the NYSE 20th size percentile |
| D5 | Sample and holdout | 1963-07 onward; the final five years as holdout |
| D6 | Publishing WRDS-derived numbers | To be confirmed with the university library before milestone 6. If not allowed, the README reports synthetic results only and gives the commands for subscribers |
| D7 | How `robust-mvo` and `backtest-bias-suite` are consumed | As git dependencies pinned to a commit |
