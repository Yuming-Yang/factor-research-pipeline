# factor-research-pipeline

A factor research pipeline, from signal to portfolio, in which every step carries its own evidence against overfitting, and which is checked on synthetic data with a known answer before it is pointed at real data.

Pick the best of 40 strategies and it will look good. This is what the pipeline says about the winner, on a panel where all 40 factors are noise and on one where two of them are real:

| | 40 noise factors | 2 real + 38 noise |
| --- | --- | --- |
| Annualized Sharpe ratio of the best one | 0.42 | 1.96 |
| 95% bootstrap interval | 0.04 to 0.82 | 1.60 to 2.41 |
| P(true Sharpe > 0), no adjustment | 0.968 | 1.000 |
| Sharpe expected from the best of 40 with no skill | 0.44 | 0.87 |
| Deflated Sharpe ratio | 0.467 | 1.000 |
| Probability of backtest overfitting | 0.615 | 0.000 |

On pure noise the winner passes a test at 5% and its bootstrap interval excludes zero. Neither knows that 40 things were tried. Its Sharpe ratio is almost exactly what the best of 40 skill-free strategies is expected to show.

On U.S. equities, the one configuration chosen in sample had a net Sharpe ratio of 1.50 through 2020 and −0.37 on the five years held back. Both numbers are reported [below](#real-data).

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Yuming-Yang/factor-research-pipeline
cd factor-research-pipeline
uv sync
uv run python scripts/overfitting_demo.py
uv run python scripts/run_synthetic.py
```

The first script prints the table above (about 15 seconds). The second runs the whole pipeline on one synthetic panel and prints the tables in the Results section (about 20 seconds). Neither needs any data or credentials.

## What it does

```
panel ──> factor evaluation ──> combination ──> portfolio ──> backtest ──> statistics
            IC, decay,           six methods,    Grinold       engine       deflated Sharpe,
            quantiles,           walk-forward,   alpha,        tested by    PBO, bootstrap,
            Newey–West,          purge and       robust-mvo    the bias     factor regression
            Holm / BY            embargo                       suite
                                      │              │
                                      └── trial ledger ──────────────────────────┘
```

- **Panel.** Monthly returns and point-in-time characteristics. A characteristic at month `t` can only be paired with returns from `t+1` on, and that pairing is built in one place. Fundamentals are dated by publication, restatements by the date they were restated.
- **Synthetic panel.** Some factors are planted with a chosen information coefficient and persistence, so the IC at any horizon is known in closed form. The rest are noise with an IC of exactly zero. The research code is tested against that truth.
- **Evaluation.** IC and rank IC, decay, quantile returns, turnover, Newey–West t-statistics, and p-values adjusted for the number of factors tested.
- **Combination.** Equal weight, ICIR, factor momentum, factor mean-variance, ridge and gradient-boosted trees. All walk-forward: each reads history through an explicit rule for when an IC, a factor return or a regression row becomes known.
- **Portfolio.** Scores become expected returns by the Grinold rule. Positions come from [`robust-mvo`](https://github.com/Yuming-Yang/robust-mvo) with a structural risk model (market or industries, beta, styles), turnover and cost terms.
- **Backtest.** A bar-by-bar engine that passes all eight traps of [`backtest-bias-suite`](https://github.com/Yuming-Yang/backtest-bias-suite) in CI.
- **Trial ledger.** Every combination and every backtest is written to an append-only file before it runs. The deflated Sharpe ratio reads its trial count from that file; it cannot be passed in.
- **Statistics.** Deflated Sharpe ratio, probability of backtest overfitting, stationary bootstrap intervals, and a regression on factor returns with Newey–West errors.

## Results

All numbers in this section are from synthetic data. Each table names the command that prints it. Real-data results are in the next section.

### One run of the whole pipeline

`uv run python scripts/run_synthetic.py`: 500 assets, 240 months, two planted factors, ten noise factors, and a quarterly reported version of the strong factor that becomes public with a lag.

Factor screen:

| Factor | True IC | Rank IC | t | Holm p | Flagged |
| --- | --- | --- | --- | --- | --- |
| `planted_strong` | 0.05 | 0.046 | 13.5 | 0.000 | yes |
| `planted_weak` | 0.02 | 0.021 | 4.8 | 0.000 | yes |
| `planted_strong_reported` | below 0.05 | 0.038 | 10.3 | 0.000 | yes |
| `noise_07` | 0 | −0.015 | −3.8 | 0.002 | **yes** |
| nine other noise factors | 0 | −0.007 to 0.004 | −1.9 to 1.1 | above 0.46 | no |

One noise factor is flagged after the Holm adjustment. That is a false positive, and it is left in because it is what the adjustment does in samples this size (see the next table).

Combination and portfolio, net of 10 basis points per trade. The best possible composite IC on this panel is 0.054.

| Method | Composite IC | Net Sharpe | 95% interval | Monthly turnover |
| --- | --- | --- | --- | --- |
| Equal weight | 0.024 | 0.50 | −0.02 to 1.04 | 0.34 |
| Factor momentum | 0.037 | 1.43 | 1.02 to 1.85 | 0.60 |
| Trees | 0.041 | 1.06 | 0.55 to 1.62 | 1.14 |
| ICIR | 0.042 | 1.83 | 1.37 to 2.34 | 0.37 |
| Factor MVO | 0.044 | 1.84 | 1.37 to 2.35 | 0.39 |
| Ridge | 0.046 | 1.93 | 1.52 to 2.38 | 0.57 |

For the best method (ridge): 6 backtests in the ledger, a Sharpe ratio of 0.73 expected from the best of 6 with no skill, a deflated Sharpe ratio of 1.000, and a probability of backtest overfitting of 0.139.

The ranking of methods says the methods work as built. It does not say which to prefer on real data: here the true model is linear, the factors are independent and the signal never changes.

### False positives after adjustment

`uv run python scripts/false_positive_rates.py`: panels of 150 assets with 25 noise factors each, nominal rate 5%.

| | 400 panels × 240 months | 200 panels × 120 months |
| --- | --- | --- |
| Unadjusted rejection rate per noise factor | 5.6% | 6.0% |
| Panels with any Holm rejection | 8.8% | 7.0% |
| Panels with any Benjamini–Yekutieli rejection | 1.8% | 1.5% |

The Holm rate is above nominal. Newey–West standard errors are a little too small in short samples, and Holm depends on very small p-values. The pipeline reports this instead of tuning it away: a Holm-adjusted 5% from this code should be read as closer to 9%.

### Risk forecasts

`uv run python scripts/risk_bias.py`: the standard deviation across months of realized return over predicted volatility, on panels with dispersion in beta and eight industries, three seeds. A calibrated model gives 1; above 1 means risk was understated.

| Portfolio | Structural model | Style factors only |
| --- | --- | --- |
| Random long-short | 0.84 to 0.97 | 0.76 to 0.92 |
| Long high beta, short low beta | 0.74 to 0.86 | 1.36 to 1.83 |
| Long one industry, short the rest | 0.77 to 0.86 | 1.42 to 1.57 |

A risk model built from the alpha factors alone understates the risk of beta and industry tilts by 40% to 80%. The structural model overstates it by 15% to 35%.

### Checks against independent code

- IC and quantile returns match `alphalens-reloaded` to 1e-12.
- Newey–West standard errors and the regression match `statsmodels` to 1e-8 or better.
- The quantile long-short portfolio traded through the engine matches the spread computed directly from forward returns to 1e-12, on a panel with late listings and delistings.
- Replacing all data after a date leaves every weight, score, position and NAV up to that date unchanged, for every method.

## Real data

Calculated based on data from CRSP US Stock Database, © Center for Research in Security Prices (CRSP), The University of Chicago Booth School of Business, and from Compustat, © S&P Global Market Intelligence. No security-level data is in this repository.

U.S. common stocks at or above the NYSE 20th size percentile (1,822 on average), eleven published factors, monthly, ten basis points of traded value per trade. The last five years were set aside before any result was read.

```bash
WRDS_USERNAME=<user> uv run --no-project --with wrds scripts/wrds_download.py
uv run python scripts/run_real.py            # in sample, about eight minutes
uv run python scripts/run_real.py --holdout  # refuses to run a second time
```

The download runs in its own environment because the `wrds` package requires an older pandas than this project.

### The result

| | In sample | Holdout |
| --- | --- | --- |
| Period | August 1969 to December 2020 | January 2021 to December 2025 |
| Net Sharpe ratio | 1.50 | −0.37 |
| 95% bootstrap interval | 1.12 to 1.90 | −1.20 to 0.40 |
| Annual return | 20.0% | −5.4% |
| Annual volatility | 13.3% | 14.8% |
| Maximum drawdown | −41% | −44% |
| Composite rank IC | 0.039 | 0.009 |
| Alpha against Fama–French five factors and momentum, per month | 1.45% (t 7.1) | −0.34% (t −0.7) |

The configuration was chosen by a rule fixed in advance (the method with the best in-sample net Sharpe ratio, which was gradient-boosted trees), committed, and then run on the holdout once. **It lost money out of sample.** The two intervals do not overlap, so this is more than five noisy years.

In sample, the deflated Sharpe ratio was 1.000 and the probability of backtest overfitting 0.071. Both were right about what they measure: the result was not an accident of picking one of six methods. Neither measures whether the backtest's assumptions hold. The in-sample numbers were flagged as too good before the holdout was run, for reasons written down at the time: turnover above 100% a month at a flat ten basis points, one-month reversal as a main input, no market impact, no borrow fee. Which of these, or a change in the market itself, explains the holdout was not investigated, because any further analysis of those five years would be a second look.

### In sample, all six methods

617 months in which every method traded.

| Method | Net Sharpe | 95% interval | Annual return | Max drawdown | Monthly turnover |
| --- | --- | --- | --- | --- | --- |
| Factor momentum | 0.70 | 0.43 to 1.01 | 10.4% | −37% | 1.02 |
| ICIR | 0.91 | 0.58 to 1.26 | 11.8% | −36% | 1.30 |
| Equal weight | 1.15 | 0.75 to 1.54 | 16.1% | −42% | 1.32 |
| Factor MVO | 1.27 | 0.95 to 1.57 | 15.7% | −27% | 1.19 |
| Ridge | 1.27 | 0.90 to 1.64 | 18.7% | −38% | 1.67 |
| Trees | 1.50 | 1.12 to 1.90 | 20.0% | −41% | 1.65 |

Only trees were run on the holdout. How the other five did there is not known.

### Rank IC and returns can disagree

| Factor | Rank IC (t) | Top-minus-bottom quintile, % per month (t) |
| --- | --- | --- |
| Momentum | 0.032 (5.4) | 0.88 (4.4) |
| Reversal | 0.030 (7.4) | 0.59 (4.3) |
| Share issuance | 0.018 (5.2) | 0.48 (4.3) |
| Gross profitability | 0.015 (3.2) | 0.40 (3.6) |
| Asset growth | 0.009 (2.3) | 0.40 (3.7) |
| Earnings yield | 0.024 (4.0) | 0.35 (1.8) |
| Book-to-market | 0.013 (2.1) | 0.30 (1.5) |
| Accruals | 0.007 (2.3) | 0.26 (3.3) |
| Size (small) | −0.010 (−2.1) | 0.22 (1.8) |
| Volatility (low) | 0.028 (3.4) | 0.05 (0.2) |
| Beta (low) | 0.018 (2.3) | −0.04 (−0.2) |

Low volatility and low beta have a clearly positive rank IC and no return spread; size has a negative rank IC and a positive spread. A rank correlation compares typical outcomes, and the typical return of a volatile stock is well below its average, because a few large gains carry the average. Rank IC therefore rewards any characteristic that sorts stocks by low volatility. The screen reports all three measures; the spread is the one that corresponds to money.

## Design trade-offs

[DECISIONS.md](DECISIONS.md) records 74 choices with the alternative that was rejected. The ones that shape the project:

- **A synthetic ground truth first.** Research code is hard to test on real data because nobody knows the right answer. The cost is that every result here describes a world that is simpler than a real market.
- **Trial counts are read, not declared.** The deflated Sharpe ratio takes its number of trials from the ledger. The cost is that a deflated ratio needs a ledger; the benefit is that failed attempts cannot be left out.
- **One engine for the trap tests and for the results.** The engine trades bars of raw prices, so the bias suite can test it directly, and monthly returns are turned into a price index for it. A separate returns-based engine would have been shorter and untested.
- **Measured shortfalls are reported.** The Holm false-positive rate and the risk model's overstatement are in the tables above. Both could have been tuned closer to their targets; each fix would have added a choice made after seeing the result.
- **Defaults are not tuned.** Risk aversion, windows and caps were set once. Choosing them from backtest results would make each choice a trial.
- **No splitter object.** The rule for when a piece of history becomes known is applied where that history is read, which is where a mistake would be made.

## Limitations

- **The real-data strategy did not survive its holdout.** Nothing here is evidence of a tradable strategy.
- **One look at five years.** The holdout interval is wide, and why the result changed sign was not investigated, to keep it a single look.
- **Backtest costs are optimistic.** A flat ten basis points for 1963 to 2025, with turnover above 100% a month.
- **The synthetic world is kind.** A linear truth, independent factors, a constant signal, one volatility for every asset, and delistings unrelated to the signal. Ridge wins because the world is linear; trees have nothing extra to find.
- **Trials are counted as independent.** Variations of one idea are correlated, so the count overstates the effective number and the deflated ratio is conservative.
- **A combination that is scored but never backtested does not count** toward the deflated Sharpe ratio. Comparing many combinations by IC and backtesting only the best understates the search.
- **The risk model is simple.** No size factor, unweighted regressions, one industry label per asset for the whole sample. It overstates the risk of factor tilts and does not make a book beta or industry neutral; that would take an optimizer constraint.
- **Costs are linear.** No market impact, one borrow rate for every asset.
- **Monthly bars.** No daily data, so no intramonth timing and no volatility measured from daily returns.
- **Three factor definitions are simplified** against their papers (volatility, share issuance, beta). DECISIONS.md lists how.
- **Bootstrap block length and Newey–West lag counts are rules of thumb**, not chosen from the data.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

218 tests. The suite runs without network access and without paid services.

## License

MIT. See [LICENSE](LICENSE). The licence covers the code; CRSP and Compustat data remain subject to their own terms.

## References

- Bailey, D. and López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*.
- Bailey, D., Borwein, J., López de Prado, M. and Zhu, Q. (2017). The probability of backtest overfitting. *Journal of Computational Finance*.
- Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*.
- Grinold, R. (1994). Alpha is volatility times IC times score. *Journal of Portfolio Management*.
- Harvey, C., Liu, Y. and Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*.
- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*.
- Lo, A. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Newey, W. and West, K. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*.
- Politis, D. and Romano, J. (1994). The stationary bootstrap. *Journal of the American Statistical Association*.

The papers behind each of the eleven factors are listed in [SPEC.md](SPEC.md).
