# Decisions

Design choices that are not obvious from the code, with the alternative that was rejected.

## Milestone 1: data layer and synthetic panel

**1. One timing convention for the whole pipeline.** Row `t` of a characteristic is what was public at the end of month `t`; row `t` of returns is the return earned during month `t`. A characteristic at `t` may only be paired with returns from `t+1` on. `Panel.forward_returns` is the only place that builds that pairing, so later modules cannot get it wrong one at a time.

**2. An asset in its last listed month is not investable.** If an asset has a return in `t` and none in `t+1`, it delisted during `t`, and there is nothing to buy at the end of `t`. The alternative, treating it as investable with a forward return of zero, would add positions that could not have been opened.

**3. A delisting inside a forward window leaves cash.** For horizons above one month, a position whose asset delists keeps its proceeds (the delisting return is already in that month's return) at zero return for the rest of the window. Rejected: dropping the observation, which is survivorship bias in the forward return; and returning a partial window, which mixes horizons.

**4. Each asset has one unbroken listed spell.** A missing return inside a spell cannot be told apart from a delisting, and decisions 2 and 3 depend on that distinction. Such panels are rejected at construction. Real data with gaps has to be cleaned before it becomes a `Panel`.

**5. Restatements.** At date `t` the visible value is the most recent period among reports published by `t`, in its most recently published version. A late restatement of an older period does not replace a newer period that is already public. Rejected: using the latest vintage of every period, which leaks the restated value backward.

**6. The synthetic panel has the same volatility for every asset.** The planted IC is then exactly the population cross-sectional Pearson correlation, and the IC at horizon `h` is `ic * rho^(h-1)`. With dispersion in volatility the Pearson IC against raw returns is attenuated by a factor that depends on the volatility distribution, and the test target stops being a single clean number. Dispersion can be added later as an option, with its own stated truth.

**7. Delistings attenuate the planted IC slightly.** Delisting returns are independent of the signal and add return variance. Averaged over 20 seeds, the default configuration gives 0.0495 for a planted IC of 0.05, against 0.0503 with delistings off. Tests that need the exact identity switch delistings off; one test checks the default configuration with a wider tolerance.

**8. The reported factor shows the cost of ignoring publication dates.** One planted factor is also released as quarterly reports with a 1 to 3 month lag. Dating those reports by period end instead of publication date raises the measured IC, and a test asserts the direction and a minimum size of that gap.

**9. The WRDS loader moved from milestone 1 to milestone 6.** Its table and column names can only be checked against a live account, so a loader written earlier would have no test that means anything. Milestones 2 to 5 run on the synthetic panel and do not need it.

## Milestone 2: factor evaluation

**10. Two ICs, two uses.** `information_coefficient` uses the return compounded over the horizon, including delistings; it is the number that matches what a position would earn, and its monthly values overlap. `decay_curve` uses the single-month return at each lag, which isolates how fast the signal fades but conditions on the asset still being listed. The first is reported; the second is a diagnostic.

**11. Rank IC ranks both sides.** Ranking only the factor and correlating with raw returns is not a Spearman correlation and stays sensitive to return outliers. The implementation matches `alphalens-reloaded` to 1e-12 on a shared panel.

**12. Newey-West lags.** The default is the larger of `horizon - 1` and `floor(4 * (n/100)^(2/9))`. The first term is forced by overlap: a 12-month return shares 11 months with its neighbour. The second allows for autocorrelation in the IC series that does not come from overlap. Rejected: `horizon - 1` alone, which gives a plain t-statistic at the one-month horizon.

**13. Newey-West and the adjustments are written from their formulas.** statsmodels checks them in tests (agreement to 1e-10) and is not a runtime dependency. The p-value uses the normal distribution.

**14. The test statistic for screening is the rank IC.** The long-short spread is reported next to it but is not what the adjusted p-values are based on. One statistic per factor keeps the number of tests equal to the number of factors.

**15. The list passed to `screen_factors` must be every factor that was looked at.** A test shows why: among 60 pure-noise factors the best one has an unadjusted p-value of 0.007 and a Holm-adjusted one of 0.40; screened alone, the same factor is "significant".

**16. Quantile ties are broken by column order.** Group sizes stay equal, which keeps the comparison with `alphalens-reloaded` exact on continuous data. On real data with many tied values (for example a zero-inflated characteristic) this is arbitrary and needs revisiting.

### False positives are above the nominal rate, and that is what is reported

The first version of the acceptance criterion required the share of noise factors flagged after adjustment to be at or below the nominal rate. Measured at a nominal 5%, with 150 assets and 25 noise factors per panel:

| Quantity | 400 panels × 240 months | 200 panels × 120 months |
| --- | --- | --- |
| Unadjusted rejection rate per noise factor, default lags | 5.6% | 6.0% |
| Same, zero lags | 5.2% | 5.3% |
| Share of panels with any Holm rejection, default lags | 8.8% | 7.0% |
| Same, zero lags | 6.0% | 5.0% |
| Share of panels with any Benjamini-Yekutieli rejection | 1.8% | 1.5% |

Reproduce with `uv run python scripts/false_positive_rates.py` (about 40 seconds).

The IC series of a noise factor has no autocorrelation by construction, so the lags only add estimation noise, and a Newey-West standard error is known to be too small on average in short samples. Holm depends on very small p-values, where that matters most.

**17. The normal reference is kept and the measured rates are reported.** The criterion was reworded to require the measurement, not a pass. Rejected for now: a Student-t or fixed-b reference distribution, and a data-driven lag count. Either would move the Holm rate toward 5%, at the cost of one more design choice to defend; the size of the gap is small next to what the adjustment removes (25 unadjusted tests at 5.6% each flag at least one noise factor in roughly three panels out of four). Anyone reading a Holm-adjusted result from this pipeline should treat its 5% as closer to 9%.

## Milestone 3: factor combination

**18. "Known at t" is defined once per kind of history.** An IC with horizon `h` formed at `s` is usable at `t` only if `s + h + embargo <= t`; a one-month factor return if `s + 1 + embargo <= t`; a regression row with an `h`-month label if `s + h + embargo <= t`. Each method reads its history through one of these three rules. There is no separate splitter object: with monthly refits, a splitter would hand out one test date at a time, and the rule is easier to check where the shift is applied.

**19. The embargo sits before the evaluation date.** In cross-validation the embargo protects training data that comes after a test fold. In a walk-forward design nothing is trained on data after the evaluation date, so that gap has nothing to protect. Here the embargo is an extra delay, on top of the purge, before history may be used. The default is zero.

**20. The lookahead test replaces later data instead of shifting it.** For each method and two horizon and embargo settings, every return and characteristic after a cutoff is replaced with noise; weights and scores up to the cutoff must be identical, and scores after it must differ. A second test applies the same check to a deliberately leaky weighting and confirms that it is caught. The pattern of missing values is kept, because whether an asset still exists at the end of month `t` is known at `t`.

**21. `investable` is not lookahead.** It reads the next row of the returns grid, but what it encodes is whether the asset delisted during month `t`, which is known at the end of `t`. The last row of a panel is never investable, so a live run needs one more row than the last date it trades on.

**22. Factor returns are regression slopes, not quantile spreads.** The one-month return of a factor is the cross-sectional mean of `z * (r - mean r)`. It uses every asset, is linear in the factor, and needs no choice of quantile count. Quantile spreads stay in the evaluation module, where they are what a reader expects to see.

**23. Ridge is fitted from per-date sufficient statistics.** Each formation date stores `X'X`, `X'y` and `y'y`; a fit over any set of dates is a sum. The penalty is `alpha` times the number of training rows and is chosen on the latest quarter of the training dates, which are purged from the dates used to fit. With factors that are close to uncorrelated, as in the synthetic panel, ridge shrinks all coefficients by about the same factor and the ranking of assets barely changes; the penalty matters when factors are correlated.

**24. The tree model is small and its hyperparameters are fixed.** Sixty trees of depth three, refitted once a year on a ten-year window. No search: each searched setting would have to be either nested inside the training window or counted as a trial, and with an IC of a few percent a larger model mostly fits noise.

**25. Linear weights are normalized to unit absolute sum and may be negative.** A factor whose sign is wrong for the sample gets a negative weight, which a test checks by flipping a planted factor. Equal weighting is the exception: it assumes the sign of every factor is known in advance.

**26. A trial is a method, its parameters, the factor list, the horizon, the embargo and a fingerprint of the data.** Running the same trial twice adds nothing to the count; changing any of these adds one. Values tried inside a training window, such as the ridge penalty, are not trials, because the choice never sees the evaluation data.

**27. `robust-mvo` and scikit-learn are runtime dependencies.** `robust-mvo` is pinned to a commit and supplies the Ledoit-Wolf covariance for the factor-level mean-variance weights.

### What the methods recover on the synthetic panel

`uv run python scripts/combination_recovery.py`. Mean composite IC over five seeds (500 assets, 240 months, 2 planted and 10 noise factors, first 60 months excluded). The best possible score has an IC of 0.054, and the plain average of all twelve factors 0.020.

| Method | Composite IC |
| --- | --- |
| Equal weight | 0.020 |
| Factor momentum | 0.042 |
| Trees | 0.047 |
| ICIR weight | 0.048 |
| Factor MVO | 0.051 |
| Ridge | 0.053 |

The order is specific to this panel: the true model is linear, the factors are independent and the signal is stable. It says the methods work as built. It does not say which one to prefer on real data.

## Milestone 4: portfolio construction and backtest

**28. One engine for the trap tests and for the results.** `backtest-bias-suite` tests daily bars with opens, closes, splits, dividends and borrow fees; the pipeline trades monthly total returns. The engine is written for bars of raw prices, so the suite can test it as it is, and `market_from_panel` turns a panel into prices for the same engine: a total-return index per asset, with each month's open equal to the previous month's close. An order decided at the end of month `t` is therefore filled at that month-end price and earns the return of `t+1`. Rejected: a separate returns-based engine for the pipeline, which would leave the engine that produces the results untested by the suite.

**29. The engine was written without reading from the suite's reference engine at run time, and in a different style.** It follows the same published semantics (sizing at the fill, costs out of cash, fractional shares, cancelled orders, delisting cash at the last close, corporate actions at the open) because those define a pass. It shares no code with the reference engine, and `btbias` is a development dependency only.

**30. What the eight traps do and do not test here.** The engine has no notion of fundamentals or index membership. For T3 and T7 the probe checks therefore exercise the slicing in the test adapter, not the engine; their outcome checks, and all of T1, T2, T4, T5, T6 and T8, test the engine. The pipeline's own point-in-time logic is `pit.py`, tested separately in milestone 1.

**31. A second check that does not depend on the suite.** The top-minus-bottom quantile portfolio traded through the engine gives the same monthly return as the spread computed directly from forward returns in the evaluation module, to 1e-12, on a panel with late listings and delistings. The two share no code beyond the panel.

**32. An asset cannot be bought in its first listed month.** It has a month-end price but no earlier one, so it gets no open. This matches `Panel.investable` one month earlier, where the asset is not yet listed.

**33. A return of -100% is floored.** The price index multiplies by `1 + r`, and a zero price cannot be traded or divided by. The growth factor is floored at 1e-9, which turns the position into a value of about zero.

**34. Scores become expected returns by the Grinold rule,** `alpha = IC x volatility x z`. The IC is the trailing mean rank IC of the composite score, known at the decision date and floored at zero; volatility is each asset's trailing standard deviation of monthly returns. Rejected: feeding scores to the optimizer directly, which leaves the risk aversion and cost terms without a common unit.

**35. The risk model is deliberately simple.** Exposures are the standardized factors at the decision date, the factor covariance is the Ledoit-Wolf estimate on factor returns realized by then, and idiosyncratic variance is trailing total variance minus the factor part, floored at a quarter of the total. The alpha factors double as risk factors; there is no market, industry or size factor. Dollar neutrality removes a common market move only if betas are equal, which holds in the synthetic panel and not in real data. This has to be revisited before the real-data run.

**36. The optimizer runs inside the engine's decision callback.** It starts from the weights the book has drifted to, so turnover and costs are measured against real holdings. Constraints hold for the targets at the decision; month-end weights drift away from them, so a recorded month-end weight can exceed the position cap.

**37. Before there is enough history, the portfolio stays in cash.** The optimized portfolio needs 12 months for volatility and IC and 24 months of factor returns. No order is sent until then.

**38. Portfolio settings are trials.** Risk aversion, caps, turnover penalty, cost and borrow rates and the quantile count are recorded with the identifier of the combination they were applied to. A combination and one portfolio built on it count as two trials.

**39. Defaults were not tuned.** Risk aversion 5, position cap 2%, gross cap 2, windows of 36 and 60 months. They give a book that uses its caps on the synthetic panel. Choosing them by looking at backtest results would be a trial each.

## Milestone 5: overfitting statistics

**40. Inference on the Sharpe ratio is per period.** The standard error `sqrt((1 - skew*SR + (kurt-1)/4*SR^2) / (n-1))` is stated for the per-period ratio, so everything is computed per month and annualized only for display. The formula assumes returns are not autocorrelated; the bootstrap interval does not.

**41. A trial, for the deflated Sharpe ratio, is a backtest.** `deflated_sharpe_from_ledger` counts the ledger entries whose method starts with `portfolio`, that ran on the same data, and that have a recorded Sharpe ratio. The count and the variance of those Sharpe ratios are read from the file, not passed in. A combination that was scored but never backtested is in the ledger and is not counted here, because it produced no Sharpe ratio to select on. That is a gap: looking at ICs across many combinations and backtesting only the best one understates the count.

**42. The ledger stores results as separate lines.** `record_result` appends the per-period Sharpe ratio and the number of periods for a trial. Trial lines never change, and a repeated run appends an identical result.

**43. With one trial the hurdle is zero.** There is no selection to correct for, and the deflated ratio equals the probabilistic one.

**44. Trials are treated as independent.** The expected-maximum formula assumes independent trials. Variations of one idea are correlated, so the count overstates the effective number of trials and the deflated ratio is conservative. Bailey and Lopez de Prado suggest clustering trials to estimate an effective number; that is not implemented.

**45. A real factor among the trials raises the hurdle.** The variance of trial Sharpe ratios includes the good ones. In the demonstration below the hurdle is 0.87 with a real factor present against 0.44 without. This is conservative, and it follows the definition.

**46. PBO uses contiguous blocks and Sharpe ratios from block sums.** Blocks keep serial dependence inside each block. Each split's Sharpe ratios are computed from per-block sums of returns and squared returns, so 12,870 splits cost a few seconds. The oldest rows are dropped to make the length divisible. PBO is a property of a set of configurations and a selection rule, not of one strategy: it needs the return series of every configuration that was tried.

**47. Bootstrap block length.** The mean block is `n^(1/3)` rounded, about 6 months for 20 years. Rejected for now: the automatic selection of Politis and White (2004), which is the better-founded choice and one more thing to implement and explain. Intervals are percentile intervals with a fixed seed.

**48. Two strategies are compared by resampling the same months for both.** The interval for a Sharpe difference is then much tighter than two separate intervals would suggest when the strategies share most of their risk. A test builds two series with a common component and checks this.

**49. Factor attribution is a regression with Newey-West errors, and the factor data is not loaded yet.** `hac_regression` returns the intercept and loadings; it matches statsmodels to 1e-8. The synthetic panel has no Fama-French factors, so the regression is tested on generated factor returns. Loading the Ken French library moves to milestone 6.

### What selection does, and what the statistics say about it

`uv run python scripts/overfitting_demo.py` (about 15 seconds). Two panels of 300 assets and 240 months. In each, 40 factors are each traded as a long-short quantile portfolio and the best is picked.

| | 40 noise factors | 2 real + 38 noise |
| --- | --- | --- |
| Best factor | `noise_18` | `planted_strong` |
| Its annualized Sharpe ratio | 0.42 | 1.96 |
| 95% bootstrap interval | 0.04 to 0.82 | 1.60 to 2.41 |
| P(true Sharpe > 0), unadjusted | 0.968 | 1.000 |
| Sharpe expected from the best of 40 with no skill | 0.44 | 0.87 |
| Deflated Sharpe ratio | 0.467 | 1.000 |
| Probability of backtest overfitting | 0.615 | 0.000 |

On pure noise the winner passes an unadjusted test at 5% and its bootstrap interval excludes zero. Neither accounts for the search. Its Sharpe ratio is almost exactly what the best of 40 skill-free strategies is expected to show, the deflated ratio is a coin flip, and the in-sample winner lands in the bottom half out of sample more often than not.

## Milestone 6, part 1: structural risk model

This replaces decision 35 as the default. The old model is kept as `risk_model="alpha_factors"` for comparison.

**50. Risk factors are market or industries, beta and styles.** Exposures at the decision date: one dummy per industry when the panel has industry labels (otherwise one column of ones), standardized trailing beta, and each factor standardized. Factor returns come from a cross-sectional regression of next month's returns on these exposures, month by month; the residuals are specific returns. The model at a date is the covariance of factor returns realized so far plus each asset's specific variance.

**51. Industry dummies carry the market.** They add up to one for every asset, so no separate intercept is needed, and beta is standardized so that it measures a tilt around the average asset. Dollar neutrality cancels the market column exactly. It does not cancel a beta tilt, which is why beta is its own factor.

**52. Correlations are shrunk, not covariances.** Factor volatilities differ by two orders of magnitude between an industry and a style. Ledoit-Wolf shrinkage toward a multiple of the identity pulls every variance toward the average: in a test the smallest factor's variance is inflated more than twentyfold. Each factor is divided by its own volatility first, so variances are kept and only correlations move toward zero. The first version of this model used the plain estimator and overstated the risk of a beta-tilted book by about 60%.

**53. Specific variance** is the trailing variance of an asset's regression residuals. Assets with under 12 months get the cross-sectional median, and no asset goes below a quarter of the median.

**54. The regression is unweighted.** Commercial models weight by the square root of market capitalization, because small stocks have noisier returns. The synthetic panel has no size, so this is left for the real-data run.

**55. Industry is one label per asset for the whole sample.** Real companies change industry. This is an approximation to revisit with real data.

### How well the model predicts risk

`uv run python scripts/risk_bias.py`. Bias statistic: the standard deviation across months of realized return over predicted volatility, for a fixed kind of portfolio. A calibrated model gives 1. Above 1 means risk was understated. Panels of 300 assets and 180 months with beta dispersion 0.4 and 8 industries, three seeds.

| Portfolio | Structural | Styles only |
| --- | --- | --- |
| Random long-short | 0.84 to 0.97 | 0.76 to 0.92 |
| Long high beta, short low beta | 0.74 to 0.86 | 1.36 to 1.83 |
| Long one industry, short the rest | 0.77 to 0.86 | 1.42 to 1.57 |

The styles-only model understates the risk of beta and industry tilts by 40% to 80%. The structural model overstates it by 15% to 35%. A likely cause is that factor returns estimated by regression contain estimation noise, which adds to their measured variance while the same noise is also counted as specific risk. That explanation was not tested. The model errs on the cautious side, and the gap is recorded here instead of being tuned away.

### What changes in the portfolio

At the default risk aversion, on the same three panels: mean absolute net beta of the targets is 0.047, 0.048 and 0.036 with the structural model against 0.049, 0.085 and 0.087 with styles only; the largest net industry position is 0.045 to 0.071 against 0.081 to 0.113. Sharpe ratios are within 0.12 of each other in every case. So the model forecasts risk much better and moves the book moderately: in one of the three panels the beta exposure does not change. Neither constraint is hard. If a book has to be beta or industry neutral, that needs a constraint in the optimizer, which `robust-mvo` does not have.

## Milestone 6, part 2: real-data loaders

None of this has been run against WRDS or the Ken French site. The queries follow the published documentation; everything downstream is tested on hand-made tables in the same shape.

**56. The `wrds` package is not a dependency.** Version 3.5.0 requires pandas below 2.3 and this project runs on pandas 3. `scripts/wrds_download.py` runs in its own environment (`uv run --no-project --with wrds`), imports only the query module, and writes three CSV files to `data/wrds`, which is git-ignored. CSV is used because it does not depend on the pandas version on either side. The script has a `--smoke` mode that fetches five years and prints the tables' shapes and columns, to be run first.

**57. Common stock filters use the CIZ flags.** `sharetype = 'NS'`, `securitytype = 'EQTY'`, `securitysubtype = 'COM'`, `usincflg = 'Y'`, `issuertype in ('ACOR', 'CORP')`, on exchanges N, A and Q, with `conditionaltype = 'RW'` and `tradingstatusflg = 'A'`. These replace share codes 10 and 11 and exchange codes 1 to 3 of the old format. The monthly return in this format already includes the delisting return.

**58. A security with a gap keeps its longest unbroken spell.** The panel needs one spell per asset (decision 4). Months outside the longest spell are dropped and counted, and the count is printed by the check script. Rejected: filling gaps with zero returns, which invents data.

**59. Annual accounts become public six months after the fiscal year end** and expire after eighteen. Compustat's annual file has no publication date; six months is the convention behind the June rebalance in Fama and French (1992). The quarterly file has an announcement date and would be more precise; it is not used in v1.

**60. Book-to-market and earnings yield use the current month's market value.** Fama and French divide by the December market value; using the latest price mixes in a little momentum. It is the simpler choice and a known variant.

**61. Every factor is signed so that higher means a higher expected return in its source paper.** Size, reversal, asset growth, accruals, share issuance, volatility and beta enter with a minus sign. Equal weighting relies on this.

**62. Three factors deviate from their papers.** Volatility is total volatility of monthly returns over 36 months, not idiosyncratic volatility from daily returns. Share issuance is annual, from Compustat split-adjusted shares, not monthly from CRSP. Beta is the trailing slope on the equal-weighted market over 60 months, without the shrinkage or the separate volatility and correlation windows of Frazzini and Pedersen.

**63. The research universe is a mask on the panel.** `Panel.universe` marks assets at or above the NYSE 20th size percentile each month. An asset outside it keeps its returns, so a position held when it drops out earns them until it is sold, but it is not investable: it gets no forward return, no score and no place in the risk model.

**64. Industries are the ten SIC divisions,** from the last SIC code on file for each security.

**65. The Ken French reader takes the fetch function as an argument.** Tests pass in zip files written in the library's layout, so no test touches the network. The parser reads the monthly section, stops at the annual one, strips padded column names and turns the missing-value codes into missing values.
