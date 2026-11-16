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

Mean composite IC over five seeds (500 assets, 240 months, 2 planted and 10 noise factors, first 60 months excluded). The best possible score has an IC of 0.054, and the plain average of all twelve factors 0.020.

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
