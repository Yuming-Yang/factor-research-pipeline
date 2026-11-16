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
