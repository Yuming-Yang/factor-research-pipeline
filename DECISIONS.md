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
