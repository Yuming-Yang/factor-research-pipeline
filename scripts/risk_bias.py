"""How well each risk model predicts the volatility of three kinds of portfolio.

    uv run python scripts/risk_bias.py

Reproduces the bias statistics in DECISIONS.md (milestone 6, part 1). The bias statistic
is the standard deviation across months of realized return over predicted volatility: 1
is calibrated, above 1 means risk was understated. Takes about 10 seconds.
"""

from __future__ import annotations

import numpy as np
from robust_mvo.covariance import ledoit_wolf
from robust_mvo.risk_model import FactorRiskModel

from factorpipe.combine import factor_returns, standardize
from factorpipe.portfolio import trailing_volatility
from factorpipe.risk import build_risk_history, trailing_beta
from factorpipe.synthetic import SyntheticConfig, make_panel

KINDS = ("random", "beta", "industry")


def bias_statistics(seed: int) -> dict[str, float]:
    config = SyntheticConfig(
        n_assets=300,
        n_months=180,
        n_noise=2,
        reported_factor=None,
        beta_dispersion=0.4,
        n_industries=8,
        seed=seed,
    )
    panel = make_panel(config).panel
    styles = list(panel.characteristics)
    history = build_risk_history(panel, styles)
    investable = panel.investable.to_numpy()
    forward = panel.forward_returns(1).to_numpy()
    z = standardize(panel, styles)
    style_returns = factor_returns(panel, z, styles).to_numpy()
    vol = trailing_volatility(panel).to_numpy()
    beta = trailing_beta(panel).to_numpy()
    first_industry = (panel.industry == "I00").to_numpy()
    rng = np.random.default_rng(0)

    ratios: dict[str, list[float]] = {}
    for i in range(70, 178):
        use = investable[i] & ~np.isnan(vol[i]) & ~np.isnan(beta[i])
        structural = history.model_at(i, use)
        f, _ = ledoit_wolf(style_returns[i - 60 : i])
        b, v = z[i][:, use].T, vol[i][use]
        d = np.maximum(v**2 - np.einsum("nk,kl,nl->n", b, f, b), 0.25 * v**2)
        models = {"structural": structural, "styles only": FactorRiskModel(b, f, d)}
        tilts = {
            "random": rng.standard_normal(use.sum()),
            "beta": beta[i][use],
            "industry": first_industry[use].astype(float),
        }
        for kind, w in tilts.items():
            w = w - w.mean()
            w = w / np.abs(w).sum()
            realized = float(w @ forward[i][use])
            for label, model in models.items():
                predicted = np.sqrt(w @ model.covariance() @ w)
                ratios.setdefault(f"{label}:{kind}", []).append(realized / predicted)
    return {key: float(np.std(values)) for key, values in ratios.items()}


def main() -> None:
    runs = [bias_statistics(seed) for seed in (3, 4, 5)]
    print(f"{'portfolio':<12}{'structural':>16}{'styles only':>16}")
    for kind in KINDS:
        cells = []
        for label in ("structural", "styles only"):
            values = [run[f"{label}:{kind}"] for run in runs]
            cells.append(f"{min(values):.2f} to {max(values):.2f}")
        print(f"{kind:<12}{cells[0]:>16}{cells[1]:>16}")


if __name__ == "__main__":
    main()
