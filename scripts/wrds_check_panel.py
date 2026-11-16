"""Build a panel from the downloaded WRDS tables and print basic coverage.

    uv run python scripts/wrds_check_panel.py

Reads ``data/wrds`` (written by ``scripts/wrds_download.py``). Prints counts and shares
only; no security-level data.
"""

from __future__ import annotations

import argparse

from factorpipe.evaluation import screen_factors
from factorpipe.wrds_data import FACTORS, build_panel, crsp_panel, load_raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/wrds")
    args = parser.parse_args()

    raw = load_raw(args.data)
    print(f"months dropped outside the longest spell: {crsp_panel(raw.crsp).months_dropped:,}")
    panel = build_panel(raw)
    months, securities = panel.returns.shape
    print(f"panel: {months} months x {securities:,} securities")
    print(f"in the size universe on the last date: {int(panel.universe.iloc[-1].sum()):,}")
    print("share of listed securities with a value on the second-to-last date:")
    listed = panel.listed.iloc[-2]
    for name in FACTORS:
        share = float(panel.characteristics[name].iloc[-2][listed].notna().mean())
        print(f"  {name:<22} {share:6.1%}")
    print(screen_factors(panel)[["rank_ic", "t_stat", "n_months"]].round(3))


if __name__ == "__main__":
    main()
