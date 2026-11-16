"""Download CRSP and Compustat from WRDS into a local folder.

The wrds package needs an older pandas than this project, so this script runs in its own
environment and imports only the query module:

    WRDS_USERNAME=<user> uv run --no-project --with wrds scripts/wrds_download.py --smoke
    WRDS_USERNAME=<user> uv run --no-project --with wrds scripts/wrds_download.py

``--smoke`` fetches five years and prints what came back without saving, to check the
table and column names against the account. The full run saves three CSV files to
``data/wrds`` (git-ignored). The wrds package asks for the password itself or reads it
from ~/.pgpass; this script never sees it.

Then, in the project environment:

    uv run python scripts/wrds_check_panel.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import wrds  # noqa: E402

from factorpipe.wrds_data import download, save_raw  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="five years, print only")
    parser.add_argument("--start", default="1960-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--out", default="data/wrds")
    args = parser.parse_args()

    connection = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
    start, end = ("2015-01-01", "2019-12-31") if args.smoke else (args.start, args.end)
    raw = download(connection, start, end)
    for name in ("crsp", "compustat", "links"):
        frame = getattr(raw, name)
        print(f"{name}: {frame.shape[0]:,} rows, columns {list(frame.columns)}")
    if not args.smoke:
        save_raw(raw, args.out)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
