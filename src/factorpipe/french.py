"""Fama-French factor returns from the Ken French data library.

The library publishes each dataset as a zipped CSV. The monthly section starts with a
header line that begins with a comma, followed by one row per month keyed ``YYYYMM``, in
percent. Annual data follows after a blank line and is ignored here.

The download path has not been exercised from this repository's tests, which use files
written in the same layout. ``load_french_factors`` takes the function that fetches a
URL as an argument for that reason.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FIVE_FACTORS = "F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOMENTUM = "F-F_Momentum_Factor_CSV.zip"
COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]

_MONTH_ROW = re.compile(r"^\s*(\d{6})\s*,")


def parse_french_csv(text: str) -> pd.DataFrame:
    """Read the monthly section of a Ken French CSV into decimal returns.

    The index is the month-end date. Column names are stripped of padding. The library
    codes missing values as -99.99 or -999; those become missing.
    """
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.lstrip().startswith(",")), None)
    if header is None:
        raise ValueError("no header line found: expected a line starting with a comma")
    names = [name.strip() for name in lines[header].split(",")[1:]]
    rows, index = [], []
    for line in lines[header + 1 :]:
        match = _MONTH_ROW.match(line)
        if not match:
            if rows:
                break  # the monthly section has ended
            continue
        fields = [field.strip() for field in line.split(",")]
        index.append(pd.Period(match.group(1), freq="M").to_timestamp(how="end").normalize())
        rows.append([float(value) for value in fields[1 : len(names) + 1]])
    if not rows:
        raise ValueError("no monthly rows found")
    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(index), columns=names)
    frame = frame.mask(frame <= -99.0)
    return frame / 100.0


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return response.read()


def _read_zip(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = archive.namelist()[0]
        return archive.read(name).decode("latin-1")


def load_french_factors(
    cache_dir: str | Path, fetch: Callable[[str], bytes] = _fetch
) -> pd.DataFrame:
    """Five factors plus momentum and the risk-free rate, monthly, as decimals.

    Files are downloaded once into ``cache_dir`` and read from there afterwards.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for filename in (FIVE_FACTORS, MOMENTUM):
        path = cache / filename
        if not path.exists():
            path.write_bytes(fetch(BASE_URL + filename))
        frames.append(parse_french_csv(_read_zip(path.read_bytes())))
    five, momentum = frames
    momentum.columns = ["Mom"]
    return five.join(momentum, how="inner")[COLUMNS]
