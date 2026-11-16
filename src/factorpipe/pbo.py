"""Probability of backtest overfitting, by combinatorially symmetric cross-validation.

Bailey, Borwein, Lopez de Prado and Zhu (2017). The question: if the configuration that
looks best on one half of the sample is picked, how often does it land in the bottom
half of all configurations on the other half?

Procedure. Take a matrix of per-period returns, one column per configuration. Cut the
rows into ``n_blocks`` contiguous blocks of equal length. For every way of choosing half
of the blocks as the in-sample set (the rest is out of sample):

1. compute each configuration's Sharpe ratio in sample and out of sample;
2. take the configuration with the best in-sample Sharpe ratio;
3. find its out-of-sample rank among all ``N`` configurations (1 = worst, ``N`` = best)
   and set ``omega = rank / (N + 1)``, then ``logit = ln(omega / (1 - omega))``.

The probability of backtest overfitting is the share of splits with ``logit <= 0``, that
is, where the in-sample winner is at or below the out-of-sample median. Blocks keep
serial dependence inside each block intact, and every block is used in sample and out of
sample equally often.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PboResult:
    #: Share of splits in which the in-sample winner is at or below the OOS median.
    pbo: float
    #: One logit per split.
    logits: np.ndarray
    #: In-sample and out-of-sample Sharpe ratio of the in-sample winner, per split.
    in_sample: np.ndarray
    out_of_sample: np.ndarray
    n_splits: int
    n_configurations: int


def probability_of_backtest_overfitting(returns: pd.DataFrame, n_blocks: int = 16) -> PboResult:
    """Run the procedure on ``returns`` (periods by configurations).

    Rows with any missing value are dropped, then the oldest rows are dropped so that the
    length divides by ``n_blocks``. ``n_blocks`` must be even; 16 gives 12,870 splits.
    """
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even number, at least 2")
    values = returns.dropna().to_numpy(dtype=float)
    n_configs = values.shape[1]
    if n_configs < 2:
        raise ValueError("need at least two configurations")
    block_len = values.shape[0] // n_blocks
    if block_len < 2:
        raise ValueError("not enough periods for this many blocks")
    values = values[values.shape[0] - block_len * n_blocks :]

    # A Sharpe ratio over any set of blocks follows from per-block sums of r and r**2.
    blocks = values.reshape(n_blocks, block_len, n_configs)
    sums, squares = blocks.sum(axis=1), (blocks**2).sum(axis=1)
    half = n_blocks // 2
    n_obs = half * block_len

    def sharpe(chosen: np.ndarray) -> np.ndarray:
        mean = sums[chosen].sum(axis=0) / n_obs
        variance = (squares[chosen].sum(axis=0) - n_obs * mean**2) / (n_obs - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(variance > 0, mean / np.sqrt(variance), -np.inf)

    all_blocks = np.arange(n_blocks)
    logits, best_in, best_out = [], [], []
    for chosen in combinations(range(n_blocks), half):
        inside = np.array(chosen)
        outside = np.setdiff1d(all_blocks, inside, assume_unique=True)
        sr_in, sr_out = sharpe(inside), sharpe(outside)
        winner = int(np.argmax(sr_in))
        rank = 1 + int((sr_out < sr_out[winner]).sum())
        omega = rank / (n_configs + 1)
        logits.append(np.log(omega / (1.0 - omega)))
        best_in.append(sr_in[winner])
        best_out.append(sr_out[winner])
    logits_array = np.array(logits)
    return PboResult(
        pbo=float((logits_array <= 0).mean()),
        logits=logits_array,
        in_sample=np.array(best_in),
        out_of_sample=np.array(best_out),
        n_splits=len(logits),
        n_configurations=n_configs,
    )
