"""The pipeline's engine must pass every trap in backtest-bias-suite."""

import pytest
from btbias.testing import ALL_TRAPS, assert_passes
from btbias_adapter import PipelineEngineAdapter


@pytest.mark.parametrize("trap", ALL_TRAPS, ids=lambda t: t.id)
def test_engine_passes(trap):
    assert_passes(PipelineEngineAdapter(), trap)


def test_all_eight_traps_are_covered():
    assert [t.id for t in ALL_TRAPS] == ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
