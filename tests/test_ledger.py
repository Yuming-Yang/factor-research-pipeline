import json

import numpy as np
import pytest

from factorpipe.combine import combine
from factorpipe.ledger import TrialLedger, fingerprint
from factorpipe.synthetic import SyntheticConfig, make_panel

SMALL = SyntheticConfig(n_assets=60, n_months=72, n_noise=2, reported_factor=None, seed=1)


@pytest.fixture
def ledger(tmp_path):
    return TrialLedger(tmp_path / "runs" / "trials.jsonl")


def test_a_new_ledger_is_empty(ledger):
    assert ledger.n_trials == 0
    assert ledger.trials() == []


def test_the_same_trial_is_counted_once(ledger):
    first = ledger.record("icir", {"window": 36}, "data-a")
    second = ledger.record("icir", {"window": 36}, "data-a")
    assert first == second
    assert ledger.n_trials == 1


def test_any_change_is_a_new_trial(ledger):
    ledger.record("icir", {"window": 36}, "data-a")
    ledger.record("icir", {"window": 24}, "data-a")
    ledger.record("mvo", {"window": 36}, "data-a")
    ledger.record("icir", {"window": 36}, "data-b")
    assert ledger.n_trials == 4


def test_parameter_order_does_not_matter(ledger):
    a = ledger.record("ridge", {"window": None, "alphas": (0.0, 1.0)}, "d")
    b = ledger.record("ridge", {"alphas": [0.0, 1.0], "window": None}, "d")
    assert a == b and ledger.n_trials == 1


def test_the_file_only_grows(ledger):
    ledger.record("equal", {}, "d")
    before = ledger.path.read_text()
    ledger.record("icir", {"window": np.int64(12)}, "d")
    after = ledger.path.read_text()
    assert after.startswith(before)
    lines = [json.loads(line) for line in after.splitlines()]
    assert [line["method"] for line in lines] == ["equal", "icir"]
    assert lines[1]["params"] == {"window": 12}


def test_parameters_that_cannot_be_written_are_rejected(ledger):
    with pytest.raises(TypeError, match="cannot be recorded"):
        ledger.record("icir", {"window": object()}, "d")
    assert ledger.n_trials == 0


def test_fingerprint_depends_on_the_data():
    a = make_panel(SMALL).panel
    b = make_panel(SyntheticConfig(**{**SMALL.__dict__, "seed": 2})).panel
    assert fingerprint(a) == fingerprint(make_panel(SMALL).panel)
    assert fingerprint(a) != fingerprint(b)


def test_combining_without_a_ledger_is_an_error():
    panel = make_panel(SMALL).panel
    with pytest.raises(TypeError, match="ledger"):
        combine(panel, "equal")


def test_every_evaluated_configuration_is_in_the_ledger(ledger):
    panel = make_panel(SMALL).panel
    first = combine(panel, "icir", ledger=ledger, window=12)
    assert ledger.n_trials == 1
    combine(panel, "icir", ledger=ledger, window=12)
    assert ledger.n_trials == 1
    combine(panel, "icir", ledger=ledger, window=24)
    combine(panel, "icir", ledger=ledger, window=12, horizon=3)
    combine(panel, "icir", ledger=ledger, window=12, factors=["planted_strong", "noise_00"])
    assert ledger.n_trials == 4
    recorded = {t["id"]: t for t in ledger.trials()}
    assert recorded[first.trial_id]["params"]["window"] == 12
    assert recorded[first.trial_id]["params"]["factors"] == list(panel.characteristics)


def test_a_rejected_call_is_not_recorded(ledger):
    panel = make_panel(SMALL).panel
    with pytest.raises(ValueError, match="unknown method"):
        combine(panel, "magic", ledger=ledger)
    with pytest.raises(ValueError, match="unknown parameters"):
        combine(panel, "equal", ledger=ledger, window=12)
    assert ledger.n_trials == 0
