"""Versioned AD declarations preserve the frozen sampling identity."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from lsa.alt.baselines import (
    ABSOLUTE_DISCOUNTING_RULE,
    ABSOLUTE_DISCOUNTING_VERSION,
    evaluate_baseline,
)
from lsa.alt.benchmark import _sampling_spec, validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["protocol.json", "smoke.json"])
@pytest.mark.parametrize("family", ["benchmark_primary", "benchmark_powers"])
def test_declared_rule_preserves_sampling_and_legacy_config(filename, family):
    protocol = json.loads((ROOT / "experiments/alt2027" / filename).read_text())
    config = protocol["experiments"][family]
    checked = validate_config(config)
    assert checked == config
    assert config["absolute_discounting"]["rule"] == ABSOLUTE_DISCOUNTING_RULE
    assert config["absolute_discounting"]["version"] == ABSOLUTE_DISCOUNTING_VERSION
    assert protocol["revision"]["sampling_changed"] is False
    legacy = deepcopy(config)
    legacy.pop("absolute_discounting")
    assert validate_config(legacy) == legacy
    assert _sampling_spec(config) == _sampling_spec(legacy)
    # Missing old metadata never selects the removed discount for a new score.
    prediction = evaluate_baseline(
        "absolute_discounting", [3, 2, 0], target=[0.4, 0.3, 0.3],
        dirichlet_exponents=legacy["dirichlet_exponents"],
    )
    assert prediction.diagnostics["estimator_version"] == ABSOLUTE_DISCOUNTING_VERSION
    assert prediction.diagnostics["discount"] == 0.5


@pytest.mark.parametrize("key", ["rule", "version", "discount", "empty_sample", "full_alphabet"])
def test_mismatched_rule_declaration_rejected(key):
    protocol = json.loads((ROOT / "experiments/alt2027/protocol.json").read_text())
    config = protocol["experiments"]["benchmark_primary"]
    config["absolute_discounting"][key] = "different"
    with pytest.raises(ValueError, match="does not match robust-ad-v1"):
        validate_config(config)
