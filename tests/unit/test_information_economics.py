"""H x T batch conversion, break-even cost, net value and inventory coverage (synthetic numbers)."""

from __future__ import annotations

import numpy as np
import pytest

from ration_reliability.errors import InvalidProblemError
from ration_reliability.information import (
    COST_COMPONENTS,
    BatchCoverage,
    batch_gross_value,
    break_even_max_cost_per_batch,
    inventory_coverage_check,
    net_value_per_batch,
    per_head_day_from_batch_cost,
)
from ration_reliability.information import ValueDefinition

OP = ValueDefinition.OPERATIONAL_DETERMINISTIC_COST_DIFFERENCE   # explicit value definition (second review R1)
#: FIX_A: these arithmetic tests pass bare floats, which must be named explicitly (they are not bound to a result;
#: outputs are labelled caller_declared_unbound_float).  Only this keyword was added; no expected value changed.
UNB = {"unbound_value_declaration": "synthetic H x T arithmetic test (no InformationValueResult)",
       # FIX3_BC (round-3 red team B-1): no money conversion without a complete decision problem; these arithmetic
       # tests run as labelled development diagnostics.  Only this keyword was added; no expected value changed.
       "diagnostic_without_decision_problem": True}


def _cov(h=120.0, t=14.0, validity=None):
    # validity period is required since the red-team fix F11; synthetic default: equal to the covered days
    return BatchCoverage(h, t, "synthetic_test_only", "synthetic_test_only", t if validity is None else validity)


def _inv_ok(c):
    """Inventory check that covers the coverage ``c`` (synthetic large inventory)."""
    return inventory_coverage_check(np.array([[1.0]]), ("x",), c, {"x": 1e12})


def test_gross_value_is_h_times_t_times_delta_c():
    c = _cov(120, 14)
    inv = _inv_ok(c)
    assert inv.fixed_multiplier_valid and inv.heads == 120 and inv.days == 14
    assert c.head_days == 1680.0
    assert batch_gross_value(0.05, c, inventory_check=inv, value_definition=OP, **UNB) == pytest.approx(84.0)
    assert break_even_max_cost_per_batch(0.05, c, inventory_check=inv, value_definition=OP, **UNB) == \
        pytest.approx(84.0)
    # round trip: a batch cost spread over the covered head-days
    assert per_head_day_from_batch_cost(84.0, c, inventory_check=inv) == pytest.approx(0.05)
    # negative per-head-day saving is kept (not truncated): no positive cost is justified
    assert batch_gross_value(-0.01, c, inventory_check=inv, value_definition=OP, **UNB) == pytest.approx(-16.8)
    assert break_even_max_cost_per_batch(0.0, c, inventory_check=inv, value_definition=OP, **UNB) == 0.0


def test_net_value_only_when_every_component_is_known():
    c = _cov(100, 10)
    inv = _inv_ok(c)
    unknown = {k: None for k in COST_COMPONENTS}
    unknown["laboratory"] = 25.0
    r = net_value_per_batch(0.1, c, unknown, inventory_check=inv, value_definition=OP, **UNB)
    assert r.net_value_per_batch is None and r.status == "net_value_unavailable_unknown_costs"
    assert set(r.unknown_components) == set(COST_COMPONENTS) - {"laboratory"}
    assert r.break_even_max_cost_per_batch == pytest.approx(100.0) and r.known_cost_total == 25.0
    known = {k: 5.0 for k in COST_COMPONENTS}
    r2 = net_value_per_batch(0.1, c, known, inventory_check=inv, value_definition=OP, **UNB)
    assert r2.net_value_per_batch == pytest.approx(100.0 - 25.0) and r2.unknown_components == ()
    with pytest.raises(InvalidProblemError, match="not declared"):
        net_value_per_batch(0.1, c, {"laboratory": 25.0}, inventory_check=inv,
                            value_definition=OP, **UNB)                    # forgetting components is not "free"
    with pytest.raises(InvalidProblemError):
        net_value_per_batch(0.1, c, dict(known, sampling=-1.0), inventory_check=inv, value_definition=OP, **UNB)


def test_coverage_validation_and_validity_period():
    with pytest.raises(InvalidProblemError):
        BatchCoverage(0, 10, "sourced", "sourced", 10)
    with pytest.raises(InvalidProblemError):
        BatchCoverage(10, 10, "guess", "sourced", 10)
    with pytest.raises(InvalidProblemError, match="validity period"):
        _cov(100, 30, validity=14)
    assert _cov(100, 14, validity=14).head_days == 1400


def test_inventory_coverage_check_uses_the_largest_ration_of_active_bins():
    ids = ("corn_silage", "soybean_meal")
    Q = np.array([[30.0, 2.0], [28.0, 3.0], [35.0, 2.5]])            # rations used in three bins
    c = _cov(100, 10)
    ok = inventory_coverage_check(Q, ids, c, {"corn_silage": 35_000.0, "soybean_meal": 3_000.0})
    assert ok.status == "covered" and ok.fixed_multiplier_valid
    assert ok.required_kg_as_fed == pytest.approx({"corn_silage": 35_000.0, "soybean_meal": 3_000.0})
    short = inventory_coverage_check(Q, ids, c, {"corn_silage": 32_000.0, "soybean_meal": 3_000.0})
    assert short.status == "not_covered" and short.shortfalls["corn_silage"] == pytest.approx(3_000.0)
    assert not short.fixed_multiplier_valid
    # the 35 kg ration only occurs in a bin with zero probability -> not required
    act = inventory_coverage_check(Q, ids, c, {"corn_silage": 32_000.0, "soybean_meal": 3_000.0},
                                   bin_probabilities=np.array([0.6, 0.4, 0.0]))
    assert act.status == "covered"
    unver = inventory_coverage_check(Q, ids, c, {"corn_silage": 40_000.0, "soybean_meal": None})
    assert unver.status == "unverified" and unver.unverified_ingredients == ("soybean_meal",)
