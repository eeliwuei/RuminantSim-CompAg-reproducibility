"""Monte Carlo interval helpers (T8.2) checked against independent closed forms."""

import math

import pytest

from ration_reliability.evaluation import clopper_pearson, mc_standard_error, one_sided_upper, zero_event_upper_bound


@pytest.mark.parametrize("n", [1, 10, 59, 1000, 10000])
def test_zero_event_bound_equals_one_sided_clopper_pearson(n):
    # Beta(1, n) quantile has the closed form 1 - (1-p)^(1/n)
    assert one_sided_upper(0, n, 0.95) == pytest.approx(zero_event_upper_bound(n, 0.05), rel=1e-10)


def test_clopper_pearson_edges_and_known_value():
    assert clopper_pearson(0, 10) == (0.0, pytest.approx(1 - 0.025 ** (1 / 10)))
    lo, hi = clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 10))
    lo, hi = clopper_pearson(5, 100)
    assert lo < 0.05 < hi
    with pytest.raises(ValueError):
        clopper_pearson(3, 2)


def test_mc_standard_error():
    assert mc_standard_error(0.05, 10000) == pytest.approx(math.sqrt(0.05 * 0.95 / 10000))
