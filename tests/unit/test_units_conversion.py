"""Contract section 8 required test 1: unit / proportion / price-basis conversions (both directions)."""

import numpy as np
import pytest

from ration_reliability.errors import UnitError
from ration_reliability.normalization import units as U


def test_percent_g_per_kg_fraction_roundtrip():
    x = np.array([0.0, 0.35, 16.0, 45.5, 100.0])
    assert np.allclose(U.percent_to_g_per_kg(x), x * 10.0)
    assert np.allclose(U.g_per_kg_to_percent(U.percent_to_g_per_kg(x)), x)
    assert np.allclose(U.percent_to_fraction(x), x / 100.0)
    assert np.allclose(U.fraction_to_percent(U.percent_to_fraction(x)), x)
    assert U.convert(160.0, "g/kg", "%") == pytest.approx(16.0)
    assert U.convert(1.0, "mg/kg", "g/kg") == pytest.approx(1e-3)


def test_dm_as_fed_concentration_roundtrip():
    # 8 % CP on DM of a material with 35 % DM -> 2.8 % as-fed; and back
    assert U.dm_to_as_fed_concentration(8.0, 0.35) == pytest.approx(2.8)
    assert U.as_fed_to_dm_concentration(2.8, 0.35) == pytest.approx(8.0)
    c = np.array([1.0, 20.0, 60.0])
    d = np.array([0.2, 0.5, 0.9])
    assert np.allclose(U.as_fed_to_dm_concentration(U.dm_to_as_fed_concentration(c, d), d), c)


def test_dm_as_fed_mass_roundtrip():
    q = np.array([40.0, 5.0])
    d = np.array([0.40, 0.80])
    x = U.as_fed_mass_to_dm(q, d)
    assert np.allclose(x, [16.0, 4.0])
    assert np.allclose(U.dm_mass_to_as_fed(x, d), q)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.2, np.nan])
def test_dm_fraction_must_be_in_unit_interval(bad):
    with pytest.raises(UnitError):
        U.as_fed_mass_to_dm(1.0, bad)


def test_price_basis_conversions():
    assert U.price_per_tonne_to_per_kg(250.0) == pytest.approx(0.25)
    assert U.convert(250.0, "XXX/t", "XXX/kg") == pytest.approx(0.25)
    assert U.convert(0.25, "XXX/kg", "XXX/t") == pytest.approx(250.0)
    # 1 lb price -> per kg
    assert U.convert(1.0, "XXX/lb", "XXX/kg") == pytest.approx(1.0 / 0.45359237)
    # per kg DM <-> per kg as-fed with an explicit DM
    assert U.price_dm_to_as_fed(0.10, 0.40) == pytest.approx(0.04)
    assert U.price_as_fed_to_dm(0.04, 0.40) == pytest.approx(0.10)
    p = np.array([0.1, 0.4])
    d = np.array([0.4, 0.8])
    assert np.allclose(U.price_as_fed_to_dm(U.price_dm_to_as_fed(p, d), d), p)


def test_energy_conversions_exact_constants():
    assert U.MJ_PER_MCAL == 4.184 and U.LB_IN_KG == 0.45359237
    assert U.mcal_to_mj(1.0) == pytest.approx(4.184)
    assert U.mj_to_mcal(U.mcal_to_mj(1.7)) == pytest.approx(1.7)
    assert U.mcal_per_lb_to_mcal_per_kg(1.0) == pytest.approx(2.2046226218487757, rel=1e-15)
    assert U.mcal_per_kg_to_mcal_per_lb(U.mcal_per_lb_to_mcal_per_kg(0.77)) == pytest.approx(0.77)
    assert U.convert(5.44, "MJ/kg", "Mcal/kg") == pytest.approx(5.44 / 4.184)
    assert U.convert(125.0, "MJ/d", "Mcal/d") == pytest.approx(125.0 / 4.184)


def test_undefined_unit_and_dimension_errors():
    with pytest.raises(UnitError):
        U.get_unit("percent")  # not registered
    with pytest.raises(UnitError):
        U.convert(1.0, "%", "Mcal/kg")  # dimension mismatch
    with pytest.raises(UnitError):
        U.convert(1.0, "USD/t", "CNY/kg")  # currency conversion refused
    with pytest.raises(UnitError):
        U.check_unit("kg/d", "mass_fraction")
    with pytest.raises(UnitError):
        U.check_basis("wet")
    with pytest.raises(UnitError):
        U.get_unit("usd/t")  # lower-case currency not accepted
    assert U.is_defined("KRW/t") and not U.is_defined("kg/head")


def test_canonical_roundtrip_all_registered_units():
    for sym, u in U.UNITS.items():
        v, can = U.to_canonical(3.7, sym)
        assert can == u.canonical
        assert U.from_canonical(v, sym) == pytest.approx(3.7)
