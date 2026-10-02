"""Random-stream isolation (T5/T6) and reference uncertainty models."""

import numpy as np
import pytest

from ration_reliability.errors import LeakageError
from ration_reliability.uncertainty import (
    CORE_STREAMS,
    DrawSet,
    IndependentNormalModel,
    PointMassModel,
    RandomStreams,
    ScenarioSetModel,
    require_stream,
    stream_index,
)

IDS = ("a", "b", "c")
NUTS = ("CP", "NDF")
MEAN = np.array([[0.10, 0.45], [0.18, 0.42], [0.48, 0.12]])
SD = np.array([[0.01, 0.03], [0.02, 0.02], [0.015, 0.01]])
DM = np.array([0.35, 0.88, 0.89])
DSD = np.array([0.02, 0.01, 0.005])


def test_core_stream_indexes_fixed():
    assert CORE_STREAMS == {"opt": 0, "validation": 1, "test": 2, "outer": 3}
    assert stream_index("assay_signal") >= 2 ** 32
    assert stream_index("assay_signal") == stream_index("assay_signal")
    with pytest.raises(ValueError):
        stream_index("a/b")


def test_streams_reproducible_distinct_and_order_independent():
    s = RandomStreams(1103)
    a1 = s.generator("opt").random(5)
    t1 = s.generator("test").random(5)
    # re-create in reverse order -> identical streams
    s2 = RandomStreams(1103)
    t2 = s2.generator("test").random(5)
    a2 = s2.generator("opt").random(5)
    assert np.array_equal(a1, a2) and np.array_equal(t1, t2)
    assert not np.allclose(a1, t1)
    assert not np.allclose(s.generator("outer", 0).random(5), s.generator("outer", 1).random(5))
    assert not np.allclose(RandomStreams(2207).generator("opt").random(5), a1)
    assert s.stream_id("outer", 3) == "root=1103/outer/3"
    ss = s.seed_sequence("validation", 7)
    assert ss.spawn_key == (1, 7) and ss.entropy == 1103


def test_point_mass_and_zero_sd_normal_agree():
    pm = PointMassModel("pm", IDS, NUTS, MEAN, DM, is_synthetic=True)
    nm = IndependentNormalModel("nm", IDS, NUTS, MEAN, np.zeros_like(SD), DM, np.zeros_like(DSD),
                                is_synthetic=True)
    s = RandomStreams(1)
    d1 = pm.draw(s, "test", 50)
    d2 = nm.draw(s, "test", 50)
    assert np.array_equal(d1.theta, d2.theta) and np.array_equal(d1.d, d2.d)
    assert np.all(d1.theta == MEAN[None]) and np.all(d1.d == DM[None])


def test_truncated_normal_respects_bounds_and_nan_stays_missing():
    mean = MEAN.copy()
    mean[2, 1] = np.nan
    nm = IndependentNormalModel("nm", IDS, NUTS, mean, SD * 20, DM, DSD * 20, theta_upper=1.0,
                                is_synthetic=True)
    ds = nm.draw(RandomStreams(3), "opt", 4000)
    th = ds.theta
    assert np.all(np.isnan(th[:, 2, 1]))
    live = th[:, [0, 1], :].reshape(-1)
    assert np.all((live >= 0.0) & (live <= 1.0))
    assert np.all((ds.d > 0) & (ds.d <= 1.0))
    # sample mean close to the mean where truncation is negligible (CP of 'b': 0.18 +/- 0.4 truncated at 0!)
    nm2 = IndependentNormalModel("nm2", IDS, NUTS, MEAN, SD, DM, DSD, is_synthetic=True)
    th2 = nm2.draw(RandomStreams(3), "opt", 20000).theta
    assert np.allclose(th2.mean(axis=0), MEAN, atol=4 * SD.max() / np.sqrt(20000) * 3)


def test_scenario_set_model_resamples_rows_jointly():
    th = np.stack([MEAN, MEAN * 1.1, MEAN * 0.9])
    d = np.stack([DM, DM * 0.95, DM * 1.0])
    m = ScenarioSetModel("set", IDS, NUTS, th, d, is_synthetic=True)
    ds = m.draw(RandomStreams(5), "opt", 200)
    for s in range(ds.n_draws):  # every draw equals one stored joint row
        k = int(np.argmin(np.abs(th - ds.theta[s]).sum(axis=(1, 2))))
        assert np.array_equal(ds.theta[s], th[k]) and np.array_equal(ds.d[s], d[k])


def test_drawset_is_read_only_and_labelled():
    ds = PointMassModel("pm", IDS, NUTS, MEAN, DM).draw(RandomStreams(1), "opt", 3)
    with pytest.raises(ValueError):
        ds.theta[0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        DrawSet(ds.theta, ds.d, "opt", "x", "m", "f", ("a", "b"), NUTS, False)
    r = ds.reordered(ingredient_ids=("c", "a", "b"), nutrient_ids=("NDF", "CP"))
    assert r.theta[0, 0, 0] == MEAN[2, 1] and r.d[0, 0] == DM[2]


def test_require_stream_blocks_leakage():
    ds = PointMassModel("pm", IDS, NUTS, MEAN, DM).draw(RandomStreams(1), "test", 2)
    with pytest.raises(LeakageError):
        require_stream(ds, {"opt"}, "fitting step")
    require_stream(ds, {"test"}, "final evaluation")
