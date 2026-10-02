"""CLOSE1: helpers and guard of the evaluation-only replay ``experiments/E0_verification/replay_dev_case_eval.py``.

Reads only the replay module (its import reads no data; ``main()`` is not called), so it also runs in the external
reproduction package.  What is checked:

1. the comparison rule: floats pass within ``1e-12 * max(1, |saved|)`` and report bitwise equality separately; counts,
   strings and booleans must be equal; a value present on one side only fails;
2. the replay never solves: the module text references no solver entry point (user rule B-436: evaluation only on
   this machine; the full re-run of dev_case_v1 waits for a server).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_REPLAY = REPO / "experiments" / "E0_verification" / "replay_dev_case_eval.py"


@pytest.fixture(scope="module")
def rp():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("close1_replay_dev_case_eval", _REPLAY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_float_comparison_tolerance_and_exact_flag(rp):
    assert rp.cmp_float(6.4255, "6.4255") == {"abs_diff": 0.0, "exact": True, "pass": True}
    near = rp.cmp_float(5.0 + 1e-13, 5.0)
    assert near["pass"] and not near["exact"]
    assert not rp.cmp_float(5.0 * (1 + 1e-9), 5.0)["pass"]
    assert rp.cmp_float(1e-15, 0.0)["pass"]                      # absolute floor 1e-12 near zero
    assert not rp.cmp_float(1e-11, 0.0)["pass"]
    assert rp.cmp_float(None, "")["pass"]                        # both absent (e.g. structural rows' test metrics)
    assert not rp.cmp_float(None, "0.5")["pass"] and not rp.cmp_float(0.5, "")["pass"]


def test_count_and_exact_comparisons(rp):
    assert rp.cmp_count(392, "392")["pass"] and not rp.cmp_count(392, "393")["pass"]
    assert rp.cmp_count(None, "")["pass"] and not rp.cmp_count(0, "")["pass"]
    assert rp.cmp_exact(rp._bool_str(True), "True")["pass"]
    assert not rp.cmp_exact("root=1103/test", "root=1103/validation")["pass"]


def test_replay_module_never_calls_a_solver():
    text = _REPLAY.read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    calls = re.findall(r"\b(get_method|milp|linprog|solve_box_robust|solve_budget_robust|run_method_block|run_methods|"
                       r"min_violations|largest_feasible|s2_consistency|information_value_block|h1_block|"
                       r"cp_sup_rdp_block)\s*\(", code)
    assert calls == []
