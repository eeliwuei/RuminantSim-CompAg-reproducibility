"""pytest configuration for the ration_reliability engine tests.

* puts ``<repo>/src`` and this directory on ``sys.path`` (no package installation needed);
* disables bytecode writing so the source tree stays clean;
* provides fixtures for the synthetic toy problem (``data/synthetic_test_only``).

All numbers used by the tests are synthetic (``is_synthetic=True``); they test the code, not
any nutritional claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for p in (str(SRC_DIR), str(TESTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

TOY_YAML = REPO_ROOT / "data" / "synthetic_test_only" / "engine_toy_problem_v1.yaml"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def toy_yaml_path() -> Path:
    assert TOY_YAML.is_file(), f"synthetic toy config missing: {TOY_YAML}"
    return TOY_YAML


@pytest.fixture()
def toy_cfg(toy_yaml_path):
    from ration_reliability.io import load_yaml

    return load_yaml(toy_yaml_path)


@pytest.fixture()
def toy_problem(toy_yaml_path):
    from ration_reliability.io import load_problem

    problem, _ = load_problem(toy_yaml_path, mode="unit_test")
    return problem


# ---------------------------------------------------------------------------------------------------------------
# Public tree: tests that read the licensed/restricted inputs, the private study archive or the committed internal
# reports are listed in tests/restricted_input_tests.txt and are skipped here with an explicit reason.  They run
# unchanged in the holder's checkout, where data/restricted_local/ and the internal reports exist.
import pathlib as _pathlib

_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_RESTRICTED_LIST = _ROOT / "tests" / "restricted_input_tests.txt"
_HAS_RESTRICTED = (_ROOT / "data" / "restricted_local").is_dir() and (_ROOT / "sources").is_dir()


def pytest_collection_modifyitems(config, items):
    if _HAS_RESTRICTED or not _RESTRICTED_LIST.exists():
        return
    import pytest as _pytest
    listed = {line.strip() for line in _RESTRICTED_LIST.read_text().splitlines() if line.strip()}
    reason = "needs the restricted inputs or the private study archive (not redistributed; see README 'Data policy')"
    for item in items:
        nodeid = item.nodeid.split("[")[0]
        if item.nodeid in listed or nodeid in listed or any(x.startswith(nodeid + "[") for x in listed):
            item.add_marker(_pytest.mark.skip(reason=reason))
