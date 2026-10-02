"""dev_case builder: configs + restricted tables -> engine problem, parameter tables, energy record, build report.

Review round 3 (F6, instruction F-2), R3F, 2026-09-25.  This is the build logic of
``data/restricted_local/dev_case_v1/build_dev_case_v1.py`` (K4 / K4c / FIX_B version, sha256 ``be942251…``, kept byte
for byte in the restricted directory as ``legacy_builder_be942251/build_dev_case_v1.py``, same file name so that
the holder verification finds the bytes recorded by earlier runs), moved into ``src`` so that every
step from the sources to ``dev_case_v1_problem.yaml`` can be read and audited without the restricted data.  The logic
and the order of the checks are unchanged; the outputs are byte-identical to the legacy script's (the comparison is in
``audit/_parts/round3/R3F_build_manifest_preflight_record.md`` and ``tests/unit/test_build_dev_case.py``).

What is where
-------------
* **This module (public)**: the logic, the case structure (ingredient ids, feed-library column names, table layout,
  source ids) and the project's own declared choices (e.g. the weight of the diagnostic row
  ``DIAG-T51-DGC-STARCH-SHARE``).  It contains no NASEM table value.
* **Public text constants** (``PUBLIC_TEXT_CONSTANTS`` below, with page numbers; FIX3_DEF, review round 3 red team
  F-2): the basal FA digestibility (p.25), the p.113 P-absorption rule coefficients, the frame-gain empty-body
  fraction (p.33) and the milk true-protein share of CP (p.30).  They are equation / text constants of the book that
  the project already publishes with page numbers in tracked files (``nutrition/energy.py``,
  ``configs/dev_case_v1/energy.yaml``, ``configs/animal_profile.yaml``, ``reports/model_audit.md``), so keeping them
  in the restricted directory protected nothing.  If the holder-side wrapper still carries them, each must equal the
  public value (checked; a different value is refused).
* **Restricted constants** (``RESTRICTED_CONSTANTS`` of the holder-side wrapper, read with ``ast.literal_eval`` by
  ``load_restricted_constants``; never executed by this module): the Table 3-1 starch digestibilities and the
  table's default for unlisted feeds (p.25) -- table values -- and two dicalcium-phosphate locator notes that quote
  feed-library absorption coefficients.  ``validate_restricted_constants`` lists every missing or malformed entry;
  there are no defaults.
* **Restricted inputs** (read, hashed into the build report): ``nasem_t19_1_core.csv``,
  ``nasem_t19_1_blayer_reconciled.csv``, the 12-row feed-library extract and ``restricted_values.yaml``.
* **Outputs** (restricted, written only under ``data/restricted_local/``): ``energy_linearisation.json``,
  ``dev_case_v1_problem.yaml``, ``ingredient_parameters.csv``, ``uncertainty_cells.csv``, ``build_report.json`` and,
  new in R3F, the sidecar ``build_identity.json`` (builder code, entry point, restricted-constants file, inputs and
  outputs by path and sha256 only).  ``build_report.json`` and the problem YAML keep the K4 builder label
  (``OUTPUT_FORMAT_LABEL``) on purpose: a rebuild is then byte-identical, so the problem hash recorded by earlier runs
  stays valid; the real builder identity is the sidecar.  Files are replaced atomically (temporary file + rename).

Build identity (FIX3_DEF, review round 3 red team F-2)
------------------------------------------------------
* **Scope.**  The builder imports ``nutrition.energy``, ``requirements``, ``constraints``, ``optimization`` (which
  imports the method modules *by name* at run time), ``uncertainty``, ``evaluation`` and ``io.config``.  A static
  import closure would miss the by-name imports, so the builder scope is the direct builder code
  (:func:`builder_code_paths`) **plus every** ``*.py`` **of the** ``src/ration_reliability`` **package**
  (:func:`builder_dependency_paths`).  A change of any of them changes the build identity.
* **Build time.**  :func:`compute_dev_case_build_identity` hashes the scope, the entry point, the constants file and
  the build inputs; :func:`write_build` writes it into the sidecar next to the outputs.
* **Run time.**  :func:`dev_case_build_identity` (what the drivers pass to ``run_record.build_run_record``) reads the
  **build-time sidecar** and verifies it against the current files (:func:`sidecar_drift`); it never presents a
  run-time recomputation as the build.  Without a sidecar, or with drift (builder code, constants, inputs or outputs
  changed since the build), the identity is marked ``usable_for_pilot_official = False`` and
  ``build_run_record`` refuses a pilot / official record; the preflight refuses such runs up front when it is told
  the run type (``preflight_dev_case(..., run_type="pilot")``).  Remedy: freeze the code, run
  ``scripts/build_dev_case.py`` once (byte-identical outputs if the logic did not change; writes the sidecar), then
  run.

``build_dev_case`` builds everything in memory and writes nothing; ``write_build`` writes the outputs and the sidecar.
It is a specification builder and pre-check, not a research run: the nominal M0 solve and the 20,000-draw factory
pre-check are the K4/K4c pre-checks of the defined problem, not results.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

__all__ = [
    "CASE_ID",
    "OUTPUT_FILES",
    "SIDECAR",
    "OUTPUT_FORMAT_LABEL",
    "CONSTANTS_SCHEMA",
    "DevCaseLayout",
    "DEV_CASE_V1",
    "DevCaseBuild",
    "RestrictedConstantsError",
    "load_restricted_constants",
    "validate_restricted_constants",
    "build_input_paths",
    "builder_code_paths",
    "builder_dependency_paths",
    "builder_scope_paths",
    "PUBLIC_TEXT_CONSTANTS",
    "READABLE_SIDECAR_SCHEMAS",
    "STRICT_BUILD_RUN_TYPES",
    "build_dev_case",
    "write_build",
    "compare_build_outputs",
    "dev_case_build_identity",
    "compute_dev_case_build_identity",
    "read_build_sidecar",
    "sidecar_drift",
    "run_build",
]

CASE_ID = "dev_case_v1"
#: output files in the order the legacy script wrote them
OUTPUT_FILES = ("energy_linearisation.json", "dev_case_v1_problem.yaml", "ingredient_parameters.csv",
                "uncertainty_cells.csv", "build_report.json")
SIDECAR = "build_identity.json"
#: builder label inside build_report.json and the problem YAML header (kept for byte identity; see the module doc)
OUTPUT_FORMAT_LABEL = "build_dev_case_v1.py (K4)"
PROBLEM_HEADER = "# RESTRICTED (NASEM values). Built by build_dev_case_v1.py (K4). Not a result.\n"
CONSTANTS_SCHEMA = "ration_reliability.dev_case_builder_constants/1"
CONSTANTS_VARIABLE = "RESTRICTED_CONSTANTS"
BUILD_IDENTITY_SIDECAR_SCHEMA = "ration_reliability.dev_case_build_identity/2"
#: sidecar schemas this module can read (v1 = R3F: builder files + entry point + constants + inputs + outputs)
READABLE_SIDECAR_SCHEMAS = ("ration_reliability.dev_case_build_identity/1", BUILD_IDENTITY_SIDECAR_SCHEMA)
#: the package whose every module belongs to the builder scope (see "Build identity" in the module docstring)
BUILDER_PACKAGE_DIR = "src/ration_reliability"
#: run types for which a build-time sidecar that matches the current files is required
STRICT_BUILD_RUN_TYPES = ("pilot", "official")

# ---- case structure (public: ids and column names only) -------------------------------------------------------------
SRC_NASEM = "SRC-C-NASEM21-T19"
SRC_LIB = "SRC-NASEM-DAIRY-PY-9b0b28e"
SRC_K4 = "SRC-K4-DEVCASE-V1-DERIVATION"
MINERALS = ("limestone_ground", "dicalcium_phosphate")
ITEMS = ("DM", "CP", "NDF", "starch", "EE", "ash", "Ca", "P", "lignin", "TFA")
PROBLEM_NUTRIENTS = ("CP", "NDF", "starch", "EE", "ash", "Ca", "P")
LIB_OF = {"DM": "Fd_DM", "CP": "Fd_CP", "NDF": "Fd_NDF", "starch": "Fd_St", "EE": "Fd_CFat", "ash": "Fd_Ash",
          "Ca": "Fd_Ca", "P": "Fd_P", "lignin": "Fd_Lg", "TFA": "Fd_FA"}
#: ingredient whose DM-only spread is reported by the factory pre-check (step 8)
DM_SPREAD_INGREDIENT = "corn_silage_typical"
#: factory pre-check sampling (K4c; opt stream only; not a run)
PRECHECK_SEED = 1103
PRECHECK_DRAWS = 20000
#: project's declared operationalisation of Table 5-1 "predominant" (model_audit §9.6): weight of the diagnostic row
DGC_WEIGHT_DRY_GROUND_CORN = 0.5
DGC_INGREDIENT = "corn_grain_dry_ground_medium"

#: NASEM (2021) equation / text constants used by the builder -- public, with page numbers (see the module docstring;
#: FIX3_DEF).  Table values (Table 3-1 and its default for unlisted feeds) and feed-library locator notes are *not*
#: here: they stay in the holder-side restricted constants.
PUBLIC_TEXT_CONSTANTS: dict[str, Any] = {
    "fa_digestibility_basal": 0.73,                                    # p.25: base FA digestibility, basal feeds
    "p_absorption_rule": {"inorganic": 0.84, "organic": 0.68},        # p.113: AC = 0.84 Pinorg + 0.68 Porg
    "frame_gain_empty_body_fraction": 0.82,                            # p.33: 1 kg frame gain = 0.82 kg EBW (cow)
    "milk_true_protein_share_of_cp": 0.94,                             # p.30: milk CP = 94 % true protein
}
PUBLIC_TEXT_CONSTANT_PAGES = {"fa_digestibility_basal": "NASEM 2021 p.25", "p_absorption_rule": "NASEM 2021 p.113",
                              "frame_gain_empty_body_fraction": "NASEM 2021 p.33",
                              "milk_true_protein_share_of_cp": "NASEM 2021 p.30"}
#: required restricted fractions (checked by ``validate_restricted_constants``)
_CONSTANT_FRACTIONS = ("starch_digestibility_default",)


class RestrictedConstantsError(ValueError):
    """The restricted constants are missing, unreadable or malformed (every problem is listed)."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class DevCaseLayout:
    """Repository-relative locations of one development case (POSIX paths)."""

    case_id: str = CASE_ID
    cfg_dir: str = "configs/dev_case_v1"
    animal_profile: str = "configs/animal_profile.yaml"
    core_csv: str = "data/restricted_local/nasem_t19_1_core.csv"
    blayer_csv: str = "data/restricted_local/nasem_t19_1_blayer_reconciled.csv"
    case_dir: str = "data/restricted_local/dev_case_v1"
    feed_library_csv: str = "data/restricted_local/dev_case_v1/nasem_feed_library_extract_9b0b28e.csv"
    restricted_values: str = "data/restricted_local/dev_case_v1/restricted_values.yaml"
    #: holder-side wrapper that carries RESTRICTED_CONSTANTS (the legacy entry point)
    constants_file: str = "data/restricted_local/dev_case_v1/build_dev_case_v1.py"
    config_names: tuple[str, ...] = ("animal.yaml", "inventory.yaml", "prices.yaml", "constraints.yaml",
                                     "energy.yaml")


DEV_CASE_V1 = DevCaseLayout()


@dataclass
class DevCaseBuild:
    """In-memory result of ``build_dev_case`` (nothing is written)."""

    outputs: dict[str, bytes]
    report: dict[str, Any]
    summary: dict[str, Any]
    inputs: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.report["failed"]

    def output_sha256(self) -> dict[str, str]:
        return {n: hashlib.sha256(b).hexdigest() for n, b in self.outputs.items()}


# =====================================================================================================================
# restricted constants
# =====================================================================================================================

def _literal_assignment(text: str, name: str) -> Any:
    tree = ast.parse(text)
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            value = node.value
        if name in targets:
            return ast.literal_eval(value)
    raise RestrictedConstantsError([f"no literal assignment {name} = {{...}} found"])


def validate_restricted_constants(c: Any, case_id: str = CASE_ID) -> list[str]:
    """Every problem of the constants mapping (empty list = valid).  Values are never echoed."""
    if not isinstance(c, Mapping):
        return ["restricted constants are not a mapping"]
    errs: list[str] = []
    if c.get("schema") != CONSTANTS_SCHEMA:
        errs.append(f"schema is {c.get('schema')!r}, expected {CONSTANTS_SCHEMA!r}")
    if c.get("case_id") != case_id:
        errs.append(f"case_id is {c.get('case_id')!r}, expected {case_id!r}")

    def frac(key: str, v: Any) -> None:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 < float(v) <= 1.0):
            errs.append(f"{key}: must be a number in (0, 1]")

    t31 = c.get("starch_digestibility_table_3_1")
    if not isinstance(t31, Mapping) or not t31:
        errs.append("starch_digestibility_table_3_1: mapping ingredient_id -> [value, Table 3-1 row label] required")
    else:
        for iid, v in t31.items():
            if not (isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], str) and v[1]):
                errs.append(f"starch_digestibility_table_3_1.{iid}: must be [value, row label]")
            else:
                frac(f"starch_digestibility_table_3_1.{iid}", v[0])
    for key in _CONSTANT_FRACTIONS:
        if key not in c:
            errs.append(f"{key}: missing")
        else:
            frac(key, c[key])
    # public text constants (FIX3_DEF): optional in the restricted file; if present they must equal the public value
    for key, pub in PUBLIC_TEXT_CONSTANTS.items():
        if key not in c:
            continue
        v = c[key]
        if isinstance(pub, Mapping):
            same = isinstance(v, Mapping) and set(v) == set(pub) and all(
                not isinstance(v[k], bool) and isinstance(v[k], (int, float)) and float(v[k]) == float(pub[k])
                for k in pub)
        else:
            same = not isinstance(v, bool) and isinstance(v, (int, float)) and float(v) == float(pub)
        if not same:
            errs.append(f"{key}: differs from the public text constant ({PUBLIC_TEXT_CONSTANT_PAGES[key]}; "
                        "PUBLIC_TEXT_CONSTANTS) -- remove it from the restricted file or correct it")
    loc = c.get("dicalcium_phosphate_ac_locators")
    if not isinstance(loc, Mapping) or set(loc) != {"AC_Ca", "AC_P"} or \
            not all(isinstance(v, str) and v for v in loc.values()):
        errs.append("dicalcium_phosphate_ac_locators: mapping with non-empty strings 'AC_Ca' and 'AC_P' required")
    return errs


def load_restricted_constants(path: str | Path, case_id: str = CASE_ID) -> dict[str, Any]:
    """Read ``RESTRICTED_CONSTANTS`` from the holder-side wrapper (a ``.py`` file, parsed, never executed) or from a
    ``.json`` / ``.yaml`` file with the same mapping.  Raises ``RestrictedConstantsError`` listing every problem."""
    p = Path(path)
    if not p.is_file():
        raise RestrictedConstantsError([f"{p.name}: file not found"])
    text = p.read_text(encoding="utf-8")
    try:
        if p.suffix == ".py":
            c = _literal_assignment(text, CONSTANTS_VARIABLE)
        elif p.suffix == ".json":
            c = json.loads(text)
        else:
            import yaml
            c = yaml.safe_load(text)
    except RestrictedConstantsError:
        raise
    except Exception as exc:  # noqa: BLE001 -- reported, never re-raised with the file content
        raise RestrictedConstantsError([f"{p.name}: unreadable ({type(exc).__name__})"]) from None
    errs = validate_restricted_constants(c, case_id)
    if errs:
        raise RestrictedConstantsError([f"{p.name}: {e}" for e in errs])
    return dict(c)


# =====================================================================================================================
# paths
# =====================================================================================================================

def build_input_paths(repo: str | Path, layout: DevCaseLayout = DEV_CASE_V1) -> list[str]:
    """Repository-relative inputs hashed into ``build_report.json["inputs"]`` (same order as the legacy script:
    restricted tables, then every ``*.yaml`` of the case config directory sorted, then the animal profile)."""
    root = Path(repo)
    cfg = sorted(p.relative_to(root).as_posix() for p in (root / layout.cfg_dir).glob("*.yaml"))
    return [layout.core_csv, layout.blayer_csv, layout.feed_library_csv, layout.restricted_values] + cfg + \
        [layout.animal_profile]


def builder_code_paths() -> list[str]:
    """Repository-relative public builder code proper (this package and the CLI); see :func:`builder_scope_paths`
    for everything the build depends on."""
    return ["src/ration_reliability/build/__init__.py", "src/ration_reliability/build/dev_case.py",
            "scripts/build_dev_case.py"]


def builder_dependency_paths(repo: str | Path) -> list[str]:
    """Every ``*.py`` of the ``src/ration_reliability`` package except :func:`builder_code_paths` (sorted,
    repository-relative; ``__pycache__`` ignored).  The whole package, because the builder's dependencies import
    method modules by name at run time (a static import closure would miss them).  Standard library only."""
    root = Path(repo)
    pkg = root / BUILDER_PACKAGE_DIR
    own = set(builder_code_paths())
    if not pkg.is_dir():
        return []
    out = sorted(p.relative_to(root).as_posix() for p in pkg.rglob("*.py")
                 if "__pycache__" not in p.relative_to(root).parts)
    return [p for p in out if p not in own]


def builder_scope_paths(repo: str | Path) -> list[str]:
    """Builder code proper + its dependencies (:func:`builder_dependency_paths`), repository-relative."""
    return builder_code_paths() + builder_dependency_paths(repo)


def _sha(p: Path) -> Optional[str]:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =====================================================================================================================
# build (logic of the legacy script, steps 1-8, same order of checks)
# =====================================================================================================================

def _check(cond: Any, msg: str, report: dict) -> None:
    report["checks"].append({"ok": bool(cond), "check": msg})
    if not cond:
        report["failed"].append(msg)


def _csv_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _literal_bound(cons: Mapping, cid: str, unit: str) -> float:
    for c in cons["constraints"]:
        if c["id"] == cid:
            b = c["bound"]
            if "value" not in b or str(b.get("unit")) != unit:
                raise ValueError(f"{cid}: a literal bound in {unit!r} is required by the build")
            return float(b["value"])
    raise ValueError(f"{cid}: constraint not found in constraints.yaml")


def build_dev_case(repo: str | Path, constants: Mapping[str, Any], *,
                   layout: DevCaseLayout = DEV_CASE_V1) -> DevCaseBuild:
    """Build every output of the case in memory (nothing is written).

    ``constants`` is the validated restricted mapping (``load_restricted_constants``).  Raises on a missing input or
    an inconsistent configuration exactly where the legacy script raised; failed consistency checks are collected in
    ``report["failed"]`` (``DevCaseBuild.ok``).
    """
    errs = validate_restricted_constants(constants, layout.case_id)
    if errs:
        raise RestrictedConstantsError(errs)
    import numpy as np
    import pandas as pd
    import yaml
    from scipy.optimize import linprog

    from ..evaluation import evaluate
    from ..hashing import stable_hash
    from ..io.config import _UniqueKeyLoader, build_problem   # the loader io.load_problem uses (duplicate keys refused)
    from ..nutrition import energy as E
    from ..nutrition import requirements as R
    from ..nutrition.constraints import linear_rows
    from ..nutrition.requirements import Qty
    from ..optimization import get_method
    from ..uncertainty import RandomStreams, UncertaintySpec, build_uncertainty_model

    root = Path(repo)
    cfg_dir = root / layout.cfg_dir
    core_p, bl_p = root / layout.core_csv, root / layout.blayer_csv
    lib_p, rv_p = root / layout.feed_library_csv, root / layout.restricted_values
    table_3_1 = {k: (float(v[0]), str(v[1])) for k, v in constants["starch_digestibility_table_3_1"].items()}
    st_default = float(constants["starch_digestibility_default"])
    dfa_basal = float(PUBLIC_TEXT_CONSTANTS["fa_digestibility_basal"])              # p.25 (public)
    p_in = float(PUBLIC_TEXT_CONSTANTS["p_absorption_rule"]["inorganic"])          # p.113 (public)
    p_org = float(PUBLIC_TEXT_CONSTANTS["p_absorption_rule"]["organic"])           # p.113 (public)
    ebw_frame = float(PUBLIC_TEXT_CONSTANTS["frame_gain_empty_body_fraction"])     # p.33 (public)
    tp_share = float(PUBLIC_TEXT_CONSTANTS["milk_true_protein_share_of_cp"])       # p.30 (public)
    dcp_loc = dict(constants["dicalcium_phosphate_ac_locators"])
    outputs: dict[str, bytes] = {}

    def y(name: str) -> Any:
        return yaml.safe_load((cfg_dir / name).read_text(encoding="utf-8"))

    report: dict[str, Any] = {"builder": OUTPUT_FORMAT_LABEL, "checks": [], "failed": [], "inputs": {}}
    for rel in build_input_paths(root, layout):
        report["inputs"][rel] = _sha(root / rel) if (root / rel).is_file() else _missing(rel)
    animal_raw, inv, prices, cons, energy_raw = (y(n) for n in layout.config_names)
    rvals = yaml.safe_load(rv_p.read_text(encoding="utf-8"))["values"]
    core = pd.read_csv(core_p)
    bl = pd.read_csv(bl_p)
    lib = {r["UID"]: r for r in csv.DictReader(lib_p.open(encoding="utf-8"))}
    # K4c: the tracked configs do not copy Table 21-1 values (value_from) or restricted values (value_ref)
    animal, log_a = R.resolve_value_refs(animal_raw, repo_root=root)
    energy, log_e = R.resolve_value_refs(energy_raw, repo_root=root, restricted_values=rvals)
    report["value_resolution"] = {"animal.yaml": log_a, "energy.yaml": [
        {k: (v if k != "value" or "value_from" in e else "restricted") for k, v in e.items()} for e in log_e]}

    # ------------------------------------------------------------------ 1 reference cow consistency
    prof = yaml.safe_load((root / layout.animal_profile).read_text(encoding="utf-8"))["fields"]
    cow = animal["reference_cow"]["inputs"]
    pairs = {"body_weight": prof["body_weight_kg"]["value"], "mature_body_weight": prof["mature_body_weight_kg"]["value"],
             "body_condition_score": prof["body_condition_score"]["value"], "days_in_milk": prof["days_in_milk"]["value"],
             "milk_yield": prof["milk_yield_kg_d"]["value"], "milk_fat": prof["milk_fat_pct"]["value"],
             "milk_true_protein": prof["milk_true_protein_pct"]["value"], "milk_lactose": prof["milk_lactose_pct"]["value"],
             "days_pregnant": prof["pregnancy"]["days_pregnant"]["value"],
             "frame_gain": prof["body_weight_change"]["frame_gain_kg_d"]["value"],
             "reserves_gain": prof["body_weight_change"]["reserves_gain_kg_d"]["value"],
             "calf_birth_weight": prof["pregnancy"]["calf_birth_weight_kg"]["value"],
             "body_weight_gain_for_minerals": prof["body_weight_change"]["An_BWgain_kg_d"]["value"]}
    for k, v in pairs.items():
        _check(abs(float(cow[k]["value"]) - float(v)) < 1e-12,
               f"animal.yaml {k} == configs/animal_profile.yaml ({'value_from' if 'value_from' in cow[k] else 'literal'})",
               report)

    # ------------------------------------------------------------------ 2 requirements (calling layer)
    req = R.requirements_from_config(animal)          # raises on derived_expected mismatch
    report["requirements"] = req.summary()
    _check(True, "requirements_from_config reproduced animal.yaml derived_expected within 1e-6", report)
    dmi = req.dmi.value
    cp_hi = _literal_bound(cons, "PN-CP-HI", "%")
    report["restricted_derived"] = {
        "cp_supply_min_equivalent_pct_dm_at_fixed_dmi": 100.0 * float(rvals["cp_supply_min_kg_d"]["value"]) / dmi,
        "pn_cp_hi_minus_equivalent_pct_points": cp_hi - 100.0 * float(rvals["cp_supply_min_kg_d"]["value"]) / dmi}

    # ------------------------------------------------------------------ 3 ingredient parameters
    def core_row(iid, item):
        r = core[(core.ingredient_id == iid) & (core.nutrient_id == item)]
        return None if len(r) == 0 else r.iloc[0]

    def bl_val(iid, item):
        r = bl[(bl.ingredient_id == iid) & (bl.nutrient_id == item)]
        return (float(r.iloc[0]["mean"]), r.iloc[0]) if len(r) else (None, None)

    ingredients = inv["ingredients"]
    ids = [g["ingredient_id"] for g in ingredients]
    params, feeds, lib_checks = [], [], []
    comp: dict[str, dict[str, float]] = {}
    ac: dict[str, dict[str, tuple]] = {}
    for g in ingredients:
        iid = g["ingredient_id"]
        uid = g["nasem_feed"]["id"]
        L = lib[uid]
        comp[iid] = {}
        for item in ITEMS:
            r = core_row(iid, item)
            libv = L[LIB_OF[item]]
            if iid in MINERALS:
                if item == "DM":
                    val, sd, n, st, loc = 100.0, 0.0, None, "sourced", f"Table 19-3 footnote a (p.413); core_id {r.core_id}"
                elif item == "ash":
                    val, sd, n, st, loc = 100.0, 0.0, None, "sourced", f"Table 19-3 footnote e (p.413); core_id {r.core_id}"
                elif item == "Ca":
                    val, sd, n, st, loc = (float(r["mean"]), 0.0, None, "sourced",
                                           f"Table 19-3 p.{int(r.printed_page)}; core_id {r.core_id}")
                elif item == "P" and iid == "dicalcium_phosphate":
                    val, sd, n, st, loc = (float(r["mean"]), 0.0, None, "sourced",
                                           f"Table 19-3 p.{int(r.printed_page)}; core_id {r.core_id}")
                elif item == "P":
                    val, sd, n, st = 0.0, 0.0, None, "research_scenario_assumption"
                    loc = ("Table 19-3 footnote b (p.413): concentrations < 1 % not shown -> P in [0, 1 %); "
                           "conservative end 0 for the lower-bound row PN-P-ABS (inventory.yaml p_content)")
                else:
                    val, sd, n, st = float(libv or 0.0), 0.0, None, "sourced"
                    loc = (f"feed library {uid} {LIB_OF[item]} = {libv} (explicit), consistent with Table 19-3 footnote e "
                           "(ash 100 % of DM)")
            else:
                _check(r is not None and r["entry_agreement"] in ("both_agree", "resolved"),
                       f"{iid}:{item} double-entry core row present and agreed", report)
                val = float(r["mean"])
                sd = None if pd.isna(r["sd"]) else float(r["sd"])
                n = None if pd.isna(r["n"]) else int(r["n"])
                st = "sourced"
                loc = f"Table 19-1 p.{int(r.printed_page)} (pdf p.{int(r.pdf_page)}); core_id {r.core_id}"
                if libv not in ("", None):
                    dec = 2 if item in ("Ca", "P") else 1
                    tol = 0.5 * 10 ** (-dec) + 1e-9
                    lib_checks.append({"ingredient_id": iid, "item": item, "table19_1": val, "feed_library": float(libv),
                                       "abs_diff": abs(val - float(libv)), "tolerance": tol,
                                       "ok": abs(val - float(libv)) <= tol})
            comp[iid][item] = val
            params.append({"ingredient_id": iid, "nasem_id": uid, "item": item, "value": val, "unit": "%",
                           "basis": "as_fed" if item == "DM" else "DM", "sd": sd, "n": n, "status": st,
                           "source_id": SRC_NASEM if st == "sourced" and not loc.startswith("feed library") else
                           (SRC_LIB if st == "sourced" else None), "locator": loc})
        # energy inputs
        if iid in MINERALS:
            rup = drup = 0.0
            dst = (st_default, "default (p.25); no starch")
            rup_loc = "feed library Fd_RUP_base 0 / Fd_dcRUP 0 (no CP)"
        else:
            rup, rrow = bl_val(iid, "RUP")
            drup, drow = bl_val(iid, "dRUP")
            dst = table_3_1.get(iid, (st_default, "default for feeds not listed (p.25)"))
            rup_loc = f"Table 19-1 footnotes c/d, p.{int(rrow.printed_page)} (B-layer {rrow.recon_id}, {drow.recon_id})"
            _check(abs(rup - float(L["Fd_RUP_base"])) <= 0.5 + 1e-9,
                   f"{iid}: RUP Table 19-1 vs feed library within print rounding", report)
            _check(abs(drup - float(L["Fd_dcRUP"])) <= 0.5 + 1e-9,
                   f"{iid}: dRUP Table 19-1 vs feed library within print rounding", report)
        _check(abs(dst[0] - float(L["Fd_dcSt"]) / 100.0) < 1e-9,
               f"{iid}: Table 3-1 starch digestibility == feed library Fd_dcSt", report)
        for item, v, loc2 in (("RUP_pct_CP", rup, rup_loc), ("dRUP_pct_RUP", drup, rup_loc),
                              ("dStarch_base", dst[0], f"Table 3-1 p.25: {dst[1]}"),
                              ("dFA", dfa_basal, "p.25 (basal feeds)")):
            params.append({"ingredient_id": iid, "nasem_id": uid, "item": item, "value": v,
                           "unit": "%" if item.endswith("pct_CP") or item.endswith("pct_RUP") else "fraction",
                           "basis": "none", "sd": 0.0, "n": None, "status": "sourced", "source_id": SRC_NASEM,
                           "locator": loc2})
        feeds.append(E.FeedEnergyInputs(iid, comp[iid]["NDF"] if iid not in MINERALS else 0.0,
                                        comp[iid]["lignin"] if iid not in MINERALS else 0.0,
                                        comp[iid]["starch"] if iid not in MINERALS else 0.0,
                                        comp[iid]["TFA"] if iid not in MINERALS else 0.0,
                                        comp[iid]["CP"] if iid not in MINERALS else 0.0,
                                        comp[iid]["ash"], rup, drup, dst[0], dfa_basal))
        # absorption coefficients
        if iid == "dicalcium_phosphate":
            ca_ac = float(core_row(iid, "Ca_absorption_coefficient")["mean"])
            p_ac = float(core_row(iid, "P_absorption_coefficient")["mean"])
            ac[iid] = {"AC_Ca": (ca_ac, "sourced", SRC_NASEM, dcp_loc["AC_Ca"]),
                       "AC_P": (p_ac, "sourced", SRC_NASEM, dcp_loc["AC_P"])}
        elif iid == "limestone_ground":
            ac[iid] = {"AC_Ca": (float(L["Fd_acCa_input"]), "sourced", SRC_NASEM,
                                 "Table 19-3 p.409 = feed library Fd_acCa_input"),
                       "AC_P": (float(L["Fd_acPtot_input"]), "sourced", SRC_LIB,
                                "feed library Fd_acPtot_input (P content taken as 0, so this AC has no effect)")}
            _check(abs(ac[iid]["AC_Ca"][0] - float(core_row(iid, "Ca_absorption_coefficient")["mean"])) < 1e-12,
                   "limestone Ca AC: feed library == Table 19-3", report)
        else:
            pin, porg = float(L["Fd_Pinorg_P"]), float(L["Fd_Porg_P"])
            p_ac = (p_in * pin + p_org * porg) / 100.0
            _check(abs(pin + porg - 100.0) < 1e-9, f"{iid}: Pinorg + Porg = 100 %", report)
            _check(abs(p_ac - float(L["Fd_acPtot_input"])) < 5e-4, f"{iid}: p.113 rule == feed library Fd_acPtot_input",
                   report)
            ac[iid] = {"AC_Ca": (float(L["Fd_acCa_input"]), "sourced", SRC_LIB,
                                 "feed library Fd_acCa_input; Table 7-1 p.109 / text p.108 class"),
                       "AC_P": (p_ac, "sourced", SRC_LIB,
                                f"p.113 rule {p_in:g} x Pinorg + {p_org:g} x Porg with feed-library fractions "
                                f"{pin:g}/{porg:g} %")}
        for name, (v, st, sid, loc) in ac[iid].items():
            params.append({"ingredient_id": iid, "nasem_id": uid, "item": name, "value": v, "unit": "1", "basis": "none",
                           "sd": 0.0, "n": None, "status": st, "source_id": sid, "locator": loc})
        fw = float(g["forage_weight"]["value"])
        _check(abs(fw - (1.0 - float(L["Fd_Conc"]) / 100.0)) < 1e-12, f"{iid}: forage weight == 1 - Fd_Conc/100 (Eq 20-29)",
               report)
    report["library_vs_table19_1"] = lib_checks
    _check(all(c["ok"] for c in lib_checks), "feed library == Table 19-1 transcription within print rounding (all cells)",
           report)

    # ------------------------------------------------------------------ 4 energy
    es = energy["settings"]
    settings = E.FixedDMISettings(dmi_kg_d=dmi, body_weight_kg=float(es["body_weight_kg"]["value"]),
                                  starch_ref_pct=float(es["starch_ref_pct"]["value"]),
                                  milk_cp_kg_d=float(cow["milk_true_protein"]["value"]) / 100
                                  * float(cow["milk_yield"]["value"]) / tp_share,
                                  body_gain_cp_kg_d=float(es["body_gain_cp_kg_d"]["value"]))
    _check(abs(settings.milk_cp_kg_d - float(es["milk_cp_kg_d"]["value"])) < 1e-6, "energy.yaml milk CP reproduced", report)
    _check(abs(dmi - float(es["dmi_kg_d"]["value"])) < 1e-6, "energy.yaml DMI == configured DMI", report)
    nmap = energy["engine_column"]["derived_from_columns"]
    lin = E.linearise_nel_fixed_dmi(feeds, settings, nutrient_map=nmap)
    nreq = E.nel_requirement_chapter3(
        body_weight=Qty(cow["body_weight"]["value"], "kg"), milk_energy=req.milk_energy.as_qty(),
        days_pregnant=Qty(cow["days_pregnant"]["value"], "d"), calf_birth_weight=Qty(cow["calf_birth_weight"]["value"], "kg"),
        days_in_milk=Qty(cow["days_in_milk"]["value"], "d"), frame_gain=Qty(cow["frame_gain"]["value"], "kg/d"),
        reserves_gain=Qty(cow["reserves_gain"]["value"], "kg/d"),
        mature_body_weight=Qty(cow["mature_body_weight"]["value"], "kg"),
        empty_body_gain_per_frame_gain=Qty(ebw_frame, "1"),
        activity=Qty(energy["requirement"]["components"]["activity"]["value_Mcal_d"], "Mcal/d"))
    _check(abs(nreq.value - float(energy["requirement"]["total_Mcal_d"])) < 1e-6, "energy.yaml NEL requirement reproduced",
           report)
    _check(abs(lin.constant_mcal_d - float(energy["constraint"]["C0_Mcal_d"])) < 1e-6, "energy.yaml C0 reproduced", report)
    _check(abs(nreq.value - lin.constant_mcal_d - float(energy["constraint"]["bound_Mcal_d"])) < 1e-6,
           "energy.yaml bound reproduced", report)
    de_checks = []
    for g in ingredients:
        iid, L = g["ingredient_id"], lib[g["nasem_feed"]["id"]]
        if iid in MINERALS:
            continue
        fl = E.FeedEnergyInputs(iid, float(L["Fd_NDF"]), float(L["Fd_Lg"]), float(L["Fd_St"]), float(L["Fd_FA"]),
                                float(L["Fd_CP"]), float(L["Fd_Ash"]), float(L["Fd_RUP_base"]), float(L["Fd_dcRUP"]),
                                float(L["Fd_dcSt"]) / 100, float(L["Fd_dcFA"]) / 100)
        de_lib = E.feed_de_base_eq3_8(fl).value
        de_tab = E.feed_de_base_eq3_8(feeds[ids.index(iid)]).value
        printed, _ = bl_val(iid, "DE_base")
        de_checks.append({"ingredient_id": iid, "eq3_8_with_library_inputs": de_lib, "library_DE_Base": float(L["Fd_DE_Base"]),
                          "diff_library": de_lib - float(L["Fd_DE_Base"]), "eq3_8_with_table19_1_inputs": de_tab,
                          "table19_1_printed_DE_base": printed, "diff_printed": de_tab - printed})
    report["de_base_checks"] = de_checks
    _check(all(abs(c["diff_library"]) < 2e-3 for c in de_checks),
           "Eq 3-8 implementation reproduces feed-library DE_Base within 0.002 Mcal/kg", report)
    outputs["energy_linearisation.json"] = json.dumps(
        {"linearisation": lin.to_record(), "requirement": nreq.to_dict(), "de_base_checks": de_checks},
        indent=2, default=float).encode("utf-8")

    # ------------------------------------------------------------------ 5 problem YAML
    sources = [
        {"source_id": SRC_NASEM, "title": "NASEM (2021) Nutrient Requirements of Dairy Cattle, 8th rev. ed., doi:10.17226/25806; "
         "local PDF sha256 e03e9415807b5e5f623cc84b54ab772b82f474d0dd874a7020a930154a95d49e; Tables 19-1/19-3 via the "
         "double-entry core table and B-layer; Table 21-3; chapters 3, 5, 7", "is_synthetic": False,
         "license_status": "all_rights_reserved_local_research_use_only",
         "owner": "National Academies of Sciences, Engineering, and Medicine"},
        {"source_id": SRC_LIB, "title": "nasem_dairy (NASEM-Model-Python) feed library NASEM_feed_library.csv at commit "
         "9b0b28ef9a6013ac278bde72e58e01395c2ac6dc (12-row extract in restricted_local)", "is_synthetic": False,
         "license_status": "code_MIT__feed_values_NASEM_derived_restricted_local", "url":
         "https://github.com/CNM-University-of-Guelph/NASEM-Model-Python", "accessed_at": "2026-09-25"},
        {"source_id": SRC_K4, "title": "K4_nutrition_case derivations for dev_case_v1 (energy column by "
         "ration_reliability.nutrition.energy; weights of diagnostic rows)", "is_synthetic": False,
         "license_status": "project_internal", "accessed_at": "2026-09-25"},
    ]
    for s in prices["sources"]:
        sources.append({"source_id": s["source_id"], "title": s["title"], "is_synthetic": False,
                        "license_status": ("us_government_public_domain" if "AMS" in s["source_id"]
                                           else "public_web_page_facts_cited"),
                        "url": s["url"], "accessed_at": "2026-09-25", "owner": s["publisher"]})
    nutrients = [{"nutrient_id": n, "dimension": "mass_fraction", "name": n} for n in PROBLEM_NUTRIENTS]
    nutrients.append({"nutrient_id": E.ENERGY_COLUMN_ID, "dimension": "energy_density",
                      "name": "per-feed NEL contribution at fixed DMI (not a feed NEL value)"})
    prow = {(p["ingredient_id"], p["item"]): p for p in params}

    def vblock(p, unit="%", basis="DM"):
        b = {"value": float(p["value"]), "unit": unit, "basis": basis, "status": p["status"]}
        if p["status"] == "sourced":
            b.update({"source_id": p["source_id"], "locator": p["locator"]})
        else:
            b.update({"rationale": p["locator"]})
        return b

    ing_blocks = []
    for k, g in enumerate(ingredients):
        iid = g["ingredient_id"]
        compb = {n: vblock(prow[(iid, n)]) for n in PROBLEM_NUTRIENTS}
        compb[E.ENERGY_COLUMN_ID] = {"value": float(lin.nominal_density[k]), "unit": "Mcal/kg", "basis": "DM",
                                     "status": "research_scenario_assumption",
                                     "rationale": "derived at table means by energy.linearise_nel_fixed_dmi "
                                                  "(configs/dev_case_v1/energy.yaml); "
                                                  "per-scenario values from EnergyColumnModel / append_energy_column"}
        coefs = {name: ({"value": v, "unit": "1", "status": st, "source_id": sid, "locator": loc} if st == "sourced" else
                        {"value": v, "unit": "1", "status": st, "rationale": loc})
                 for name, (v, st, sid, loc) in ac[iid].items()}
        coefs["dgc_starch_share_w"] = {"value": DGC_WEIGHT_DRY_GROUND_CORN if iid == DGC_INGREDIENT
                                       else -DGC_WEIGHT_DRY_GROUND_CORN, "unit": "1",
                                       "status": "research_scenario_assumption",
                                       "rationale": "weight of DIAG-T51-DGC-STARCH-SHARE: 1{dry ground corn} - 0.5 "
                                                    "(operationalisation of 'predominant', model_audit §9.6)"}
        ing_blocks.append({"ingredient_id": iid, "name_en": g["nasem_feed"]["name"], "category": g["category"],
                           "is_stochastic": bool(g["is_stochastic"]), "original_label": g["nasem_feed"]["name"],
                           "external_ids": {"nasem_feed_id": g["nasem_feed"]["id"]},
                           "group_weights": {"forage": float(g["forage_weight"]["value"])},
                           "dm_estimate": vblock(prow[(iid, "DM")], unit="%", basis="as_fed"),
                           "composition": compb, "coefficients": coefs})
    cblocks = []
    specs_ca, specs_p = R.mineral_constraint_specs(req, numerical_tolerance=1e-6)
    nel_spec_bound = nreq.value - lin.constant_mcal_d
    for c in cons["constraints"]:
        eng = c["engine"]
        if eng["kind"] in ("builtin_variable_bound", "post_hoc"):
            continue
        b = c["bound"]
        if "value_ref" in b:
            key = b["value_ref"].split("#", 1)[1]
            val = float(rvals[key]["value"])
            status, loc = rvals[key]["status"], rvals[key]["locator"]
        else:
            val, status, loc = float(b["value"]), c["status"], c["source"]
        if c["id"] == "PN-CA-ABS":
            _check(abs(val - specs_ca.bound) < 1e-6 and dict(eng["terms"]) == dict(specs_ca.terms),
                   "PN-CA-ABS matches calling layer", report)
        if c["id"] == "PN-P-ABS":
            _check(abs(val - specs_p.bound) < 1e-6 and dict(eng["terms"]) == dict(specs_p.terms),
                   "PN-P-ABS matches calling layer", report)
        if c["id"] == "PN-NEL-FIXEDDMI":
            _check(abs(val - nel_spec_bound) < 1e-6, "PN-NEL-FIXEDDMI bound == NEL_req - C0", report)
            val = float(nel_spec_bound)
        if c["id"] == "SH-DM-PLAN":
            _check(abs(val - dmi) < 1e-6, "SH-DM-PLAN bound == configured DMI", report)
        bb = {"value": val, "unit": eng["unit"], "basis": eng.get("basis", "DM"), "status": status}
        if status == "sourced":
            bb.update({"source_id": SRC_NASEM, "locator": loc})
        else:
            bb.update({"rationale": loc + ("" if "rationale" not in c else "; " + " ".join(str(c["rationale"]).split())[:400])})
        cblocks.append({"constraint_id": c["id"], "name": c["expression"], "kind": eng["kind"], "terms": dict(eng["terms"]),
                        "sense": eng["sense"], "bound": bb, "constraint_class": c["class"],
                        "numerical_tolerance": float(c["numerical_tolerance"]), "dm_source": eng["dm_source"],
                        "claim_scope": str(c.get("claim_scope", ""))[:300]})
    price_items = {}
    for iid in ids:
        it = prices["items"][iid]
        ev = dict(it["engine_value"])
        blk = {"value": float(ev["value"]), "unit": ev["unit"], "basis": ev["basis"], "status": it["status"]}
        if ev["basis"] == "DM":
            blk["dm_basis_semantics"] = ev["dm_basis_semantics"]
        sid = it.get("raw_quote", {}).get("source_id") or it.get("rule_source_id")
        if it["status"] == "sourced":
            blk.update({"source_id": sid, "locator": it["locator"]})
        else:
            blk.update({"source_id": sid, "rationale": " ".join(str(it["rationale"]).split())[:400]})
        price_items[iid] = blk
    cfg = {
        "schema": "ration_reliability.problem/0.1", "problem_id": layout.case_id, "dataset_status": "research_scenario",
        "is_synthetic": False,
        "description": "Development case dev_case_v1 (review round 2, R4). Restricted: contains NASEM values. Not a result.",
        "sources": sources, "nutrients": nutrients, "ingredients": ing_blocks, "constraints": cblocks,
        "prices": {"price_id": prices["price_id"], "currency": prices["currency"], "is_scenario": True,
                   "price_year_or_date": prices["price_period"], "items": price_items,
                   "notes": "configs/dev_case_v1/prices.yaml"},
        "animal_profile": {"profile_id": animal["reference_cow"]["profile_id"], "species": "dairy_cattle",
                           "animal_stage": "stable_lactation",
                           "nutrition_standard_and_version": "NASEM 2021 (version adjudicated by nasem_dairy 9b0b28e)",
                           "dmi_equation": "Eq 2-1 (p.12)",
                           "attributes": {"dmi": {"value": dmi, "unit": "kg/d", "basis": "DM",
                                                  "status": "research_scenario_assumption",
                                                  "rationale": "Eq 2-1 with BCS 3.0 (configs/dev_case_v1/animal.yaml)"}}},
        "run_context": {"protocol_sha256": None, "primary_assumption_id": None, "fit_sets": ["opt", "validation"]},
    }
    prob_text = PROBLEM_HEADER + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    prob_bytes = prob_text.encode("utf-8")
    outputs["dev_case_v1_problem.yaml"] = prob_bytes
    report["problem_yaml_sha256"] = hashlib.sha256(prob_bytes).hexdigest()

    def parse_problem(mode: str):
        # same loader and validator as io.load_problem, on the in-memory text (the legacy script re-read its file)
        return build_problem(yaml.load(prob_text, Loader=_UniqueKeyLoader), mode=mode)  # noqa: S506

    vals = {}
    for mode in ("smoke", "pilot"):
        try:
            _problem, rep = parse_problem(mode)
            vals[mode] = {"ok": rep.ok, "errors": rep.errors, "warnings": rep.warnings, "n_pending": rep.n_pending,
                          "n_assumptions": rep.n_assumptions, "n_synthetic": rep.n_synthetic_values}
        except Exception as exc:  # noqa: BLE001 -- ConfigValidationError lists all issues
            vals[mode] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report["validator"] = vals
    _check(vals["pilot"].get("ok") and vals["pilot"].get("n_pending") == 0,
           "validator: dev problem accepted in pilot mode, 0 pending", report)
    problem, _ = parse_problem("pilot")

    # ------------------------------------------------------------------ 6 parameter tables
    outputs["ingredient_parameters.csv"] = _csv_bytes(params)
    cells = []
    for iid in ids:
        for item in ("DM",) + PROBLEM_NUTRIENTS:
            p = prow[(iid, item)]
            stoch = iid not in MINERALS
            cells.append({"ingredient_id": iid, "component": item, "mean_pct": p["value"],
                          "sd_pct": p["sd"] if stoch else 0.0, "n": p["n"], "lower_pct": 0.0, "upper_pct": 100.0,
                          "is_stochastic": stoch,
                          "variance_basis": "observed_incl_sampling_and_lab" if stoch else "not_applicable",
                          "provenance_status": p["status"], "source_id": p["source_id"], "locator": p["locator"]})
    outputs["uncertainty_cells.csv"] = _csv_bytes(cells)
    _check(all(c["sd_pct"] is not None and not (isinstance(c["sd_pct"], float) and math.isnan(c["sd_pct"])) for c in cells),
           "every stochastic cell of the problem nutrients has an SD", report)

    # ------------------------------------------------------------------ 7 nominal pre-check (not a run)
    res = get_method("M0_nominal")(problem)
    pre: dict[str, Any] = {"status": str(res.status)}
    q = None
    if res.decision is not None:
        q = np.asarray(res.decision.q_as_fed)
        th = problem.nominal_theta()
        ev = evaluate(res.decision, th[None], problem.dm_estimates()[None], problem.compiled, prices=problem.prices)
        cc = problem.compiled
        margins = {cid: float(ev.margin[0, k]) for k, cid in enumerate(ev.constraint_ids)} \
            if hasattr(ev, "constraint_ids") else None
        x = q * problem.dm_estimates()
        nl = E.nonlinear_diet_nel(x, feeds, body_weight_kg=settings.body_weight_kg, milk_cp_kg_d=settings.milk_cp_kg_d,
                                  body_gain_cp_kg_d=settings.body_gain_cp_kg_d)
        pre.update({"objective_usd_per_head_day": float(res.objective),
                    "q_as_fed": dict(zip(problem.ingredient_ids, q.tolist())),
                    "planned_dm_share_pct": dict(zip(problem.ingredient_ids, (100 * x / x.sum()).tolist())),
                    "nominal_margins": margins, "nominal_structural_ok": bool(ev.structural_ok),
                    "nominal_structural_margins": {cid: float(ev.structural_margin[0, k]) if np.ndim(ev.structural_margin) == 2
                                                   else float(ev.structural_margin[k])
                                                   for k, cid in enumerate(ev.structural_ids)},
                    "nominal_joint_violation": bool(ev.joint_violation[0]),
                    "nel_linear_mcal_d": float(x @ lin.nominal_density + lin.constant_mcal_d),
                    "nel_nonlinear_chain": nl, "nel_requirement": nreq.value})
        # NEL headroom: maximise the linear NEL supply subject to every other imposed row at nominal
        rows = []
        for k, cid in enumerate(cc.constraint_ids):
            if cc.classes[k].value == "diagnostic_only" or cid == "PN-NEL-FIXEDDMI":
                continue
            rows.append(k)
        lr = linear_rows(cc.subset(rows), th[None], problem.dm_estimates()[None], problem.dm_estimates())
        A, bvec, iseq = lr.A[0], lr.b, lr.is_eq
        cobj = -(problem.dm_estimates() * lin.nominal_density)
        r2 = linprog(cobj, A_ub=A[~iseq], b_ub=bvec[~iseq], A_eq=A[iseq], b_eq=bvec[iseq],
                     bounds=[(0, None)] * len(q), method="highs")
        if r2.status == 0:
            mx = -r2.fun + lin.constant_mcal_d
            pre["nel_headroom"] = {"max_linear_nel_mcal_d": float(mx), "requirement_mcal_d": nreq.value,
                                   "headroom_pct": 100 * (mx / nreq.value - 1),
                                   "max_nel_planned_dm_share_pct": dict(zip(problem.ingredient_ids,
                                                                             (100 * r2.x * problem.dm_estimates() / dmi)
                                                                             .tolist()))}
    report["nominal_precheck_not_a_run"] = pre
    n_checks_k4 = len(report["checks"])

    # ------------------------------------------------------------------ 8 R3-factory pre-check (K4c; not a run)
    fx: dict[str, Any] = {}
    _check(float(energy["settings"]["starch_ref_pct"]["value"]) == float(rvals["table5_1_starch_max_pct"]["value"]),
           "energy S_ref == PN-T4 bound (same restricted key)", report)
    mm_id = "NASEM2021_T19_1:commercial_lab_results(sampling+analysis error included, not separated; DC-01)"
    core_sha = _sha(core_p)
    tm = np.array([[prow[(iid, n)]["value"] / 100.0 for n in PROBLEM_NUTRIENTS] for iid in ids])
    ts = np.array([[(prow[(iid, n)]["sd"] if iid not in MINERALS else 0.0) / 100.0 for n in PROBLEM_NUTRIENTS]
                   for iid in ids])
    dmn = np.array([prow[(iid, "DM")]["value"] / 100.0 for iid in ids])
    dsd = np.array([(prow[(iid, "DM")]["sd"] if iid not in MINERALS else 0.0) / 100.0 for iid in ids])
    over = {}
    for iid in ids:
        for item in PROBLEM_NUTRIENTS + ("DM",):
            p = prow[(iid, item)]
            over[(iid, item)] = {"data_fingerprint": stable_hash("dev_case_v1_cell/v1", core_sha, iid, item, p["value"],
                                                                 p["sd"], p["n"], p["locator"]),
                                 "provenance_status": p["status"], "source_id": p["source_id"],
                                 "locator": p["locator"]}
    n_comp = len(PROBLEM_NUTRIENTS) + 1
    n_min = sum(1 for iid in ids if iid in MINERALS)
    n_stoch_cells, n_point_cells = (len(ids) - n_min) * n_comp, n_min * n_comp
    try:
        spec = UncertaintySpec.from_arrays(
            "DEV_CASE_V1_PRECHECK_TN_MM", ids, PROBLEM_NUTRIENTS, tm, ts, dmn, dsd, purpose="smoke",
            moment_semantics="target_marginal_moments", is_synthetic=False,
            family_rule={"primary_family": "TN_MM", "fallback_families": ["BETA_MM"], "on_exhausted": "error",
                         "status": "research_scenario_assumption", "selection_basis": "declared_rule",
                         "rationale": "K4c pre-check only (same declared rule as the K3 smoke default); the family "
                                      "choice for runs is K7's / D-19"},
            cell_defaults={"variance_basis": "observed_incl_sampling_and_lab", "decomposition_id": "none",
                           "decomposition_source": None, "measurement_model_id": mm_id},
            theta_bounds=(0.0, 1.0), d_bounds=(0.0, 1.0), cell_overrides=over,
            notes="dev_case_v1 K4c factory pre-check; independent marginals (C0); not a run")
        fmodel = build_uncertainty_model(spec)
        wrapped = E.EnergyColumnModel(fmodel, lin)
        md = fmodel.metadata
        statuses: dict[str, int] = {}
        for c in md.cells:
            statuses[c.fit_status] = statuses.get(c.fit_status, 0) + 1
        fx["fit_status_counts"] = statuses
        fx["families"] = sorted({c.family for c in md.cells if c.family})
        fx["family_counts_matched_cells"] = {f: sum(1 for c in md.cells if c.fit_status == "matched" and c.family == f)
                                             for f in fx["families"]}
        fx["fallback_cells"] = sorted(f"{c.ingredient_id}:{c.component}" for c in md.cells
                                      if c.fit_status == "matched" and c.family != "TN_MM")
        fx["spec_fingerprint"] = spec.fingerprint()
        fx["world_fingerprint"] = wrapped.fingerprint()
        _check(set(statuses) <= {"matched", "point"},
               "factory pre-check: every cell moment-matched or a declared point value", report)
        _check(statuses.get("matched", 0) == n_stoch_cells and statuses.get("point", 0) == n_point_cells,
               f"factory pre-check: {n_stoch_cells} stochastic cells matched, {n_point_cells} mineral point cells", report)
        _check(tuple(wrapped.nutrient_ids) == tuple(problem.nutrient_ids)
               and tuple(wrapped.ingredient_ids) == tuple(problem.ingredient_ids),
               "factory pre-check: wrapped model axes == problem axes (energy column last)", report)
        th0, d0 = wrapped.nominal_state()
        dth = float(np.max(np.abs(th0 - problem.nominal_theta())))
        dd = float(np.max(np.abs(d0 - problem.dm_estimates())))
        fx["max_abs_nominal_theta_diff"] = dth
        fx["max_abs_nominal_d_diff"] = dd
        _check(dth < 1e-12 and dd < 1e-12,
               "factory pre-check: wrapped nominal state == problem nominal values (incl. NEL_fixedDMI)", report)
        # energy-row spread at the nominal q (verifies the audit report's first-order claim; opt stream only)
        if res.decision is not None:
            n_s = PRECHECK_DRAWS
            dw = wrapped.draw(RandomStreams(PRECHECK_SEED), "opt", n_s)
            k_e = list(problem.nutrient_ids).index(E.ENERGY_COLUMN_ID)
            xr = q[None, :] * dw.d                                            # realised DM per draw
            nel = np.einsum("si,si->s", xr, dw.theta[:, :, k_e]) + lin.constant_mcal_d
            xr_fixd = q[None, :] * problem.dm_estimates()[None, :]
            nel_comp = np.einsum("si,si->s", xr_fixd, dw.theta[:, :, k_e]) + lin.constant_mcal_d
            k_cs = ids.index(DM_SPREAD_INGREDIENT)
            dcs = problem.dm_estimates().copy()[None, :].repeat(n_s, 0)
            dcs[:, k_cs] = dw.d[:, k_cs]
            nel_cs = np.einsum("si,si->s", q[None, :] * dcs, np.repeat(problem.nominal_theta()[None, :, k_e], n_s, 0)) \
                + lin.constant_mcal_d
            fx["energy_row_at_nominal_q"] = {
                "stream": dw.stream_id, "n_draws": n_s, "requirement_mcal_d": nreq.value,
                "mean_supply_mcal_d": float(nel.mean()), "sd_supply_mcal_d": float(nel.std(ddof=1)),
                "sd_composition_only_fixed_dm": float(nel_comp.std(ddof=1)),
                "sd_corn_silage_dm_only": float(nel_cs.std(ddof=1)),
                "share_below_requirement": float(np.mean(nel < nreq.value - 1e-6)),
                "note": "pre-check of the nominal M0 q only; not a run, not a result; opt stream"}
    except Exception as exc:  # noqa: BLE001 -- the failure itself is the pre-check result
        fx["error"] = f"{type(exc).__name__}: {exc}"
        _check(False, "factory pre-check: specification/model built", report)
    report["factory_precheck_not_a_run"] = fx
    report["summary"] = {"n_checks": len(report["checks"]), "n_failed": len(report["failed"]),
                         "n_checks_k4_original_steps_1_7": n_checks_k4,
                         "n_checks_k4c_added": len(report["checks"]) - n_checks_k4}
    outputs["build_report.json"] = json.dumps(report, indent=2, ensure_ascii=False, default=float).encode("utf-8")
    summary = {"summary": report["summary"], "failed": report["failed"], "validator": vals,
               "precheck_status": pre.get("status"), "objective": pre.get("objective_usd_per_head_day"),
               "headroom_pct": pre.get("nel_headroom", {}).get("headroom_pct"),
               "factory_precheck": {k: v for k, v in fx.items() if k in ("fit_status_counts", "error")}}
    ordered = {n: outputs[n] for n in OUTPUT_FILES}
    return DevCaseBuild(outputs=ordered, report=report, summary=summary, inputs=dict(report["inputs"]))


def _missing(rel: str) -> str:
    raise FileNotFoundError(f"build input missing: {rel} (run scripts/preflight_dev_case.py for the list and the "
                            "rebuild conditions)")


# =====================================================================================================================
# writing, comparison, identity
# =====================================================================================================================

def _under_restricted(root: Path, out_dir: Path) -> bool:
    rl = (root / "data" / "restricted_local").resolve()
    od = out_dir.resolve()
    return od == rl or rl in od.parents


def write_build(repo: str | Path, build: DevCaseBuild, out_dir: str | Path, *, entry_point: str | Path,
                constants_path: str | Path, layout: DevCaseLayout = DEV_CASE_V1) -> dict[str, Any]:
    """Write the outputs (legacy order) and the ``build_identity.json`` sidecar into ``out_dir``.

    ``out_dir`` must be inside ``data/restricted_local/`` (the outputs carry NASEM values).  Returns the sidecar."""
    root = Path(repo)
    od = Path(out_dir)
    if not od.is_absolute():
        od = root / od
    if not _under_restricted(root, od):
        raise ValueError("out_dir must be inside data/restricted_local/ (the build outputs carry restricted values)")
    od.mkdir(parents=True, exist_ok=True)
    for name, b in build.outputs.items():
        _atomic_write(od / name, b)
    ident = compute_dev_case_build_identity(root, layout=layout, entry_point=entry_point,
                                            constants_path=constants_path, output_dir=od)
    ident["build_ok"] = build.ok
    ident["summary"] = build.report["summary"]
    _atomic_write(od / SIDECAR, (json.dumps(ident, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return ident


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to a temporary file next to ``path`` and rename it over ``path`` (a concurrent reader sees the
    old or the new bytes, never a partial file)."""
    import os
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _json_fields(a: Any, b: Any, path: str = "") -> list[str]:
    """Paths at which two parsed JSON/YAML documents differ (at most 50)."""
    out: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(dict.fromkeys(list(a) + list(b))):
            if k not in a or k not in b:
                out.append(f"{path}/{k} (only in {'second' if k not in a else 'first'})")
            else:
                out += _json_fields(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path} (length {len(a)} vs {len(b)})")
        for k, (x, y) in enumerate(zip(a, b)):
            out += _json_fields(x, y, f"{path}[{k}]")
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
        out.append(path or "/")
    return out[:50]


def compare_build_outputs(expected: Mapping[str, bytes], actual_dir: str | Path) -> dict[str, Any]:
    """Byte-level and field-level comparison of in-memory outputs with the files in ``actual_dir``.

    Field level: JSON and YAML are parsed and compared key by key, CSV row by row and cell by cell (as text).  Only
    paths / counts are reported, never values (the files are restricted)."""
    import yaml
    d = Path(actual_dir)
    per_file = {}
    for name, b in expected.items():
        p = d / name
        rec: dict[str, Any] = {"exists": p.is_file()}
        if p.is_file():
            a = p.read_bytes()
            rec.update({"byte_identical": a == b, "sha256_expected": hashlib.sha256(b).hexdigest(),
                        "sha256_actual": hashlib.sha256(a).hexdigest()})
            if a != b:
                try:
                    if name.endswith(".json"):
                        diffs = _json_fields(json.loads(b), json.loads(a))
                    elif name.endswith(".yaml"):
                        diffs = _json_fields(yaml.safe_load(b.decode("utf-8")), yaml.safe_load(a.decode("utf-8")))
                    else:
                        ra = list(csv.reader(io.StringIO(a.decode("utf-8"), newline="")))
                        rb = list(csv.reader(io.StringIO(b.decode("utf-8"), newline="")))
                        diffs = [f"row {i}" for i, (x, z) in enumerate(zip(rb, ra)) if x != z][:50]
                        if len(ra) != len(rb):
                            diffs.append(f"rows {len(rb)} vs {len(ra)}")
                    rec["field_differences"] = diffs
                    rec["field_identical"] = not diffs
                except Exception as exc:  # noqa: BLE001
                    rec["field_comparison_error"] = type(exc).__name__
                    rec["field_identical"] = False
            else:
                rec["field_identical"] = True
        per_file[name] = rec
    return {"files": per_file,
            "all_byte_identical": all(r.get("byte_identical") for r in per_file.values()),
            "all_field_identical": all(r.get("field_identical") for r in per_file.values())}


def _rel_or_name(root: Path, p: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.name


def compute_dev_case_build_identity(repo: str | Path, *, layout: DevCaseLayout = DEV_CASE_V1,
                                    entry_point: Optional[str | Path] = None,
                                    constants_path: Optional[str | Path] = None,
                                    output_dir: Optional[str | Path] = None) -> dict[str, Any]:
    """Build identity of the case **as the files are now** (``run_record.build_identity`` over the builder scope
    -- :func:`builder_scope_paths` --, the entry point, the restricted constants file, the build inputs and the
    outputs).  Restricted files enter by path and sha256 only.  Used at build time by :func:`write_build`; a run
    record must use :func:`dev_case_build_identity` (the build-time sidecar), not this."""
    from ..io.run_record import build_identity

    root = Path(repo)
    ep = Path(entry_point) if entry_point is not None else root / layout.constants_file
    cp = Path(constants_path) if constants_path is not None else root / layout.constants_file
    od = Path(output_dir) if output_dir is not None else root / layout.case_dir
    ep = ep if ep.is_absolute() else root / ep
    cp = cp if cp.is_absolute() else root / cp
    od = od if od.is_absolute() else root / od
    scope = builder_scope_paths(root)
    scripts: list[Path] = [root / rel for rel in scope]
    seen = set(scope)
    for extra in (ep, cp):
        rel = _rel_or_name(root, extra)
        if rel not in seen:
            scripts.append(extra)
            seen.add(rel)
    inputs = [root / rel for rel in build_input_paths(root, layout)]
    outputs = [od / n for n in OUTPUT_FILES]
    ident = build_identity(root, build_scripts=scripts, build_inputs=inputs, build_outputs=outputs)
    ident["schema_sidecar"] = BUILD_IDENTITY_SIDECAR_SCHEMA
    ident["case_id"] = layout.case_id
    ident["output_format_label"] = OUTPUT_FORMAT_LABEL
    ident["entry_point"] = _rel_or_name(root, ep)
    ident["constants_file"] = _rel_or_name(root, cp)
    ident["output_dir"] = _rel_or_name(root, od)
    ident["builder_scope"] = {"rule": "builder code proper + every *.py of " + BUILDER_PACKAGE_DIR + " (by-name "
                                      "imports of method modules; a static import closure would miss them)",
                              "n_files": len(scope), "builder_code_paths": builder_code_paths()}
    return ident


def read_build_sidecar(repo: str | Path, layout: DevCaseLayout = DEV_CASE_V1) -> tuple[Optional[dict], str]:
    """``(sidecar document or None, status)`` with status ``present`` / ``absent`` / ``unreadable`` /
    ``unknown_schema``.  Standard library only (the preflight uses it)."""
    side = Path(repo) / layout.case_dir / SIDECAR
    if not side.is_file():
        return None, "absent"
    try:
        doc = json.loads(side.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- reported as a status
        return None, "unreadable"
    if not isinstance(doc, dict):
        return None, "unreadable"
    if doc.get("schema_sidecar") is not None and doc.get("schema_sidecar") not in READABLE_SIDECAR_SCHEMAS:
        return doc, "unknown_schema"
    return doc, "present"


def sidecar_drift(repo: str | Path, doc: Mapping[str, Any], layout: DevCaseLayout = DEV_CASE_V1) -> list[dict]:
    """What changed since the build recorded in the sidecar ``doc`` (standard library only).

    One entry ``{"path", "kind", "change"}`` per difference: ``kind`` is ``builder_code`` (a file of the builder
    scope), ``constants`` (the restricted constants file), ``entry_point``, ``build_input`` or ``build_output``;
    ``change`` is ``changed``, ``missing_now`` or ``added_since_build`` (a builder-scope file that did not exist at
    build time).  Entries recorded outside the repository (by name only) are not checked."""
    root = Path(repo)
    constants_rel = str(doc.get("constants_file") or layout.constants_file)
    entry_rel = doc.get("entry_point")
    out: list[dict] = []
    recorded_scripts = set()
    scope_now = builder_scope_paths(root)
    for key, kind in (("build_scripts", "builder_code"), ("build_inputs", "build_input"),
                      ("build_outputs", "build_output")):
        for e in doc.get(key) or []:
            if not isinstance(e, Mapping) or e.get("outside_repository") or "path" not in e:
                continue
            rel = str(e["path"])
            k = kind
            if key == "build_scripts":
                recorded_scripts.add(rel)
                if rel == constants_rel:
                    k = "constants"
                elif rel == entry_rel and rel not in scope_now:
                    k = "entry_point"
            now = _sha(root / rel)
            if now != e.get("sha256"):
                out.append({"path": rel, "kind": k, "change": "missing_now" if now is None else "changed"})
    for rel in scope_now:
        if rel not in recorded_scripts and (root / rel).is_file():
            out.append({"path": rel, "kind": "builder_code", "change": "added_since_build"})
    case_outputs = {f"{layout.case_dir}/{n}" for n in OUTPUT_FILES}
    rec_outputs = {str(e.get("path")) for e in doc.get("build_outputs") or [] if isinstance(e, Mapping)}
    if rec_outputs and not rec_outputs <= case_outputs:
        out.append({"path": str(doc.get("output_dir")), "kind": "build_output",
                    "change": "sidecar_describes_another_output_directory"})
    return out


def dev_case_build_identity(repo: str | Path, *, layout: DevCaseLayout = DEV_CASE_V1) -> dict[str, Any]:
    """Build identity for a **run record**: the build-time sidecar, verified against the current files.

    * sidecar present and no drift: the identity recorded at build time (``source = build_time_sidecar``,
      ``usable_for_pilot_official = True``); its hash is re-derived from the recorded entries as an integrity check;
    * sidecar present with drift (builder code, constants, inputs or outputs changed since the build): the recorded
      identity with ``sidecar.status = stale`` and the drift list, ``usable_for_pilot_official = False``;
    * no readable sidecar (e.g. outputs of the legacy K4 script): an identity computed **now** from the current
      files, labelled ``source = computed_at_run_time`` -- not a record of the build that produced the outputs --,
      ``usable_for_pilot_official = False``.

    ``run_record.build_run_record`` refuses a pilot / official record whose identity is not usable; a smoke / debug
    record keeps it with its labels.  Remedy: run ``scripts/build_dev_case.py`` once on the frozen code."""
    from ..hashing import stable_hash
    from ..io.run_record import BUILD_IDENTITY_SCHEMA

    root = Path(repo)
    rel_side = f"{layout.case_dir}/{SIDECAR}"
    doc, status = read_build_sidecar(root, layout)
    if doc is None or status != "present":
        cur = compute_dev_case_build_identity(root, layout=layout)
        cur.update({"source": "computed_at_run_time",
                    "sidecar": {"path": rel_side, "status": status, "drift": []},
                    "usable_for_pilot_official": False,
                    "provenance_note": "no readable build-time sidecar: this identity describes the files as they are "
                                       "at run time, not the build that produced the outputs (whose builder is not "
                                       "recorded); rebuild with scripts/build_dev_case.py before a pilot/official run"})
        return cur
    drift = sidecar_drift(root, doc, layout)
    ident = {k: v for k, v in doc.items() if k not in ("source", "sidecar", "usable_for_pilot_official")}

    def key(es: Any) -> list[tuple[str, Optional[str]]]:
        return [(str(e["path"]), e.get("sha256")) for e in (es or [])]

    rederived = stable_hash(BUILD_IDENTITY_SCHEMA, key(doc.get("build_scripts")), key(doc.get("build_inputs")))
    integrity = rederived == doc.get("build_identity_sha256")
    ident.update({"source": "build_time_sidecar",
                  "sidecar": {"path": rel_side, "sha256": _sha(root / rel_side),
                              "status": "consistent" if (not drift and integrity) else "stale",
                              "hash_rederived_from_entries": integrity, "drift": drift},
                  "usable_for_pilot_official": bool(not drift and integrity and doc.get("complete", False)
                                                    and doc.get("schema") == BUILD_IDENTITY_SCHEMA)})
    return ident


def iter_required_inputs(repo: str | Path, layout: DevCaseLayout = DEV_CASE_V1) -> Iterable[str]:
    """Build inputs plus the restricted-constants file (repository-relative)."""
    yield from build_input_paths(repo, layout)
    yield layout.constants_file


# =====================================================================================================================
# entry point shared by scripts/build_dev_case.py and the holder-side wrapper
# =====================================================================================================================

def run_build(repo: str | Path, *, constants_path: str | Path, entry_point: str | Path,
              out_dir: Optional[str | Path] = None, check_only: bool = False,
              compare_dir: Optional[str | Path] = None, layout: DevCaseLayout = DEV_CASE_V1,
              stream: Any = None) -> int:
    """Build the case; exit code 0 ok / 1 failed checks or (``check_only``) outputs differ / 2 inputs missing.

    ``check_only``: build in memory and compare byte by byte and field by field with the outputs in ``compare_dir``
    (default: the case directory); nothing is written.  Otherwise write the outputs and ``build_identity.json`` into
    ``out_dir`` (default: the case directory; must be inside ``data/restricted_local/``).
    """
    import sys as _sys
    out = stream or _sys.stdout
    root = Path(repo)
    missing = [rel for rel in list(build_input_paths(root, layout)) + [layout.constants_file]
               if not (root / rel).is_file()]
    for n in layout.config_names:
        rel = f"{layout.cfg_dir}/{n}"
        if not (root / rel).is_file() and rel not in missing:
            missing.append(rel)
    cp = Path(constants_path)
    cp_abs = cp if cp.is_absolute() else root / cp
    if not cp_abs.is_file() and cp_abs.resolve() != (root / layout.constants_file).resolve():
        missing.append(cp.name)
    if missing:
        print("build not started: build inputs missing (run scripts/preflight_dev_case.py for the rebuild "
              "conditions):\n  - " + "\n  - ".join(dict.fromkeys(missing)), file=_sys.stderr)
        return 2
    try:
        constants = load_restricted_constants(cp_abs, layout.case_id)
    except RestrictedConstantsError as exc:
        print("build not started: restricted constants invalid:\n  - " + "\n  - ".join(exc.problems), file=_sys.stderr)
        return 2
    build = build_dev_case(root, constants, layout=layout)
    if check_only:
        cmp_dir = Path(compare_dir) if compare_dir is not None else root / layout.case_dir
        if not cmp_dir.is_absolute():
            cmp_dir = root / cmp_dir
        cmp = compare_build_outputs(build.outputs, cmp_dir)
        rel_dir = cmp_dir.resolve().relative_to(root.resolve()).as_posix() if root.resolve() in cmp_dir.resolve().parents \
            else cmp_dir.name
        print(json.dumps({"mode": "check (in memory, nothing written)", "compared_with": rel_dir,
                          "build_summary": build.report["summary"], "build_failed": build.report["failed"],
                          "all_byte_identical": cmp["all_byte_identical"],
                          "all_field_identical": cmp["all_field_identical"],
                          "files": {n: {k: v for k, v in r.items() if k != "sha256_actual"}
                                    for n, r in cmp["files"].items()}}, indent=2, ensure_ascii=False), file=out)
        return 0 if (cmp["all_byte_identical"] and build.ok) else 1
    od = Path(out_dir) if out_dir is not None else root / layout.case_dir
    ident = write_build(root, build, od, entry_point=entry_point, constants_path=cp, layout=layout)
    print(json.dumps({**build.summary, "build_identity_sha256": ident["build_identity_sha256"],
                      "outputs_written": list(build.outputs) + [SIDECAR]}, indent=2, ensure_ascii=False, default=str),
          file=out)
    return 0 if build.ok else 1
