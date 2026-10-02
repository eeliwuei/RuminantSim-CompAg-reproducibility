"""YAML problem configuration: loader, validator (contract T9) and builder.

Schema ``ration_reliability.problem/0.1`` (see ``docs/ENGINE_API.md`` for a full example)::

    schema: ration_reliability.problem/0.1
    problem_id: <str>
    dataset_status: synthetic_test_only | research_scenario | empirical
    is_synthetic: <bool>
    sources: [{source_id, title, is_synthetic, license_status, [url, owner, accessed_at, lineage_id, notes]}]
    nutrients: [{nutrient_id, dimension: mass_fraction|energy_density, [name]}]
    ingredients:
      - {ingredient_id, name_en, category, is_stochastic,
         dm_estimate: <value>, composition: {<nutrient_id>: <value>},
         [name_zh, original_label, group_weights, coefficients: {<name>: <value unit "1">},
          lineage_id, external_ids, notes]}
    constraints:
      - {constraint_id, kind, terms, sense, bound: <value>, constraint_class, numerical_tolerance,
         [name, dm_source, animal_stage, standard_version, claim_scope, quantity_type, notes]}
    prices: {price_id, currency, is_scenario, items: {<ingredient_id>: <value>}, [price_year_or_date, notes]}
    animal_profile: {profile_id, species, animal_stage, [attributes: {<name>: <value>}, ...]}   # optional
    run_context: {[protocol_sha256, primary_assumption_id, fit_sets]}                          # optional

    <value> = {value, unit, status, [basis, source_id, locator, rationale, dm_basis_semantics]}

Validator rules (errors unless stated)
--------------------------------------
* unknown keys anywhere; duplicate YAML keys; missing required keys;
* undefined units / bases; unit dimension not fitting the field;
* provenance: ``sourced`` needs a declared non-synthetic source + locator; ``research_scenario_assumption``
  needs a rationale; ``pending_user_decision`` needs ``value: null`` and a rationale;
  ``synthetic_test_only`` needs ``is_synthetic: true`` and a synthetic source;
* composition must be on a DM basis (convert upstream with a logged DM); DM estimate is a
  fraction of the as-fed material; per-kg-DM prices need ``dm_basis_semantics`` from
  :data:`DM_PRICE_SEMANTICS` (``settled_on_measured_dm`` is rejected: separate contract scenario);
* ``is_synthetic`` <=> ``dataset_status == synthetic_test_only``; a file under a
  ``synthetic_test_only`` directory must be synthetic; ``empirical`` data may not contain any
  synthetic source/value ("smoke input designated as empirical");
* ``run_context.fit_sets`` may never contain ``test``;
* mode ``pilot``/``official``: no synthetic input, no pending values, no source with unclear
  licence; ``official`` additionally needs ``run_context.protocol_sha256`` and
  ``run_context.primary_assumption_id``;
* constraints must compile (grammar, dimensions, class rules);
* constraint with ``numerical_tolerance`` <= 0 (error; red-team C10);
* warning: research-scenario assumptions present.
* ``pending_override`` (optional key of a ``<value>`` block, B-133): marks a value that a builder
  relabelled from ``pending_user_decision`` to ``research_scenario_assumption`` under an explicit
  override (``original_status``, ``reason``, ``decision_ids``); malformed overrides are errors, and
  such values are refused in ``pilot``/``official`` (the decision is still open).

Builder guard (B-133; ``configs/constraints.yaml`` ``joint_event.dev_start_proposal_scope`` (2))
-----------------------------------------------------------------------------------------------
Config builders that turn a registry entry (e.g. ``configs/constraints.yaml``) into a ``<value>``
block must use :func:`registry_value_block` (or :func:`check_pending_relabel`).  A
``pending_user_decision`` entry is emitted as ``value: null`` with its own status; relabelling it to
``research_scenario_assumption`` raises :class:`PendingRelabelError` unless an explicit
:class:`PendingOverride` (reason + decision ids) is given, and the override is then written into the
block (``pending_override``) so that the validator can see it.  A builder can never upgrade a status
to ``sourced``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from ..datamodel import (
    AnimalProfile,
    ConstraintClass,
    ConstraintKind,
    ConstraintSpec,
    DMSource,
    IngredientRecord,
    NutrientSpec,
    PriceScenario,
    Provenance,
    RationProblem,
    Sense,
    SourcedValue,
    ValueStatus,
)
from ..errors import ConfigValidationError, InvalidProblemError, UnitError
from ..hashing import file_sha256, stable_hash
from ..normalization import units as U

__all__ = [
    "SCHEMA_ID",
    "MODES",
    "DM_PRICE_SEMANTICS",
    "DATASET_STATUSES",
    "ValidationReport",
    "load_yaml",
    "validate_problem_config",
    "build_problem",
    "load_problem",
    "PENDING_OVERRIDE_KEY",
    "PendingOverride",
    "PendingRelabelError",
    "check_pending_relabel",
    "registry_value_block",
]

SCHEMA_ID = "ration_reliability.problem/0.1"
MODES = ("unit_test", "smoke", "pilot", "official")

#: Allowed ``dm_basis_semantics`` of a per-kg-DM price (T2.2; red-team fix C08).
#: ``converted_with_decision_time_dm_estimate``: the quote is per kg DM and is converted once to a
#: fixed price per kg as-fed with the decision-time DM estimate ``d_hat`` (it never varies with draws).
#: ``settled_on_measured_dm``: the supplier bills the measured DM of the delivered batch -- a separate
#: contract scenario, rejected by the default as-fed price model.
DM_PRICE_SEMANTICS = ("converted_with_decision_time_dm_estimate", "settled_on_measured_dm")
DATASET_STATUSES = ("synthetic_test_only", "research_scenario", "empirical")
_LICENSE_UNCLEAR = {None, "", "unknown", "unclear", "pending_user_decision"}

_TOP_REQ = {"schema", "problem_id", "dataset_status", "is_synthetic", "sources", "nutrients", "ingredients",
            "constraints", "prices"}
_TOP_OPT = {"description", "animal_profile", "run_context", "notes"}
_SRC_REQ = {"source_id", "title", "is_synthetic", "license_status"}
_SRC_OPT = {"url", "owner", "accessed_at", "lineage_id", "notes"}
_NUT_REQ = {"nutrient_id", "dimension"}
_NUT_OPT = {"name"}
_ING_REQ = {"ingredient_id", "name_en", "category", "is_stochastic", "dm_estimate", "composition"}
_ING_OPT = {"name_zh", "original_label", "group_weights", "coefficients", "lineage_id", "external_ids", "notes"}
_VAL_REQ = {"value", "unit", "status"}
_VAL_OPT = {"basis", "source_id", "locator", "rationale", "dm_basis_semantics", "pending_override"}
#: Key of a ``<value>`` block that records a builder override of a pending decision (B-133).
PENDING_OVERRIDE_KEY = "pending_override"
_OVR_REQ = {"original_status", "reason", "decision_ids"}
_OVR_OPT = {"approved_by", "notes"}
_CON_REQ = {"constraint_id", "kind", "terms", "sense", "bound", "constraint_class", "numerical_tolerance"}
_CON_OPT = {"name", "dm_source", "animal_stage", "standard_version", "claim_scope", "quantity_type", "notes"}
_PRI_REQ = {"price_id", "currency", "is_scenario", "items"}
_PRI_OPT = {"price_year_or_date", "notes"}
_ANI_REQ = {"profile_id", "species", "animal_stage"}
_ANI_OPT = {"attributes", "nutrition_standard_and_version", "dmi_equation", "notes"}
_RUN_OPT = {"protocol_sha256", "primary_assumption_id", "fit_sets"}


@dataclass
class ValidationReport:
    """Result of validating one configuration."""

    mode: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_pending: int = 0
    n_assumptions: int = 0
    n_synthetic_values: int = 0
    config_sha256: Optional[str] = None
    path: Optional[str] = None
    n_pending_overrides: int = 0

    @property
    def ok(self) -> bool:
        """True when there are no errors."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Plain dict."""
        return {"mode": self.mode, "ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings),
                "n_pending": self.n_pending, "n_assumptions": self.n_assumptions,
                "n_synthetic_values": self.n_synthetic_values, "config_sha256": self.config_sha256,
                "path": self.path, "n_pending_overrides": self.n_pending_overrides}


# --------------------------------------------------------------------------------------------
# builder guard: no silent relabelling of pending decisions (B-133)
# --------------------------------------------------------------------------------------------

class PendingRelabelError(ConfigValidationError):
    """A builder tried to turn a ``pending_user_decision`` value into another status without an override."""


@dataclass(frozen=True)
class PendingOverride:
    """Explicit, recorded permission to use a value whose registry status is ``pending_user_decision``.

    ``reason`` says why the open decision is bypassed (e.g. a development run of the recommended
    default); ``decision_ids`` names the open decisions (e.g. ``("PUD-P2a-03",)``).  The override is
    written into the value block (``pending_override``) and the validator refuses such values in
    ``pilot``/``official`` mode, because the decision itself is still open.
    """

    reason: str
    decision_ids: tuple[str, ...]
    approved_by: Optional[str] = None
    notes: Optional[str] = None

    def issues(self, where: str = "pending_override") -> list[str]:
        """Rule violations (empty list if the override is well formed)."""
        out: list[str] = []
        if not isinstance(self.reason, str) or not self.reason.strip():
            out.append(f"{where}: reason must be a non-empty string")
        ids = self.decision_ids
        if isinstance(ids, str) or not isinstance(ids, (tuple, list)) or not ids \
                or not all(isinstance(x, str) and x.strip() for x in ids):
            out.append(f"{where}: decision_ids must be a non-empty list of decision ids")
        return out

    def to_block(self) -> dict[str, Any]:
        """The ``pending_override`` mapping written into a ``<value>`` block."""
        blk: dict[str, Any] = {"original_status": ValueStatus.PENDING_USER_DECISION.value,
                               "reason": self.reason, "decision_ids": list(self.decision_ids)}
        if self.approved_by:
            blk["approved_by"] = self.approved_by
        if self.notes:
            blk["notes"] = self.notes
        return blk


def _as_override(override: Any, where: str) -> Optional[PendingOverride]:
    if override is None:
        return None
    if isinstance(override, PendingOverride):
        ov = override
    elif isinstance(override, dict):
        unknown = sorted(set(override) - {"reason", "decision_ids", "approved_by", "notes"})
        if unknown:
            raise PendingRelabelError([f"{where}: unknown override keys {unknown}"])
        ids = override.get("decision_ids")
        ov = PendingOverride(reason=override.get("reason"),  # type: ignore[arg-type]
                             decision_ids=tuple(ids) if isinstance(ids, (list, tuple)) else ids,  # type: ignore[arg-type]
                             approved_by=override.get("approved_by"), notes=override.get("notes"))
    else:
        raise PendingRelabelError([f"{where}: override must be a PendingOverride or a mapping"])
    errs = ov.issues(f"{where}.pending_override")
    if errs:
        raise PendingRelabelError(errs)
    return ov


def check_pending_relabel(original_status: Any, target_status: Any, *, override: Any = None,
                          where: str = "value") -> Optional[dict[str, Any]]:
    """Builder-side rule for moving a registry value into a problem configuration.

    Returns the ``pending_override`` block to write (only for an overridden pending -> assumption
    relabel), otherwise ``None``.  Raises :class:`PendingRelabelError` when

    * ``original_status`` is ``pending_user_decision`` and ``target_status`` is
      ``research_scenario_assumption`` without a well-formed ``override`` (silent relabel);
    * ``target_status`` is ``sourced`` while ``original_status`` is not (a builder cannot create a source);
    * an ``override`` is given for anything other than pending -> assumption (it would be meaningless);
    * a status is not a :class:`ValueStatus`.

    ``pending_user_decision`` -> ``synthetic_test_only`` is allowed: it is an explicit test/smoke
    placeholder that the validator keeps out of pilot/official runs.
    """
    try:
        orig = ValueStatus(original_status)
        tgt = ValueStatus(target_status)
    except ValueError as exc:
        raise PendingRelabelError([f"{where}: {exc}"]) from None
    relabel = orig is ValueStatus.PENDING_USER_DECISION and tgt is ValueStatus.RESEARCH_SCENARIO_ASSUMPTION
    if override is not None and not relabel:
        raise PendingRelabelError([f"{where}: an override only applies to pending_user_decision -> "
                                   f"research_scenario_assumption (got {orig.value} -> {tgt.value})"])
    if tgt is ValueStatus.SOURCED and orig is not ValueStatus.SOURCED:
        raise PendingRelabelError([f"{where}: a builder cannot upgrade {orig.value!r} to 'sourced'"])
    if relabel:
        ov = _as_override(override, where)
        if ov is None:
            raise PendingRelabelError([
                f"{where}: registry status is pending_user_decision; relabelling it as research_scenario_assumption "
                "needs an explicit PendingOverride (reason + decision_ids) -- otherwise emit status "
                "pending_user_decision with value null and let the validator refuse it (constraints.yaml "
                "joint_event.dev_start_proposal_scope (2))"])
        return ov.to_block()
    return None


def registry_value_block(registry_status: Any, value: Optional[float], unit: str, *, basis: str = "none",
                         target_status: Any = None, source_id: Optional[str] = None,
                         locator: Optional[str] = None, rationale: Optional[str] = None,
                         override: Any = None, where: str = "value") -> dict[str, Any]:
    """Build a ``<value>`` block from a registry entry, enforcing :func:`check_pending_relabel`.

    ``target_status`` defaults to ``registry_status``.  A pending target always gets ``value: None``
    (a number passed with it is rejected, no placeholder numbers).  An overridden relabel carries the
    ``pending_override`` mapping and a rationale that starts with the override reason.
    """
    tgt = registry_status if target_status is None else target_status
    ovr = check_pending_relabel(registry_status, tgt, override=override, where=where)
    tgt = ValueStatus(tgt)
    if tgt is ValueStatus.PENDING_USER_DECISION:
        if value is not None:
            raise PendingRelabelError([f"{where}: pending_user_decision must be emitted with value null"])
        if not rationale:
            raise PendingRelabelError([f"{where}: pending_user_decision needs a rationale (what must be decided)"])
    elif value is None:
        raise PendingRelabelError([f"{where}: status {tgt.value!r} needs a numeric value"])
    blk: dict[str, Any] = {"value": None if value is None else float(value), "unit": unit, "status": tgt.value,
                           "basis": basis}
    if source_id is not None:
        blk["source_id"] = source_id
    if locator is not None:
        blk["locator"] = locator
    if ovr is not None:
        why = f"PENDING OVERRIDE ({', '.join(ovr['decision_ids'])}): {ovr['reason']}"
        blk["rationale"] = why if not rationale else f"{why}; {rationale}"
        blk[PENDING_OVERRIDE_KEY] = ovr
    elif rationale is not None:
        blk["rationale"] = rationale
    return blk


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys (PyYAML silently keeps the last one)."""


def _construct_mapping(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ConfigValidationError([f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"])
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(path: str | Path) -> Any:
    """Load YAML with the safe loader, rejecting duplicate keys."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass


class _Ctx:
    def __init__(self, mode: str, cfg: dict):
        self.mode = mode
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.n_pending = 0
        self.n_assumptions = 0
        self.n_synth = 0
        self.n_overrides = 0
        self.cfg_synthetic = bool(cfg.get("is_synthetic", False)) if isinstance(cfg, dict) else False
        self.sources: dict[str, dict] = {}

    def keys(self, obj: Any, req: set, opt: set, where: str) -> bool:
        if not isinstance(obj, dict):
            self.errors.append(f"{where}: expected a mapping")
            return False
        missing = sorted(req - set(obj))
        unknown = sorted(set(obj) - req - opt)
        if missing:
            self.errors.append(f"{where}: missing required keys {missing}")
        if unknown:
            self.errors.append(f"{where}: unknown keys {unknown}")
        return not missing


def _value(ctx: _Ctx, blk: Any, where: str, *, dims: Optional[set] = None,
           bases: Optional[set] = None, price: bool = False) -> Optional[SourcedValue]:
    opt = _VAL_OPT if price else (_VAL_OPT - {"dm_basis_semantics"})
    n_err0 = len(ctx.errors)
    if not ctx.keys(blk, _VAL_REQ, opt, where):
        return None
    try:
        status = ValueStatus(blk["status"])
    except ValueError:
        ctx.errors.append(f"{where}: unknown status {blk['status']!r}; allowed {[s.value for s in ValueStatus]}")
        return None
    prov = Provenance(status=status, source_id=blk.get("source_id"), locator=blk.get("locator"),
                      rationale=blk.get("rationale"))
    val = blk.get("value")
    if val is not None:
        try:
            val = float(val)
        except (TypeError, ValueError):
            ctx.errors.append(f"{where}: value must be numeric or null")
            return None
    sv = SourcedValue(value=val, unit=str(blk.get("unit")), provenance=prov, basis=str(blk.get("basis", "none")))
    ctx.errors.extend(sv.issues(where))
    try:
        u = U.get_unit(sv.unit)
        if dims is not None and u.dimension not in dims:
            ctx.errors.append(f"{where}: unit {sv.unit!r} ({u.dimension}) not allowed here; expected {sorted(dims)}")
    except UnitError:
        pass  # already reported by sv.issues
    if bases is not None and sv.basis not in bases:
        ctx.errors.append(f"{where}: basis {sv.basis!r} not allowed here; expected {sorted(bases)}")
    sid = prov.source_id
    if sid is not None and sid not in ctx.sources:
        ctx.errors.append(f"{where}: source_id {sid!r} is not declared in sources")
    if status is ValueStatus.SOURCED and sid in ctx.sources and ctx.sources[sid].get("is_synthetic"):
        ctx.errors.append(f"{where}: status 'sourced' cannot point to a synthetic source")
    if status is ValueStatus.SYNTHETIC_TEST_ONLY:
        ctx.n_synth += 1
        if not ctx.cfg_synthetic:
            ctx.errors.append(f"{where}: synthetic_test_only value in a non-synthetic configuration")
        if sid in ctx.sources and not ctx.sources[sid].get("is_synthetic"):
            ctx.errors.append(f"{where}: synthetic_test_only value must cite a synthetic source")
    if status is ValueStatus.PENDING_USER_DECISION:
        ctx.n_pending += 1
    if status is ValueStatus.RESEARCH_SCENARIO_ASSUMPTION:
        ctx.n_assumptions += 1
    if PENDING_OVERRIDE_KEY in blk:
        _check_override_block(ctx, blk[PENDING_OVERRIDE_KEY], status, where)
    # an invalid value block is reported and never converted/used
    return None if len(ctx.errors) > n_err0 else sv


def _check_override_block(ctx: _Ctx, ob: Any, status: ValueStatus, where: str) -> None:
    """Validate a ``pending_override`` mapping (B-133) and apply the mode rule."""
    w = f"{where}.{PENDING_OVERRIDE_KEY}"
    if not ctx.keys(ob, _OVR_REQ, _OVR_OPT, w):
        return
    if ob.get("original_status") != ValueStatus.PENDING_USER_DECISION.value:
        ctx.errors.append(f"{w}: original_status must be 'pending_user_decision'")
    ids = ob.get("decision_ids")
    ctx.errors.extend(PendingOverride(reason=ob.get("reason"),  # type: ignore[arg-type]
                                      decision_ids=tuple(ids) if isinstance(ids, (list, tuple)) else ids,  # type: ignore[arg-type]
                                      ).issues(w))
    if status is not ValueStatus.RESEARCH_SCENARIO_ASSUMPTION:
        ctx.errors.append(f"{w}: only a research_scenario_assumption value can carry a pending override "
                          f"(status is {status.value!r})")
    ctx.n_overrides += 1
    if ctx.mode in ("pilot", "official"):
        ctx.errors.append(f"{w}: value relabelled from pending_user_decision by a builder override is not allowed "
                          f"in mode {ctx.mode!r}; the decision ({ids}) is still open")
    else:
        ctx.warnings.append(f"{w}: value relabelled from pending_user_decision under an explicit override "
                            f"({ids}); not usable in pilot/official")


def _parse(cfg: Any, mode: str, path: Optional[str | Path]) -> tuple[Optional[RationProblem], ValidationReport]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rep = ValidationReport(mode=mode, path=None if path is None else str(path))
    ctx = _Ctx(mode, cfg if isinstance(cfg, dict) else {})
    if not ctx.keys(cfg, _TOP_REQ, _TOP_OPT, "config"):
        rep.errors, rep.warnings = ctx.errors, ctx.warnings
        return None, rep
    if cfg["schema"] != SCHEMA_ID:
        ctx.errors.append(f"config: schema must be {SCHEMA_ID!r}")
    ds = cfg["dataset_status"]
    if ds not in DATASET_STATUSES:
        ctx.errors.append(f"config: dataset_status must be one of {DATASET_STATUSES}")
    is_syn = cfg["is_synthetic"]
    if not isinstance(is_syn, bool):
        ctx.errors.append("config: is_synthetic must be a boolean")
    if (ds == "synthetic_test_only") != (is_syn is True):
        ctx.errors.append("config: is_synthetic must be true iff dataset_status == synthetic_test_only")
    if path is not None and "synthetic_test_only" in Path(path).parts and is_syn is not True:
        ctx.errors.append("config: files under a synthetic_test_only directory must declare is_synthetic: true")
    if mode in ("pilot", "official") and is_syn is True:
        ctx.errors.append(f"config: synthetic/smoke input cannot be used in mode {mode!r}")

    # --- sources
    srcs = cfg["sources"] if isinstance(cfg["sources"], list) else []
    if not isinstance(cfg["sources"], list) or not srcs:
        ctx.errors.append("config: sources must be a non-empty list (required source is empty)")
    for n, s in enumerate(srcs):
        w = f"sources[{n}]"
        if not ctx.keys(s, _SRC_REQ, _SRC_OPT, w):
            continue
        sid = s["source_id"]
        if sid in ctx.sources:
            ctx.errors.append(f"{w}: duplicate source_id {sid!r}")
        ctx.sources[sid] = s
        if not isinstance(s["is_synthetic"], bool):
            ctx.errors.append(f"{w}: is_synthetic must be boolean")
        if s["is_synthetic"] and ds == "empirical":
            ctx.errors.append(f"{w}: synthetic source in an empirical dataset (smoke input designated as empirical)")
        if mode in ("pilot", "official") and s.get("license_status") in _LICENSE_UNCLEAR:
            ctx.errors.append(f"{w}: licence status unclear ({s.get('license_status')!r}) not allowed in mode {mode!r}")
        if mode in ("pilot", "official") and s["is_synthetic"]:
            ctx.errors.append(f"{w}: synthetic source not allowed in mode {mode!r}")

    # --- nutrients
    nutrients: list[NutrientSpec] = []
    for n, nb in enumerate(cfg["nutrients"] or []):
        w = f"nutrients[{n}]"
        if not ctx.keys(nb, _NUT_REQ, _NUT_OPT, w):
            continue
        try:
            nutrients.append(NutrientSpec(str(nb["nutrient_id"]), str(nb["dimension"]), str(nb.get("name", ""))))
        except InvalidProblemError as exc:
            ctx.errors.append(f"{w}: {exc}")
    nut_dim = {x.nutrient_id: x.dimension for x in nutrients}
    if len(nut_dim) != len(nutrients):
        ctx.errors.append("nutrients: duplicate nutrient_id")

    # --- ingredients
    ingredients: list[IngredientRecord] = []
    for n, ib in enumerate(cfg["ingredients"] or []):
        w = f"ingredients[{n}]"
        if not ctx.keys(ib, _ING_REQ, _ING_OPT, w):
            continue
        iid = str(ib["ingredient_id"])
        w = f"ingredient {iid}"
        prov: dict[str, Provenance] = {}
        dm = _value(ctx, ib["dm_estimate"], f"{w}.dm_estimate", dims={"mass_fraction"}, bases={"as_fed"})
        dm_can = None
        if dm is not None:
            prov["dm_estimate"] = dm.provenance
            dm_can = dm.canonical()
            if dm_can is None:
                ctx.errors.append(f"{w}.dm_estimate: a decision-time DM estimate is required to convert x -> q")
        comp: dict[str, float] = {}
        cb = ib["composition"]
        if not isinstance(cb, dict):
            ctx.errors.append(f"{w}.composition: expected a mapping")
            cb = {}
        for nid, blk in cb.items():
            if nid not in nut_dim:
                ctx.errors.append(f"{w}.composition: nutrient {nid!r} is not declared")
                continue
            unit_dim = "mass_fraction" if nut_dim[nid] == "mass_fraction" else "energy_density"
            sv = _value(ctx, blk, f"{w}.composition.{nid}", dims={unit_dim}, bases={"DM"})
            if sv is None:
                continue
            prov[f"composition:{nid}"] = sv.provenance
            c = sv.canonical()
            comp[nid] = float("nan") if c is None else c
        coefs: dict[str, float] = {}
        cblk = ib.get("coefficients") or {}
        if not isinstance(cblk, dict):
            ctx.errors.append(f"{w}.coefficients: expected a mapping")
            cblk = {}
        for cname, blk in cblk.items():
            sv = _value(ctx, blk, f"{w}.coefficients.{cname}", dims={"dimensionless"})
            if sv is None:
                continue
            if sv.value is None:
                ctx.errors.append(f"{w}.coefficients.{cname}: a pending coefficient cannot be used; omit it")
                continue
            prov[f"coefficient:{cname}"] = sv.provenance
            coefs[str(cname)] = float(sv.value)
        gw = ib.get("group_weights") or {}
        if not isinstance(gw, dict):
            ctx.errors.append(f"{w}.group_weights: expected a mapping")
            gw = {}
        if not isinstance(ib["is_stochastic"], bool):
            ctx.errors.append(f"{w}.is_stochastic must be boolean")
        if dm_can is None:
            continue
        try:
            ingredients.append(IngredientRecord(
                ingredient_id=iid, name_en=str(ib["name_en"]), dm_estimate=dm_can, composition=comp,
                category=str(ib["category"]), group_weights={str(k): float(v) for k, v in gw.items()},
                coefficients=coefs,
                name_zh=ib.get("name_zh"), original_label=ib.get("original_label"),
                is_stochastic=bool(ib["is_stochastic"]), is_synthetic=bool(is_syn), lineage_id=ib.get("lineage_id"),
                external_ids=dict(ib.get("external_ids") or {}), provenance=prov, notes=str(ib.get("notes", ""))))
        except (InvalidProblemError, ValueError, TypeError) as exc:
            ctx.errors.append(f"{w}: {exc}")
    ing_dm = {g.ingredient_id: g.dm_estimate for g in ingredients}

    # --- constraints
    specs: list[ConstraintSpec] = []
    for n, cb in enumerate(cfg["constraints"] or []):
        w = f"constraints[{n}]"
        if not ctx.keys(cb, _CON_REQ, _CON_OPT, w):
            continue
        cid = str(cb["constraint_id"])
        w = f"constraint {cid}"
        sv = _value(ctx, cb["bound"], f"{w}.bound")
        if sv is None:
            continue
        try:
            spec = ConstraintSpec(
                constraint_id=cid, name=str(cb.get("name", cid)), kind=ConstraintKind(cb["kind"]),
                terms={str(k): float(v) for k, v in dict(cb["terms"]).items()}, sense=Sense(cb["sense"]),
                bound=sv.value, unit=sv.unit, constraint_class=ConstraintClass(cb["constraint_class"]),
                numerical_tolerance=float(cb["numerical_tolerance"]), provenance=sv.provenance, basis=sv.basis,
                dm_source=DMSource(cb.get("dm_source", "scenario")), animal_stage=cb.get("animal_stage"),
                standard_version=cb.get("standard_version"), claim_scope=cb.get("claim_scope"),
                quantity_type=str(cb.get("quantity_type", "model_quantity")), notes=str(cb.get("notes", "")))
        except (ValueError, TypeError) as exc:
            ctx.errors.append(f"{w}: {exc}")
            continue
        if not float(spec.numerical_tolerance) > 0:
            ctx.errors.append(f"{w}: numerical_tolerance must be > 0; with 0 floating-point rounding at binding "
                              f"constraints counts as violations (red-team C10)")
            continue
        specs.append(spec)

    # --- prices
    pb = cfg["prices"]
    price_obj: Optional[PriceScenario] = None
    if ctx.keys(pb, _PRI_REQ, _PRI_OPT, "prices"):
        cur = str(pb["currency"])
        if not (len(cur) == 3 and cur.isalpha() and cur.isupper()):
            ctx.errors.append("prices.currency must be a 3-letter upper-case code (XXX for synthetic)")
        if not isinstance(pb["is_scenario"], bool):
            ctx.errors.append("prices.is_scenario must be boolean")
        items = pb["items"] if isinstance(pb["items"], dict) else {}
        vals: dict[str, float] = {}
        pprov: dict[str, Provenance] = {}
        log: list[str] = []
        for iid, blk in items.items():
            w = f"prices.items.{iid}"
            if iid not in ing_dm:
                ctx.errors.append(f"{w}: unknown ingredient")
                continue
            sv = _value(ctx, blk, w, dims={"price_per_mass"}, bases={"as_fed", "DM"}, price=True)
            if sv is None or sv.value is None:
                if sv is not None:
                    ctx.errors.append(f"{w}: price value is required to build the problem")
                continue
            ucur = U.get_unit(sv.unit).note
            if ucur != cur:
                ctx.errors.append(f"{w}: currency {ucur!r} differs from prices.currency {cur!r}")
                continue
            v = float(U.convert(sv.value, sv.unit, f"{cur}/kg"))
            if sv.unit != f"{cur}/kg":
                log.append(f"{iid}: {sv.value:g} {sv.unit} -> {v:g} {cur}/kg ({sv.basis})")
            if sv.basis == "DM":
                sem = blk.get("dm_basis_semantics")
                if not sem:
                    ctx.errors.append(f"{w}: per-kg-DM price needs dm_basis_semantics (T2.2), one of "
                                      f"{DM_PRICE_SEMANTICS}")
                    continue
                if sem == "settled_on_measured_dm":
                    ctx.errors.append(f"{w}: dm_basis_semantics='settled_on_measured_dm' (supplier settles on the "
                                      "measured DM of the delivered batch) is a separate contract scenario; it cannot "
                                      "be merged into the default as-fed price model (T2.2)")
                    continue
                if sem not in DM_PRICE_SEMANTICS:
                    ctx.errors.append(f"{w}: unknown dm_basis_semantics {sem!r}; allowed: {DM_PRICE_SEMANTICS} "
                                      "(free text is not accepted)")
                    continue
                v2 = float(U.price_dm_to_as_fed(v, ing_dm[iid]))
                log.append(f"{iid}: {v:g} {cur}/kg DM -> {v2:g} {cur}/kg as-fed using d_hat={ing_dm[iid]:g} "
                           f"({sem})")
                v = v2
            vals[iid] = v
            pprov[iid] = sv.provenance
        missing = sorted(set(ing_dm) - set(items))
        if missing:
            ctx.errors.append(f"prices.items: missing prices for {missing}")
        price_obj = PriceScenario(price_id=str(pb["price_id"]), currency=cur, prices_per_kg_as_fed=vals,
                                  is_scenario=bool(pb["is_scenario"]), is_synthetic=bool(is_syn),
                                  price_year_or_date=pb.get("price_year_or_date"), provenance=pprov,
                                  conversion_log=tuple(log))

    # --- animal profile
    animal = None
    ab = cfg.get("animal_profile")
    if ab is not None and ctx.keys(ab, _ANI_REQ, _ANI_OPT, "animal_profile"):
        attrs: dict[str, SourcedValue] = {}
        for name, blk in (ab.get("attributes") or {}).items():
            sv = _value(ctx, blk, f"animal_profile.attributes.{name}")
            if sv is not None:
                attrs[str(name)] = sv
        animal = AnimalProfile(profile_id=str(ab["profile_id"]), species=str(ab["species"]),
                               animal_stage=str(ab["animal_stage"]), attributes=attrs,
                               nutrition_standard_and_version=ab.get("nutrition_standard_and_version"),
                               dmi_equation=ab.get("dmi_equation"), is_synthetic=bool(is_syn),
                               notes=str(ab.get("notes", "")))

    # --- run context
    rc = cfg.get("run_context") or {}
    if cfg.get("run_context") is not None:
        ctx.keys(rc, set(), _RUN_OPT, "run_context")
    fit_sets = rc.get("fit_sets") or []
    if "test" in fit_sets:
        ctx.errors.append("run_context.fit_sets contains 'test' (test set used for fitting)")
    if mode == "official":
        if not rc.get("protocol_sha256"):
            ctx.errors.append("run_context.protocol_sha256 missing (no frozen protocol hash)")
        if not rc.get("primary_assumption_id"):
            ctx.errors.append("run_context.primary_assumption_id missing (primary assumption not chosen)")

    # --- mode-level rules
    if mode in ("pilot", "official") and ctx.n_pending:
        ctx.errors.append(f"{ctx.n_pending} pending_user_decision value(s) not allowed in mode {mode!r}")
    if ds == "empirical" and ctx.n_synth:
        ctx.errors.append("config: empirical dataset contains synthetic_test_only values")
    if ctx.n_assumptions:
        ctx.warnings.append(f"{ctx.n_assumptions} research_scenario_assumption value(s): results are conditional "
                            f"on these declared assumptions")

    problem = None
    if not ctx.errors and price_obj is not None:
        try:
            problem = RationProblem(problem_id=str(cfg["problem_id"]), ingredients=tuple(ingredients),
                                    nutrients=tuple(nutrients), constraints=tuple(specs), prices=price_obj,
                                    animal=animal, is_synthetic=bool(is_syn), dataset_status=str(ds))
            _ = problem.compiled  # compile now so grammar/unit/class errors are reported here
        except InvalidProblemError as exc:
            ctx.errors.append(str(exc))
            problem = None

    rep.errors, rep.warnings = ctx.errors, ctx.warnings
    rep.n_pending, rep.n_assumptions, rep.n_synthetic_values = ctx.n_pending, ctx.n_assumptions, ctx.n_synth
    rep.n_pending_overrides = ctx.n_overrides
    if path is not None and Path(path).exists():
        rep.config_sha256 = file_sha256(path)
    else:
        rep.config_sha256 = stable_hash(cfg) if isinstance(cfg, dict) else None
    return (problem if rep.ok else None), rep


def validate_problem_config(cfg: Any, mode: str = "unit_test", path: Optional[str | Path] = None) -> ValidationReport:
    """Validate a parsed configuration; never raises for content errors (see ``report.errors``)."""
    return _parse(cfg, mode, path)[1]


def build_problem(cfg: Any, mode: str = "unit_test",
                  path: Optional[str | Path] = None) -> tuple[RationProblem, ValidationReport]:
    """Validate and build; raises :class:`ConfigValidationError` listing every issue."""
    problem, rep = _parse(cfg, mode, path)
    if not rep.ok or problem is None:
        raise ConfigValidationError(rep.errors or ["problem could not be built"])
    return problem, rep


def load_problem(path: str | Path, mode: str = "unit_test") -> tuple[RationProblem, ValidationReport]:
    """Load a YAML file, validate it for ``mode`` and build the :class:`RationProblem`."""
    cfg = load_yaml(path)
    return build_problem(cfg, mode=mode, path=path)

