"""ration_reliability: transparent ration-formulation core for the cost-reliability study.

This package is the narrowed, independently written solver/evaluator core required by
contract section 8 (``docs/contract/RuminantSim_Agent_Execution_Pack/01_MASTER_AGENT_INSTRUCTIONS_ZH.md``)
and technical protocol T1-T4, T9 (``02_TECHNICAL_PROTOCOL.md``).  It does not import or copy any
legacy RuminantSim code.

Sub-packages (status per contract section 2.3; updated 2026-09-24, W2)
---------------------------------------------------------------------
- ``normalization``  unit registry and conversions (%, g/kg, DM <-> as-fed, prices, Mcal/MJ).
- ``nutrition``      supply D(q, theta), N_j(q, theta), cost C(q), constraint compilation and
                     linearisation (T2); every constraint needs ``numerical_tolerance > 0``.
- ``uncertainty``    ``UncertaintyModel`` interface, ``DrawSet`` and named random streams.
- ``evaluation``     the single public evaluator ``evaluate`` / ``evaluate_drawset`` shared by every
                     method (T3), and ``structural_check`` (batch structural check on the same code path).
- ``optimization``   method registry ``REGISTRY`` with M0 nominal LP, M1 safety margin
                     (``margin_scale`` and ``apply_to_dm`` required), M2 joint chance-constrained SAA,
                     M2b marginal Bonferroni SAA, M3a box / M3b budget / M3c scenario-set robust LP
                     (HiGHS via SciPy); validation-stream parameter selection; ``METHOD_EQUIVALENCES``
                     and ``equivalence_annotations`` (equivalent method pairs, acceptance D06).
- ``io``             YAML problem-config loader/validator (T9; ``dm_basis_semantics`` is an enum) and
                     run records (pilot/official runs must record ``mip_rel_gap`` explicitly).
- ``information``    information-value prototype (finite candidate library, signal bins, policy MILP,
                     decision-time action sets, value net of randomisation, frozen-policy evaluation,
                     batch economics, assay strategies); ``AssaySpec`` / ``InformationPolicy``
                     containers live in ``datamodel``.
- ``reporting``      planned (empty).

Everything above except ``reporting`` is ``implemented`` and ``unit_passed`` (synthetic data);
M0 (``nominal_point``) and M2 are also ``smoke_passed``.  There is no pilot and no official run.
Interface: ``docs/ENGINE_API.md``; information value: ``docs/INFORMATION_VALUE_DESIGN.md``.

Nothing in this package is a nutritional result.  Passing tests is evidence that the defined
problem is computed and solved correctly, not evidence of any animal effect (contract section 8).
"""

from __future__ import annotations

__all__ = ["__version__", "ENGINE_API_VERSION"]

#: Package version (engine core, stage 4 part a).
__version__ = "0.1.0.dev0"

#: Version tag of the public engine API described in ``docs/ENGINE_API.md``.
ENGINE_API_VERSION = "engine-api/0.1"
