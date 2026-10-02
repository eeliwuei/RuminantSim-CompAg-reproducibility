"""Input preflight for dev_case drivers (review round 3, instruction F-4; R3F, 2026-09-25).

Before a driver reads anything, ``preflight_dev_case`` lists every file the driver needs, says which are missing or
stale and how each can be rebuilt, and returns an exit code; ``require_dev_case_inputs`` prints that short list and
stops the driver (``SystemExit``) instead of letting it fail later with a long traceback.  Nothing is written and no
placeholder result is produced.  The round-3 review ran ``run_dev_case_v1.py --dry-run`` on the delivered package and
got ``FileNotFoundError`` on ``build_report.json``: the package has no restricted inputs; that is a missing input, not
a solver failure, and this module now says so.

Exit codes (same convention as ``scripts/package_release.py``)::

    0  READY         every required file present and the build outputs consistent with their inputs
    1  INCONSISTENT  present, but stale or unreadable (e.g. an input changed since the build) -> rebuild first
    2  MISSING       required files missing (BLOCKED); the list says which can be rebuilt and from what
    3  USAGE         bad arguments (CLI only)

Status words: the preflight checks presence and hash consistency only; passing it is not a verification of any
number (``code_tested`` at most).  This module imports only the standard library, so it also runs where numpy /
scipy are broken.

Run type (FIX3_DEF, review round 3 red team F-2): ``run_type="pilot"`` or ``"official"`` makes the build-identity
checks strict -- no build-time sidecar ``build_identity.json``, an unreadable one, or any builder-scope file (the
builder code and every module of ``src/ration_reliability``, see ``dev_case.builder_scope_paths``) changed since the
build is INCONSISTENT (exit 1): rebuild once with ``scripts/build_dev_case.py`` on the frozen code first.  Without a
run type (smoke, debug, ``--dry-run``) these are warnings.  Changed restricted constants, changed build inputs and
build outputs that differ from the ones the sidecar recorded are INCONSISTENT in every mode.

Drivers
-------
``run_dev_case_v1.py`` calls ``require_dev_case_inputs(REPO, driver="run_dev_case_v1")`` at start-up and again in its
``check_inputs`` (used by ``replay_dev_case_eval.py``).  A later ablation driver that consumes the dev_case inputs must
do the same -- with its own extra files in ``extra`` or a profile in ``DRIVER_PROFILES`` --
``tests/unit/test_preflight.py`` fails for a driver under ``experiments/`` that reads the dev_case problem without it,
and (FIX3_DEF, red team F-4) executes every such driver in a package-shaped copy: exit 2, the short list, no
traceback, nothing written.  A driver about to write a pilot / official record should pass ``run_type=`` so that a
missing or stale build sidecar stops it before any work (``run_record.build_run_record`` refuses such a record at the
end in any case).
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO

from .dev_case import (
    DEV_CASE_V1,
    OUTPUT_FILES,
    SIDECAR,
    STRICT_BUILD_RUN_TYPES,
    DevCaseLayout,
    read_build_sidecar,
    sidecar_drift,
)

__all__ = [
    "EXIT_READY", "EXIT_INCONSISTENT", "EXIT_MISSING", "EXIT_USAGE",
    "Requirement", "DRIVER_PROFILES", "dev_case_requirements", "preflight_dev_case", "format_preflight",
    "require_dev_case_inputs",
]

EXIT_READY, EXIT_INCONSISTENT, EXIT_MISSING, EXIT_USAGE = 0, 1, 2, 3
STATUS_BY_EXIT = {EXIT_READY: "READY", EXIT_INCONSISTENT: "INCONSISTENT", EXIT_MISSING: "MISSING (BLOCKED)"}

# role -> (heading, rebuild condition); no restricted value appears here
ROLES: dict[str, tuple[str, str]] = {
    "tracked_config": ("被跟踪的公开配置",
                       "从 git 或交付包恢复（代码与公开配置在包内）"),
    "restricted_source": ("受限来源表（持有方提供）",
                          "交付包不能重建：由持有方按许可（仅本地研究使用）在授权环境提供；数值来自 NASEM 2021 本地 PDF 的"
                          "双录入与 B 层核对（P3）"),
    "restricted_library_extract": ("受限饲料库摘录（持有方提供）",
                                   "交付包不能重建：nasem_dairy commit 9b0b28e 的 NASEM_feed_library.csv 12 行摘录（代码 MIT，"
                                   "数值源自 NASEM），由持有方提供"),
    "restricted_values": ("受限数值表（持有方提供）",
                          "交付包不能重建：K4 自本地 PDF 转录的 Table 21-3 / Table 5-1 等受限值，由持有方提供"),
    "restricted_constants": ("受限构建常数（持有方提供）",
                             "交付包不能重建：持有方薄封装 build_dev_case_v1.py 中的 RESTRICTED_CONSTANTS（Table 3-1 等 "
                             "NASEM 常数；构建逻辑本身在 src/ration_reliability/build/dev_case.py，公开可审）"),
    "build_output": ("构建产物（受限）",
                     "受限输入齐全后运行 scripts/build_dev_case.py 重建（约 1 秒；只写 data/restricted_local/）"),
    "driver_restricted_input": ("驱动额外受限输入（持有方提供）",
                                "交付包不能重建：由持有方提供（见驱动文档）"),
    "driver_tracked_config": ("驱动所需公开配置", "从 git 或交付包恢复"),
}


@dataclass(frozen=True)
class Requirement:
    """One file a driver needs (repository-relative POSIX path)."""

    path: str
    role: str
    restricted: bool
    rebuild: str = ""

    def condition(self) -> str:
        return self.rebuild or ROLES.get(self.role, ("", ""))[1]


#: extra files per driver (beyond the build inputs, constants and outputs)
DRIVER_PROFILES: dict[str, tuple[Requirement, ...]] = {
    "run_dev_case_v1": (
        Requirement("configs/dev_case_v1/README.md", "driver_tracked_config", False),
        Requirement("configs/methods.yaml", "driver_tracked_config", False),
        Requirement("configs/uncertainty.yaml", "driver_tracked_config", False),
        Requirement("configs/protocol.yaml", "driver_tracked_config", False),
        Requirement("data/restricted_local/dev_case_v1/fixb_cp_sup_rdp_scaling.yaml", "driver_restricted_input", True,
                    "交付包不能重建：FIX_B 自本地 PDF 转录的 Table 21-3 值（E5-CP-SUP-RDP-SCALED 开发诊断），由持有方提供"),
    ),
}
DRIVER_PROFILES["check_inputs"] = DRIVER_PROFILES["run_dev_case_v1"]


def _sha(p: Path) -> Optional[str]:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dev_case_requirements(repo: str | Path, layout: DevCaseLayout = DEV_CASE_V1, *, driver: Optional[str] = None,
                          extra: Iterable[Requirement] = ()) -> list[Requirement]:
    """Every file the dev_case build and ``driver`` need (build inputs, constants, outputs, driver extras)."""
    root = Path(repo)
    reqs = [Requirement(f"{layout.cfg_dir}/{n}", "tracked_config", False) for n in layout.config_names]
    # any further *.yaml in the case directory is a build input too (the builder hashes every one)
    known = {r.path for r in reqs}
    for p in sorted((root / layout.cfg_dir).glob("*.yaml")):
        rel = p.relative_to(root).as_posix()
        if rel not in known:
            reqs.append(Requirement(rel, "tracked_config", False))
    reqs.append(Requirement(layout.animal_profile, "tracked_config", False))
    reqs += [Requirement(layout.core_csv, "restricted_source", True),
             Requirement(layout.blayer_csv, "restricted_source", True),
             Requirement(layout.feed_library_csv, "restricted_library_extract", True),
             Requirement(layout.restricted_values, "restricted_values", True),
             Requirement(layout.constants_file, "restricted_constants", True)]
    reqs += [Requirement(f"{layout.case_dir}/{n}", "build_output", True) for n in OUTPUT_FILES]
    profile = DRIVER_PROFILES.get(driver or "", ())
    seen = {r.path for r in reqs}
    for r in list(profile) + list(extra):
        if r.path not in seen:
            reqs.append(r)
            seen.add(r.path)
    return reqs


def _consistency(root: Path, layout: DevCaseLayout, strict: bool = False
                 ) -> tuple[list[str], list[str], dict[str, Any]]:
    """(problems, warnings, info) of the build outputs against their recorded inputs and the build sidecar;
    ``strict`` (pilot / official) turns a missing sidecar and changed builder code into problems."""
    problems: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}
    case = root / layout.case_dir
    try:
        br = json.loads((case / "build_report.json").read_text(encoding="utf-8"))
        summ = br["summary"]
        rec_inputs = dict(br["inputs"])
        rec_problem = br["problem_yaml_sha256"]
    except Exception as exc:  # noqa: BLE001 -- reported as a short line
        return [f"build_report.json 不可读或缺字段（{type(exc).__name__}）：重建"], warnings, info
    info["build_summary"] = {k: summ.get(k) for k in ("n_checks", "n_failed")}
    if summ.get("n_failed") != 0:
        problems.append(f"build_report.json 记录 {summ.get('n_failed')} 项构建检查失败：先修正输入再重建")
    for rel, h in rec_inputs.items():
        now = _sha(root / rel)
        if now is None:
            problems.append(f"构建时登记的输入现已缺失：{rel}")
        elif now != h:
            problems.append(f"输入在构建后被改动：{rel}（须重建）")
    cfg_now = sorted(p.relative_to(root).as_posix() for p in (root / layout.cfg_dir).glob("*.yaml"))
    unregistered = [c for c in cfg_now if c not in rec_inputs]
    for c in unregistered:
        problems.append(f"配置目录新增了构建报告未登记的输入：{c}（须重建）")
    if _sha(case / f"{layout.case_id}_problem.yaml") != rec_problem:
        problems.append(f"{layout.case_id}_problem.yaml 与构建报告登记的哈希不同（须重建）")
    doc, st = read_build_sidecar(root, layout)
    info["build_identity_sidecar"] = st
    rebuild = "运行 scripts/build_dev_case.py 重建一次（写出旁注；构建逻辑未变时产物逐字节不变）"
    if doc is None or st != "present":
        what = {"absent": f"无 {SIDECAR}（构建旁注）：产物的构建者没有记录（例如由旧版受限脚本生成的产物）",
                "unreadable": f"{SIDECAR} 不可读", "unknown_schema": f"{SIDECAR} 的 schema 不认识"}[st]
        if strict:
            problems.append(f"{what}；pilot/official 运行拒绝：{rebuild}")
        else:
            warnings.append(f"{what}；scripts/build_dev_case.py --check 可核对与新构建逻辑逐字节一致；"
                            "pilot/official 运行将被拒绝，须先重建写出旁注")
        return problems, warnings, info
    info["build_identity_sha256"] = doc.get("build_identity_sha256")
    drift = sidecar_drift(root, doc, layout)
    info["sidecar_drift"] = drift
    consts = [d["path"] for d in drift if d["kind"] == "constants"]
    if consts:
        problems.append(f"受限构建常数在构建后被改动：{layout.constants_file}（须重建）")
    code = [d for d in drift if d["kind"] in ("builder_code", "entry_point")]
    if code:
        listed = ", ".join(f"{d['path']}（{d['change']}）" for d in code[:8]) + (" …" if len(code) > 8 else "")
        msg = f"构建代码在构建后有改动（构建范围 = 构建代码 + src/ration_reliability 全部模块）：{listed}"
        if strict:
            problems.append(msg + f"；pilot/official 运行拒绝：{rebuild}")
        else:
            warnings.append(msg + "；运行 scripts/build_dev_case.py --check 核对产物是否仍逐字节一致；"
                                  "pilot/official 运行须先重建")
    for d in drift:
        if d["kind"] == "build_output":
            problems.append(f"构建产物与旁注登记的不同：{d['path']}（{d['change']}；须重建）")
        elif d["kind"] == "build_input":
            problems.append(f"构建输入与旁注登记的不同：{d['path']}（{d['change']}；须重建）")
    return problems, warnings, info


def preflight_dev_case(repo: str | Path, *, layout: DevCaseLayout = DEV_CASE_V1, driver: Optional[str] = None,
                       extra: Iterable[Requirement] = (), run_type: Optional[str] = None) -> dict[str, Any]:
    """Presence and consistency of every file ``driver`` needs; never raises for a missing or broken file.

    ``run_type`` in ``STRICT_BUILD_RUN_TYPES`` (pilot, official) makes the build-identity checks strict (module
    docstring)."""
    root = Path(repo)
    strict = run_type in STRICT_BUILD_RUN_TYPES
    reqs = dev_case_requirements(root, layout, driver=driver, extra=extra)
    missing = [r for r in reqs if not (root / r.path).is_file()]
    missing_build_inputs = [r for r in missing if r.role in ("tracked_config", "restricted_source",
                                                              "restricted_library_extract", "restricted_values",
                                                              "restricted_constants")]
    out: dict[str, Any] = {"schema": "ration_reliability.preflight/1", "case_id": layout.case_id, "driver": driver,
                           "run_type": run_type, "build_identity_checks": "strict" if strict else "warn",
                           "n_required": len(reqs), "missing": [], "problems": [], "warnings": [], "info": {},
                           "writes_nothing": True}
    for r in missing:
        cond = r.condition()
        if r.role == "build_output" and missing_build_inputs:
            cond = "先补齐上列缺失的构建输入，再运行 scripts/build_dev_case.py 重建"
        out["missing"].append({**asdict(r), "rebuild_condition": cond})
    if missing:
        out["exit_code"] = EXIT_MISSING
    else:
        probs, warns, info = _consistency(root, layout, strict)
        out["problems"], out["warnings"], out["info"] = probs, warns, info
        out["exit_code"] = EXIT_INCONSISTENT if probs else EXIT_READY
    out["status"] = STATUS_BY_EXIT[out["exit_code"]]
    out["rebuildable_from_package"] = not any(m["restricted"] and m["role"] != "build_output" for m in out["missing"])
    return out


def format_preflight(rep: dict[str, Any]) -> str:
    """Short human-readable text (Chinese); no values, no traceback."""
    head = f"[preflight {rep['case_id']}" + (f" | {rep['driver']}" if rep.get("driver") else "") + \
        (f" | run_type={rep['run_type']}" if rep.get("run_type") else "") + "] "
    code = rep["exit_code"]
    if code == EXIT_READY:
        lines = [head + f"READY：{rep['n_required']} 个所需文件齐全，构建产物与登记输入一致（退出码 0）。"]
        lines += [f"  注意：{w}" for w in rep.get("warnings", [])]
        return "\n".join(lines)
    if code == EXIT_INCONSISTENT:
        lines = [head + f"INCONSISTENT：文件齐全但构建产物与输入不一致（退出码 1）。未开始求解，未写任何文件。"]
        lines += [f"  - {p}" for p in rep["problems"]]
        lines.append("  处理：运行 scripts/build_dev_case.py 重建（只写 data/restricted_local/），再重跑本命令。")
        return "\n".join(lines)
    lines = [head + f"BLOCKED：缺 {len(rep['missing'])} 个所需文件（退出码 2）。未开始求解，未写任何文件，不产生占位结果。"]
    by_role: dict[str, list[dict]] = {}
    for m in rep["missing"]:
        by_role.setdefault(m["role"], []).append(m)
    for role, ms in by_role.items():
        conds = sorted({m["rebuild_condition"] for m in ms})
        lines.append(f"  {ROLES.get(role, (role, ''))[0]}：")
        lines += [f"    - {m['path']}" for m in ms]
        lines += [f"    重建条件：{c}" for c in conds]
    if not rep["rebuildable_from_package"]:
        lines.append("  结论：缺少持有方受限输入，本包不能重跑开发案例；这是缺输入，不是求解失败。"
                     "可做的是单元测试与不读受限输入的检查（证据层级 code_tested）。")
    return "\n".join(lines)


def require_dev_case_inputs(repo: str | Path, *, driver: str, layout: DevCaseLayout = DEV_CASE_V1,
                            extra: Iterable[Requirement] = (), stream: Optional[TextIO] = None,
                            quiet_when_ready: bool = True, run_type: Optional[str] = None) -> dict[str, Any]:
    """Run the preflight; when it is not READY print the short list to ``stream`` (stderr) and raise
    ``SystemExit(exit_code)``.  Returns the report when READY.  A driver about to write a pilot / official record
    passes ``run_type`` so that a missing or stale build sidecar stops it before any work (FIX3_DEF)."""
    rep = preflight_dev_case(repo, layout=layout, driver=driver, extra=extra, run_type=run_type)
    if rep["exit_code"] != EXIT_READY:
        print(format_preflight(rep), file=stream or sys.stderr)
        raise SystemExit(rep["exit_code"])
    if not quiet_when_ready:
        print(format_preflight(rep), file=stream or sys.stderr)
    return rep
