"""Call-site and wording scan of the information-value readings (round-3 red team B-2; FIX3_BC).

Why: the R3B scan (``reports/value_callsite_scan.json``) matched old *names* only, so a line that states the
withdrawn reading without an old name -- ``CURRENT_STATE.json`` line 381, "可执行依据 0.0214 / 0.0269 USD/头/日 ...
随机化通道份额 11.7 % / 16.6 %" -- was missed (0 hits for CURRENT_STATE).  This scanner adds **semantic phrases**
(可执行依据, 份额 / share, 支付上限 / payment bound, 愿付 / willingness to pay, 检测优先级 / assay priority,
information-supported, 纯信息收益 ...) next to the old and new names, classifies every file (authoritative state page,
decision log, historical record, code, tests, docs, reports, review input) and marks each hit's context as
``negated_or_withdrawn`` (the line negates or withdraws the reading) or ``affirmative`` (a human must check it).  The
context mark is a heuristic of the instrument, not a verdict.

The tests check the instrument (every pattern is found; the CURRENT_STATE line of the finding is found and marked
affirmative; negations are recognised) and that the committed report was produced with the current pattern list and
scanned the authoritative files.  They do not require any file to be free of hits: rewriting the authoritative pages
belongs to their owners.

Script mode (holder, working tree): ``python tests/unit/test_value_callsite_scan.py --write`` rewrites
``reports/value_callsite_scan.json``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "reports" / "value_callsite_scan.json"
SCHEMA = "ration_reliability.value_callsite_scan/2"

#: old names (R3B renames) -- a hit means the old reading may be present
OLD_NAMES = ("executable_information_supported_value", "EXECUTABLE_INFORMATION_SUPPORTED_VALUE",
             "randomization_share_of_operational", "operational_information_supported",
             "require_information_supported_operational_value", "executable_decision_basis")
#: new names -- listed so that their use can be audited too
NEW_NAMES = ("heuristic_min_of_two_policy_values", "HEURISTIC_MIN_OF_TWO_POLICY_VALUES",
             "matched_uninformative_contrast_ratio", "operational_not_above_randomized_reference",
             "require_operational_not_above_randomized_reference", "diagnostic_without_decision_problem",
             "CompleteDecisionProblem")
#: semantic phrases of the withdrawn readings (B-2: names alone miss them); case-insensitive for the English ones
SEMANTIC_PHRASES = ("可执行依据", "可执行结论", "通道份额", "随机化份额", "来源份额", "支付上限", "愿付", "盈亏平衡",
                    "检测优先级", "纯信息收益", "信息支撑", "去除随机化",
                    "willingness to pay", "willingness-to-pay", "payment bound", "break-even", "assay priorit",
                    "randomisation share", "randomization share", "share of the operational", "share of operational",
                    "information-supported", "information supported", "executable basis", "money basis")
_NEG = re.compile(r"(不是|不能|不得|不再|不作|不进|不换算|不称|不据此|禁止|撤回|删除|取消|拒绝|~~|withdrawn|refus|never|"
                  r"\bnot\b|\bno\b|deprecated|弃用|heuristic|启发式|excluded from paper|不代表|并不)", re.IGNORECASE)
#: phrases that state the withdrawn reading itself: a general negation elsewhere on the line ("不能直接换算成愿付检测费")
#: does not clear them -- only an explicit withdrawal / re-label marker does (the correction_report.md line 223 case)
STRONG_PHRASES = ("可执行依据", "通道份额", "随机化份额", "来源份额", "纯信息收益", "information-supported",
                  "information supported", "executable basis", "executable_information_supported_value",
                  "executable_decision_basis", "randomization_share_of_operational")
_WITHDRAWN = re.compile(r"(~~|撤回|删除|withdrawn|deprecated|弃用|R3B|FIX3_BC|heuristic|启发式|原名|旧名|formerly|alias|"
                        r"更名|改名|renamed)", re.IGNORECASE)
_TEXT_SUFFIXES = (".py", ".md", ".json", ".yaml", ".yml", ".csv", ".txt", ".toml", ".ini", ".cfg")
_SKIP_PREFIXES = ("data/restricted_local/", "release_staging/", "legacy/")

FILE_CLASSES = (
    ("authoritative_state", ("CURRENT_STATE.json",)),
    ("decision_or_action_log", ("DECISIONS.md", "BLOCKERS.md", "USER_ACTIONS_", "correction_report.md", "CHANGELOG.md",
                                "AGENT_PROGRESS.md", "AGENT_HANDOFF.md", "交付说明")),
    ("historical_record_readonly", ("results/", "logs/", "audit/_parts/round1", "audit/_parts/round2", "audit/")),
    ("review_input_readonly", ("docs/review_",)),
    ("code", ("src/", "experiments/", "scripts/")),
    ("tests", ("tests/",)),
    ("docs", ("docs/", "README", "manuscript/")),
    ("reports", ("reports/",)),
    ("configs", ("configs/",)),
)

#: files written by FIX3_BC (its allocation); hits there were reviewed by hand in this round
FIX3_BC_ALLOCATION = ("src/ration_reliability/information/", "docs/value_definition.md",
                      "docs/value_semantics_decision.md", "src/ration_reliability/evaluation/reference.py",
                      "experiments/E1_cost_reliability/", "configs/dev_case_v1/reference_constraints.csv",
                      "docs/reference_problem_v1.md", "tests/unit/test_fix_a_value_and_guard.py",
                      "tests/unit/test_garbling_counterexample.py",
                      "tests/unit/test_money_needs_a_complete_decision_problem.py",
                      "tests/unit/test_value_callsite_scan.py", "reports/value_callsite_scan.json",
                      "reports/garbling_counterexample.json", "reports/value_definition_counterexample_results.json")


def file_class(rel: str) -> str:
    for cls, prefixes in FILE_CLASSES:
        if any(rel == p or rel.startswith(p) for p in prefixes):
            return cls
    return "other"


def _patterns() -> list[tuple[str, str, re.Pattern]]:
    out = []
    for n in OLD_NAMES:
        out.append(("old_name", n, re.compile(re.escape(n))))
    for n in NEW_NAMES:
        out.append(("new_name", n, re.compile(re.escape(n))))
    for n in SEMANTIC_PHRASES:
        out.append(("semantic_phrase", n, re.compile(re.escape(n), re.IGNORECASE)))
    return out


PATTERNS = _patterns()


def scan_text(rel: str, text: str) -> list[dict]:
    """Hits of every pattern in ``text`` (one record per line with at least one hit)."""
    hits = []
    for ln, line in enumerate(text.splitlines(), start=1):
        found = [(kind, name) for kind, name, rx in PATTERNS if rx.search(line)]
        if not found:
            continue
        kinds = sorted({k for k, _ in found})
        strong = any(n.lower() in line.lower() for n in STRONG_PHRASES)
        cleared = bool(_WITHDRAWN.search(line)) if strong else bool(_NEG.search(line))
        hits.append({"file": rel, "line": ln, "file_class": file_class(rel), "kinds": kinds,
                     "old_names": [n for k, n in found if k == "old_name"],
                     "new_names": [n for k, n in found if k == "new_name"],
                     "semantic_phrases": [n for k, n in found if k == "semantic_phrase"],
                     "strong_phrase": strong,
                     "context": "negated_or_withdrawn" if cleared else "affirmative",
                     "in_fix3_bc_allocation": any(rel == p or rel.startswith(p) for p in FIX3_BC_ALLOCATION),
                     "text": line.strip()[:400]})
    return hits


def scanned_files(repo: Path = REPO) -> list[str]:
    r = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                       capture_output=True, timeout=60, check=True)
    rels = sorted({x for x in r.stdout.decode("utf-8", "surrogateescape").split("\0") if x})
    return [x for x in rels if x.endswith(_TEXT_SUFFIXES) and not x.startswith(_SKIP_PREFIXES)
            and (repo / x).is_file() and x != "reports/value_callsite_scan.json"]


def build_report(repo: Path = REPO) -> dict:
    files = scanned_files(repo)
    hits: list[dict] = []
    for rel in files:
        try:
            text = (repo / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits += scan_text(rel, text)
    by_file: dict[str, dict] = {}
    for h in hits:
        b = by_file.setdefault(h["file"], {"file_class": h["file_class"], "n_hits": 0, "n_affirmative": 0,
                                            "n_old_name": 0, "n_semantic": 0,
                                            "in_fix3_bc_allocation": h["in_fix3_bc_allocation"]})
        b["n_hits"] += 1
        b["n_affirmative"] += h["context"] == "affirmative"
        b["n_old_name"] += "old_name" in h["kinds"]
        b["n_semantic"] += "semantic_phrase" in h["kinds"]
    owners = sorted(f for f, b in by_file.items()
                    if b["file_class"] in ("authoritative_state", "decision_or_action_log") and b["n_affirmative"])
    return {
        "schema": SCHEMA,
        "title": ("information-value call-site and wording scan (FIX3_BC, round-3 red team B-2): old names, new names and "
                  "semantic phrases of the withdrawn readings"),
        "evidence_level": "code_tested",
        "method": ("line scan of every git-tracked or untracked-but-not-ignored text file of the working tree "
                   "(git ls-files --cached --others --exclude-standard); data/restricted_local, release_staging and "
                   "legacy are not scanned; the context mark (negated_or_withdrawn / affirmative) is a keyword "
                   "heuristic of the instrument, not a verdict -- every affirmative hit in an authoritative or decision "
                   "file needs its owner's reading"),
        "generator": "tests/unit/test_value_callsite_scan.py --write",
        "patterns": {"old_names": list(OLD_NAMES), "new_names": list(NEW_NAMES),
                     "semantic_phrases": list(SEMANTIC_PHRASES)},
        "negation_regex": _NEG.pattern,
        "n_files_scanned": len(files),
        "files_scanned_authoritative": [f for f in files if file_class(f) in ("authoritative_state",
                                                                              "decision_or_action_log")],
        "authoritative_or_decision_files_with_affirmative_hits": owners,
        "owner_action": ("rewrite per docs/value_semantics_decision.md (money only from a complete decision problem; "
                         "the minimum is a heuristic; B/Δ_op is a diagnostic ratio; nothing is an assay priority); "
                         "proposed wording in audit/_parts/round3/FIX3_BC_record.md; these files are outside the "
                         "FIX3_BC allocation and were not modified"),
        "summary_by_file": dict(sorted(by_file.items())),
        "hits": hits,
    }


# =================================================================================================
# tests (the instrument and the committed report)
# =================================================================================================

FINDING_LINE = ('"S2 信息价值 α = 0.10 / 0.05：可执行依据 0.0214 / 0.0269 USD/头/日，operational 0.0242 / 0.0310，'
                '随机化通道份额 11.7 % / 16.6 %，unidentified_scenario"')


def test_the_scanner_finds_the_line_the_name_scan_missed():
    hits = scan_text("CURRENT_STATE.json", "{\n" + FINDING_LINE + "\n}")
    assert len(hits) == 1 and hits[0]["line"] == 2
    h = hits[0]
    assert h["file_class"] == "authoritative_state" and h["old_names"] == []          # no old name on that line
    assert {"可执行依据", "通道份额"} <= set(h["semantic_phrases"]) and h["context"] == "affirmative"
    # correction_report.md line 223: the withdrawn reading next to a general negation is still affirmative
    line223 = ("**可执行依据 `executable_information_supported_value` 0.0214 / 0.0269**；随机化通道份额 11.7 % / 16.6 %"
               "（`partly_randomization_channel`，所以 operational 值不能直接换算成愿付检测费）")
    assert scan_text("correction_report.md", line223)[0]["context"] == "affirmative"
    assert scan_text("docs/x.md", "~~可执行依据~~（R3B 撤回）")[0]["context"] == "negated_or_withdrawn"


def test_every_pattern_is_detected_and_negations_are_marked():
    for kind, name, _ in PATTERNS:
        h = scan_text("docs/x.md", f"prefix {name} suffix")
        assert h and name in (h[0]["old_names"] + h[0]["new_names"] + h[0]["semantic_phrases"]), name
    assert scan_text("docs/x.md", "the minimum is not a payment bound")[0]["context"] == "negated_or_withdrawn"
    assert scan_text("docs/x.md", "~~可执行依据~~（删除）")[0]["context"] == "negated_or_withdrawn"
    assert scan_text("docs/x.md", "可执行依据 is not changed")[0]["context"] == "affirmative"      # strong phrase
    assert scan_text("docs/x.md", "Willingness To Pay of the assay")[0]["context"] == "affirmative"
    assert scan_text("docs/x.md", "nothing relevant here") == []
    assert file_class("DECISIONS.md") == "decision_or_action_log" and file_class("results/pilot/x.json") == \
        "historical_record_readonly" and file_class("src/a.py") == "code"


def test_committed_report_uses_the_current_patterns_and_scanned_the_authoritative_files():
    rec = json.loads(REPORT.read_text(encoding="utf-8"))
    assert rec["schema"] == SCHEMA and rec["evidence_level"] == "code_tested"
    assert rec["patterns"] == {"old_names": list(OLD_NAMES), "new_names": list(NEW_NAMES),
                               "semantic_phrases": list(SEMANTIC_PHRASES)}
    for f in ("CURRENT_STATE.json", "DECISIONS.md", "correction_report.md"):
        assert f in rec["files_scanned_authoritative"], f
    assert all(set(h) >= {"file", "line", "context", "file_class"} for h in rec["hits"])


if __name__ == "__main__":
    import datetime

    rep = build_report()
    rep["generated_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = json.dumps(rep, ensure_ascii=False, indent=1) + "\n"
    if "--write" in sys.argv:
        REPORT.write_text(text, encoding="utf-8")
        print(f"wrote {REPORT} ({len(rep['hits'])} hits in {len(rep['summary_by_file'])} files; authoritative with "
              f"affirmative hits: {rep['authoritative_or_decision_files_with_affirmative_hits']})")
    else:
        sys.stdout.write(text)
