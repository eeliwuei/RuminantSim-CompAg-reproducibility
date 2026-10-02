"""Final metadata/hash audit only; no scientific tests, draws, or fitting."""
from collections import Counter
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "work/experiments_20261001"
OUT = ROOT / "outputs/research_extension_20261001"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


protocol_specs = [
    ("parent", "protocol.json", "预登记方案.json", "4f5da0fb98de3279c064dcc1e7c7c2261a8c9fcb5d4eb5bf0994e8c6144ba3cf"),
    ("tree", "protocol_policy_tree.json", "预登记_决策树.json", "e3c8f1a197d081ace5ab2ee9175b859e5f47492298d061f48f16f1db2e9618db"),
    ("recourse", "protocol_recourse.json", "预登记_条件配方.json", "ffbb24c0df4da953928d891107aa1ac6e2297df06fc897d1bcaea70dcb08ae06"),
    ("source_copula", "protocol_source_copula.json", "预登记_来源相关性.json", "6dacf2c1dfdc641c4f0c440fd9dca123c97c660f2acfc9145d19c38ce682e77a"),
    ("energy_certificate", "protocol_energy_certificate.json", "预登记_能量下界.json", "c758229fba4f0b49719ae431c4f292fba52cb6243a93e27e06a52857e2d469f4"),
]
protocols, protocol_rows = {}, []
for family, name, public, expected in protocol_specs:
    p, q = BASE / name, OUT / public
    assert sha(p) == expected
    x, published = read(p), read(q)
    assert published.get("private_protocol_sha256", published.get("protocol_sha256")) == expected
    pins = dict(x.get("code_sha256_before_formal", {}))
    pin_schema = "code_sha256_before_formal"
    if family == "source_copula":
        pin_schema = "source_copula.code_sha256 + source_copula.orchestration_code_sha256"
        pins = {"source_copula/" + k: h for k, h in x["source_copula"]["code_sha256"].items()}
        pins.update(x["source_copula"]["orchestration_code_sha256"])
        assert published["public_protocol"]["source_copula"]["code_sha256"] == x["source_copula"]["code_sha256"]
        for path, digest in published["required_file_sha256"].items():
            assert sha(BASE / path) == digest, path
    else:
        assert published["code_sha256_before_formal"] == pins
    assert pins
    for path, digest in pins.items():
        assert sha(BASE / path) == digest, path
    protocols[family] = x
    protocol_rows.append({"family": family, "protocol_path": str(p.relative_to(ROOT)),
        "protocol_sha256": expected, "public_preregistration_path": str(q.relative_to(ROOT)),
        "public_preregistration_sha256": sha(q), "code_pin_schema": pin_schema,
        "code_files_verified": len(pins), "all_code_hashes_match": True})

matrix_path = OUT / "完整计算_独立完成矩阵.json"
matrix = read(matrix_path)
assert matrix["status"] == "all_registered_computation_and_analysis_complete"
metadata_changes = []
for name, record in matrix["analysis_checks"].items():
    path = ROOT / record["path"]
    digest = sha(path)
    if digest != record["sha256"]:
        assert name == "price_DMI", "Unapproved stale matrix entry: " + name
        assert read(path)["status"] == "complete_verified"
        metadata_changes.append({"field": "analysis_checks.price_DMI.sha256", "old": record["sha256"], "new": digest})
        record["sha256"] = digest
    value = read(path)
    assert record.get("status") == value.get("status")
    assert record.get("complete") == value.get("complete")
if metadata_changes:
    matrix_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n")

p = protocols["parent"]
expected_parent = set()
for case, r in itertools.product(p["cases"], range(3)):
    expected_parent |= {f"{module}__{case}__r{r}" for module in ("information", "correlation")}
    for world in p["worlds"]:
        expected_parent.add(f"price_reference__{case}__{world}__r{r}")
        expected_parent |= {f"certificate__{case}__{world}__r{r}__b{b}" for b in range(p["certificate"]["batches_per_root"])}
for case, world, r in itertools.product(p["dmi_frontier"]["cases"], p["dmi_frontier"]["worlds"], range(3)):
    expected_parent.add(f"dmi_frontier__{case}__{world}__r{r}")
expected_sets = {"parent": expected_parent}
for family in ("tree", "recourse"):
    p = protocols[family]
    expected_sets[family] = {f"{case}__{world}__r{r}" for case, world, r in itertools.product(p["cases"], p["worlds"], range(3))}
p = protocols["source_copula"]
expected_sets["source_copula"] = {f"{case}__r{r}" for case, r in itertools.product(p["cases"], range(3))}
p = protocols["energy_certificate"]
expected_sets["energy_certificate"] = {f"{case}__{world}__r{r}__b{b}" for case, world, r, b in
    itertools.product(p["cases"], p["worlds"], range(3), range(p["certificate"]["batches_per_root"]))}
dispatch_rows = []
assert {r["family"] for r in matrix["families"]} == set(expected_sets)
for item in matrix["families"]:
    path = ROOT / item["dispatch_done_path"]
    assert sha(path) == item["sha256"]
    value = read(path)
    records = value["results"]
    names = [r.get("task", r.get("name")) for r in records]
    assert len(set(names)) == len(names) and set(names) == expected_sets[item["family"]]
    assert len(records) == item["registered_tasks"] == item["records_found"]
    assert all(r["exit_code"] == 0 for r in records)
    assert item["nonzero_exit_records"] == 0 and value.get("failures", 0) == 0
    dispatch_rows.append({"family": item["family"], "dispatch_done_path": item["dispatch_done_path"],
        "sha256": sha(path), "records": len(records), "all_identities_unique_and_registered": True,
        "nonzero_exit_records": 0})
assert sum(r["records"] for r in dispatch_rows) == matrix["formal_computational_tasks"] == 588

receipt_specs = [("math", "数学证书_公开产物回执.json", "files", 7),
    ("parent_certificate", "parent_certificate/独立完成回执.json", "files", 4),
    ("source_copula", "source_copula/完成验证凭据.json", "result_file_sha256", 9),
    ("price_dmi", "price_dmi/completion_receipt.json", "all_file_sha256", 15)]
receipt_rows = []
for family, name, key, expected_n in receipt_specs:
    path = OUT / name
    value = read(path)
    assert value.get("complete", False) or value.get("status") in ("complete_verified", "complete_verified_and_packaged")
    assert len(value[key]) == expected_n
    for filename, digest in value[key].items():
        assert sha(path.parent / filename) == digest, name + "/" + filename
    receipt_rows.append({"family": family, "receipt_path": str(path.relative_to(ROOT)),
        "receipt_sha256": sha(path), "child_files_verified": expected_n, "all_child_hashes_match": True})
source_receipt = read(OUT / "source_copula/完成验证凭据.json")
assert source_receipt["pre_draw_receipt_sha256"] == sha(OUT / "预登记_来源相关性.json")
assert source_receipt["dispatch_done_sha256"] == next(r["sha256"] for r in dispatch_rows if r["family"] == "source_copula")
price_receipt = read(OUT / "price_dmi/completion_receipt.json")
assert price_receipt["analysis_summary_sha256"] == matrix["analysis_checks"]["price_DMI"]["sha256"]
assert price_receipt["analysis_code_sha256"] == sha(BASE / "analysis/price_dmi_merge.py")
assert price_receipt["verified_cells"] == 72 and price_receipt["verified_choice_slots"] == 1920
analysis_rows = [{"name": k, "path": v["path"], "sha256": sha(ROOT / v["path"]), "matches_matrix": True}
                 for k, v in matrix["analysis_checks"].items()]
policy = read(ROOT / matrix["analysis_checks"]["policy"]["path"])
pca = read(ROOT / matrix["analysis_checks"]["PCA_oracle"]["path"])
pc = read(ROOT / matrix["analysis_checks"]["parent_correlation"]["path"])
sc = read(ROOT / matrix["analysis_checks"]["source_copula"]["path"])
price = read(ROOT / matrix["analysis_checks"]["price_DMI"]["path"])
counts = Counter(r["family"] for r in price["cells"])
assert counts == {"price_reference": 60, "dmi_frontier": 12}
base_cells = len(read(OUT / "parent_certificate/基础非能量下界_结果.json")["rows"])
energy_cells = len(read(OUT / "能量全空间下界_结果.json")["rows"])
policy_counts = Counter(r["family"] for r in policy["verification_records"])
assert policy_counts == {"tree": 60, "recourse": 60}
assert sum(policy_counts.values()) == policy["verified_cells"] == 120
units = {"information": pca["cells"], "parent_correlation": pc["n_completed_jobs"],
    "price": counts["price_reference"], "DMI": counts["dmi_frontier"],
    "base_certificate": base_cells, "tree": policy_counts["tree"],
    "recourse": policy_counts["recourse"], "source_copula": sc["n_completed_jobs"],
    "energy_certificate": energy_cells}
assert all(isinstance(v, int) and not isinstance(v, bool) for v in units.values())
terms = list(units.values())
assert terms == [60, 60, 60, 12, 20, 60, 60, 60, 20]
assert units == matrix["merged_units_calculation"]
assert sum(terms) == sum(matrix["merged_units_calculation"].values()) == matrix["merged_units"] == 412
assert matrix["summary_count_correction"]["previous_displayed_total"] == 472
assert matrix["summary_count_correction"]["correct_total"] == 412
target = OUT / "最终身份核对.json"
assert target.exists(), "This invocation corrects the specifically authorized old receipt"
old_hash = sha(target)
assert old_hash == "0f4e6ce8a4c37654467f9dcb29a499532142b3a2ece1d6f363fe49387230d755"
old = read(target)
assert old["merged_units"] == 472
assert old["protocols"] == protocol_rows and old["dispatch"] == dispatch_rows
assert old["public_receipts"] == receipt_rows and old["analysis_summary_hashes"] == analysis_rows
review_time = dt.datetime.now(dt.timezone.utc).isoformat()

result = {"schema": "RRS.final_identity_only_audit/1", "complete": True,
    "checked_utc": review_time,
    "scope": "Read-only protocol/code identity, dispatcher completion and existing public receipt hashes; no rerun of scientific tests or inference.",
    "formal_protocols_verified": 5, "protocols": protocol_rows,
    "formal_dispatch_records_verified": 588, "dispatch": dispatch_rows,
    "nonzero_exits": 0, "unexpected_or_duplicate_registered_identities": 0,
    "public_subreceipts_verified": len(receipt_rows), "public_child_file_hashes_verified": 35,
    "public_receipts": receipt_rows, "analysis_summary_hashes": analysis_rows,
    "completion_matrix_path": str(matrix_path.relative_to(ROOT)), "completion_matrix_sha256": sha(matrix_path),
    "completion_matrix_metadata_changes": metadata_changes, "merged_unit_breakdown": units,
    "merged_units": 412, "matrix_consistent": True,
    "explicit_integer_sum_verification": {"terms": terms, "expression": "60+60+60+12+20+60+60+60+20",
        "sum": sum(terms), "all_terms_independently_matched_to_existing_summary_counts": True,
        "explicit_matrix_integer_map_matches": True,
        "DMI_unit_definition": "One merged case/world/root frontier. Its six internal fraction grid points are not six independent merged units."},
    "supersession": {"previous_receipt_sha256": old_hash, "previous_auditor_sha256": old["auditor_sha256"],
        "previous_merged_units": 472, "corrected_merged_units": 412,
        "previous_completion_matrix_sha256": old["completion_matrix_sha256"],
        "corrected_completion_matrix_sha256": sha(matrix_path), "reviewed_utc": review_time,
        "reason": "Previous audit incorrectly expanded 12 DMI frontiers into 72 internal fraction-grid units, adding 60. Independent addition of the nine merged-family integers gives 412. Scientific evidence, five protocols, 588 dispatcher identities and 35 public child hashes are unchanged and rechecked.",
        "protocol_dispatch_receipt_and_analysis_hash_records_unchanged_verified": True},
    "scientific_tests_rerun": False, "new_draws_generated": False,
    "frozen_protocol_source_or_runner_modified": False,
    "archive_validation_scope": "This receipt precedes the full archive; ZIP member validation is a separate subsequent gate.",
    "auditor_sha256": sha(Path(__file__))}
target.chmod(0o644)
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
target.chmod(0o444)
print(json.dumps({"complete": True, "protocols": 5, "dispatch_records": 588,
    "public_child_hashes": 35, "merged_units": 412, "matrix_metadata_changes": len(metadata_changes),
    "output_sha256": sha(target), "matrix_sha256": sha(matrix_path)}, ensure_ascii=False))
