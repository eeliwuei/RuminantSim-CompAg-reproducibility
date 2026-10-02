#!/bin/bash
# Official run v2 -- preparation on the compute host (official-run plan §5; F4).  Runs NOTHING official: no reserved
# root is opened here (the last step is the measured `--official --dry-run`).  Every step is logged.
#
# Preconditions (plan §5, B5): the repository is checked out at the frozen commit (git bundle from the Mac; the commit
# that adds configs/protocol_freeze.json, the v2 pin and protocol.yaml is_frozen: true), the work tree is clean, the
# restricted inputs are in data/restricted_local/ (user-approved copy), and the Mac-built build-output bundle
# (scripts/replay_build_outputs.py reference, run on the Mac at the same commit) has been copied here.
#
# Usage (from the repository root):
#   bash scripts/px_official_prepare.sh --bundle <build_outputs_bundle.tgz> [--python <py>] [--log-root <dir>]
#
# Steps: 1 repository (HEAD, clean tree) -> 2 host facts (nproc, memory, python) -> 3 replay the Mac build bytes
# (replay_build_outputs.py apply; its own preflight and dry run) -> 4 full pytest on the replaced bytes -> 5 preflight
# (run type official) -> 6 environment check against the lock -> 7 `--official --dry-run` (must say ready:
# matches_frozen_anchored, protocol frozen, 0 draws) -> 8 the work tree is still clean.
# Exit 0 = PREPARE_OK; otherwise the first failing step's number.
set -u -o pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 90
# the freeze gate's cleanliness scope: code-manifest scope + the specification files outside it
SCOPE="src experiments scripts tests configs pytest.ini requirements-lock.txt environment.lock.json \
  docs/uncertainty_data_layers.md docs/reference_problem_v2.md reports/sd_scaling_sources.csv"
PY="../venv/bin/python"
BUNDLE=""
LOGROOT="../official_logs"
while [ $# -gt 0 ]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --log-root) LOGROOT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 91 ;;
  esac
done
if [ -z "$BUNDLE" ] || [ ! -f "$BUNDLE" ]; then
  echo "--bundle <build_outputs_bundle.tgz> is required and must exist" >&2
  exit 91
fi
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOGROOT/prepare_$TS"
mkdir -p "$LOG" || exit 92
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

note() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG/steps.log"; }
step() {   # step <n> <name> <command...>: stdout/stderr to $LOG/<name>.{out,err}; returns the command's status
  local n="$1" name="$2"; shift 2
  note "step $n $name: $*"
  "$@" > "$LOG/$name.out" 2> "$LOG/$name.err"
  local rc=$?
  note "step $n $name exit=$rc"
  return $rc
}
fail() { note "PREPARE_FAILED at step $1 ($2)"; echo "PREPARE_FAILED step=$1" > "$LOG/RESULT"; exit "$1"; }

# 1 repository
HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail 1 "not a git repository"
DIRTY="$(git status --porcelain --untracked-files=all -- $SCOPE | wc -l | tr -d ' ')"
note "HEAD=$HEAD dirty_paths=$DIRTY"
[ "$DIRTY" = "0" ] || fail 1 "work tree not clean"
[ -f configs/protocol_freeze.json ] || fail 1 "configs/protocol_freeze.json missing: not the frozen commit"

# 2 host facts (plan §5: workers = min(8, nproc - 2))
{ echo "host=$(hostname)"; echo "nproc=$(nproc 2>/dev/null || sysctl -n hw.ncpu)";
  (free -g 2>/dev/null || true); echo "python=$("$PY" -V 2>&1)"; uptime; } > "$LOG/host.txt" 2>&1
note "host facts: $(tr '\n' ' ' < "$LOG/host.txt" | cut -c1-200)"

# 3 replay the Mac build bytes (F4; B-445): the script checks HEAD = the bundle's HEAD, backs up the host's own bytes
step 3 replay "$PY" scripts/replay_build_outputs.py apply --bundle "$BUNDLE" --work-dir "$LOG/replay" \
  --superseded-dir "$LOG/superseded_build_outputs" --python "$PY" || fail 3 "replay_build_outputs.py apply"

# 4 the full test suite on the replaced bytes
step 4 pytest "$PY" -m pytest -q -p no:cacheprovider -rs || fail 4 "pytest"
note "pytest: $(grep -E '[0-9]+ (passed|failed)' "$LOG/pytest.out" | tail -1)"

# 5 preflight (official run type)
step 5 preflight "$PY" scripts/preflight_dev_case.py --driver run_dev_case_v1 --run-type official || fail 5 "preflight"

# 6 environment vs the lock
step 6 envcheck "$PY" scripts/check_environment.py
ENVSTATUS="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$LOG/envcheck.out" 2>/dev/null)"
note "environment: $ENVSTATUS"
[ "$ENVSTATUS" = "matches_lock" ] || fail 6 "environment does not match the lock"

# 7 the official dry run: freeze gate + official-mode validation, measured 0 draws / solves / files
step 7 official_dryrun "$PY" experiments/E1_cost_reliability/run_endpoint_ablation.py --official --dry-run \
  || fail 7 "--official --dry-run refused (see official_dryrun.out gate.reasons)"
"$PY" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('status=%s freeze=%s draws=%s solves=%s files=%s' % (d['status'], d.get('freeze'), d['draws'], d['solves'],
                                                          d['files_written']))
sys.exit(0 if d['status'].startswith('ready') and d['draws'] == 0 and d.get('freeze') == 'matches_frozen_anchored'
         else 1)" "$LOG/official_dryrun.out" > "$LOG/official_dryrun.summary" 2>&1 || fail 7 "dry run not ready"
note "official dry run: $(cat "$LOG/official_dryrun.summary")"

# 8 nothing changed in the tracked scope
DIRTY2="$(git status --porcelain --untracked-files=all -- $SCOPE | wc -l | tr -d ' ')"
note "dirty_after_checks=$DIRTY2"
[ "$DIRTY2" = "0" ] || fail 8 "the checks changed the work tree"

echo "PREPARE_OK head=$HEAD" > "$LOG/RESULT"
note "PREPARE_OK (log $LOG); next: screen -dmS rrs_official bash scripts/px_official_run.sh --workers <N>"
exit 0
