#!/bin/bash
# Official run v2 on the compute host (official-run plan §5; B6).  Screen-friendly: no terminal input, everything goes
# to a log directory; the markers OFFICIAL_STARTED / OFFICIAL_EXIT / OFFICIAL_ENDED are written there.
#
# Run it only after scripts/px_official_prepare.sh reported PREPARE_OK at the same commit, e.g.
#   screen -dmS rrs_official bash scripts/px_official_run.sh --workers 8
#   screen -dmS rrs_official bash scripts/px_official_run.sh --workers 8 --root-k 0 1 2
#
# What it does: checks that the work tree is clean and the freeze record present (else exit 2 before anything runs),
# exports one BLAS / OpenMP thread per process, and runs under `nice -n 10`
#   run_endpoint_ablation.py --official --authorised-compute --authorisation-file <file> --root-k <k...> --workers <N>
# which re-checks every gate (authorisation scope run_official_v2 + this host, root indices, the freeze gate, preflight,
# environment lock) BEFORE any reserved root is opened, runs one process per job, then merges the jobs per root and
# writes the run record (exit 0 = official; 5 = official_invalidated, outputs kept and disclosed).  If the merge alone
# failed (exit 2 after the jobs), it can be repeated with:  <python> run_endpoint_ablation.py --official-merge <RUN_ID>
# Do not change any file of the repository between the freeze commit and the end of the run (plan §7).
set -u -o pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 90
# the freeze gate's cleanliness scope: code-manifest scope + the specification files outside it
SCOPE="src experiments scripts tests configs pytest.ini requirements-lock.txt environment.lock.json \
  docs/uncertainty_data_layers.md docs/reference_problem_v2.md reports/sd_scaling_sources.csv"
PY="../venv/bin/python"
WORKERS=""
ROOTS="0 1 2"
AUTH="data/restricted_local/compute_authorisation_official.json"
LOGROOT="../official_logs"
while [ $# -gt 0 ]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --root-k) shift; ROOTS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do ROOTS="$ROOTS $1"; shift; done ;;
    --python) PY="$2"; shift 2 ;;
    --authorisation-file) AUTH="$2"; shift 2 ;;
    --log-root) LOGROOT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 91 ;;
  esac
done
[ -n "$WORKERS" ] || { echo "--workers N is required (plan §5: min(8, nproc - 2); D-539: 8)" >&2; exit 91; }
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOGROOT/run_$TS"
mkdir -p "$LOG" || exit 92

HEAD="$(git rev-parse HEAD 2>/dev/null)"
DIRTY="$(git status --porcelain --untracked-files=all -- $SCOPE | wc -l | tr -d ' ')"
{ echo "head=$HEAD"; echo "dirty_paths=$DIRTY"; echo "roots=$ROOTS"; echo "workers=$WORKERS"; echo "host=$(hostname)";
  echo "nproc=$(nproc 2>/dev/null || sysctl -n hw.ncpu)"; } > "$LOG/run_context.txt"
if [ "$DIRTY" != "0" ] || [ ! -f configs/protocol_freeze.json ]; then
  echo "refused: work tree not clean or configs/protocol_freeze.json missing (nothing ran)" | tee "$LOG/REFUSED"
  exit 2
fi

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOG/OFFICIAL_STARTED"
# shellcheck disable=SC2086  # ROOTS is a list of root indices
nice -n 10 "$PY" experiments/E1_cost_reliability/run_endpoint_ablation.py --official --authorised-compute \
  --authorisation-file "$AUTH" --root-k $ROOTS --workers "$WORKERS" > "$LOG/official_run.log" 2>&1
RC=$?
echo "$RC" > "$LOG/OFFICIAL_EXIT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOG/OFFICIAL_ENDED"
grep -o '"run_id": "[^"]*"' "$LOG/official_run.log" | head -1 > "$LOG/RUN_ID" || true
exit "$RC"
