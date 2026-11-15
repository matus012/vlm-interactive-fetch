#!/bin/bash
#SBATCH --job-name=e3_116
#SBATCH --partition=gpu_short
# account: submit with sbatch -A "$PERUN_ACCOUNT"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --open-mode=append
# (path below is relative to the submit dir; create logs/ there first)
#SBATCH --output=logs/%x_%j.out
# Task 010 E3: the sealed 100 (seal-005) played by the fixed scripts agent/e3.py, no vLLM. One job:
#   1. hpc/seal_regen.py (row 21, as hpc/phase6.sbatch) from the seal-005 tree $PROJ/dev/seal005/code -> <out>/verified.jsonl.
#   2. hpc/phase6_batch.sh with "--script e3a" -> <out>/e3a, then "--script e3b" -> <out>/e3b (N workers, perceived,
#      NVIDIA EGL for the photos / locate px). The base-url argument is unused by a script run.
# Submit: cd $PROJ/code && sbatch hpc/e3.sbatch <run commit, full sha> <tag> [N]     -> $PROJ/runs/010/<tag>/
# Restart-safe as phase6.sbatch (regen must give the first start's verified sha256; finished tasks are skipped).
set -uo pipefail
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
CODE=${SLURM_SUBMIT_DIR:?run under sbatch}
JOB=${SLURM_JOB_ID:?run under sbatch}; RESTART=${SLURM_RESTART_COUNT:-0}
RUN_COMMIT=${1:?freeze-checked run commit (full sha)}; TAG=${2:?tag}; N=${3:-24}
OUT=$PROJ/runs/010/$TAG
SEAL_TREE=$PROJ/dev/seal005/code; SEALED=$PROJ/sealed/test_100.jsonl; META=$PROJ/sealed/test_100.meta.json
VERIFIED=$OUT/verified.jsonl; NAV_SIF=$PROJ/images/nav116.sif
STOP=$OUT/stop/${JOB}_${RESTART}
T0=$(date +%s)
mkdir -p "$OUT/stop" "$OUT/logs"
log() { echo "[$(date -Is)] [$(( $(date +%s) - T0 ))s] $*" | tee -a "$OUT/job.log"; }
BPID=""
trap '[ -z "$BPID" ] || kill "$BPID" 2>/dev/null' EXIT
trap 'log "USR1: stop requested"; touch "$STOP"' USR1
[ -d "$CODE/.git" ] && [ -f "$CODE/agent/e3.py" ] && [ -f "$CODE/hpc/seal_regen.py" ] || { log "bad code dir $CODE"; exit 2; }
[ -f "$SEAL_TREE/.git/HEAD" ] && [ -f "$SEALED" ] && [ -f "$META" ] || { log "missing seal-005 tree or sealed file / meta"; exit 2; }
HEAD_SHA=$(cat "$CODE/.git/HEAD")
log "job $JOB restart $RESTART node $(hostname) cpus $SLURM_CPUS_PER_TASK code $HEAD_SHA run_commit $RUN_COMMIT seal005 tree $(cat "$SEAL_TREE/.git/HEAD") tag $TAG N $N $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
[ "${#RUN_COMMIT}" = 40 ] && [ "$HEAD_SHA" = "$RUN_COMMIT" ] || { log "code HEAD $HEAD_SHA is not the freeze-checked run commit $RUN_COMMIT"; exit 2; }
[ "$(nvidia-smi -L | wc -l)" -eq 1 ] || { log "expected 1 GPU"; exit 1; }
[ -f "$NAV_SIF" ] || { log "missing nav116.sif"; exit 1; }
TR=$(date +%s)
REGEN=$(apptainer exec --nv --cleanenv \
    --env MUJOCO_GL=egl,PROJ=$PROJ,MENAGERIE_DIR=$PROJ/ext/mujoco_menagerie,PYTHONDONTWRITEBYTECODE=1,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=1 \
    -B "$PROJ" --pwd "$SEAL_TREE" "$NAV_SIF" \
    python "$CODE/hpc/seal_regen.py" --worktree "$SEAL_TREE" --sealed "$SEALED" --meta "$META" --out "$VERIFIED" \
    2> "$OUT/logs/seal_regen_${JOB}_${RESTART}.err")
RC=$?
log "seal_regen rc=$RC wall_s $(( $(date +%s) - TR )): $REGEN"
[ "$RC" = 0 ] || { log "seal-005 regeneration failed: no task runs"; exit 1; }
SHA=$(sha256sum "$VERIFIED" | cut -d' ' -f1)
[ -f "$OUT/verified.sha256" ] || echo "$SHA" > "$OUT/verified.sha256"
[ "$SHA" = "$(cat "$OUT/verified.sha256")" ] || { log "verified sha256 $SHA differs from the first start's"; exit 1; }
log "verified $SHA"
for S in e3a e3b; do
    TS=$(date +%s)
    bash "$CODE/hpc/phase6_batch.sh" "$CODE" "$VERIFIED" "$OUT/$S" "$N" "http://unused" "$STOP" "--script $S" &
    BPID=$!
    while kill -0 "$BPID" 2>/dev/null; do wait "$BPID"; RC=$?; done
    BPID=""
    log "batch $S rc=$RC wall_s $(( $(date +%s) - TS ))"
    if [ "$RC" = 75 ]; then scontrol requeue "$JOB" && log "requeued"; exit 0; fi
done
exit 0
