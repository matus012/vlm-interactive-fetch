#!/bin/bash
#SBATCH --job-name=p6_116
#SBATCH --partition=gpu_short
# account: submit with sbatch -A "$PERUN_ACCOUNT"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --open-mode=append
# (path below is relative to the submit dir; create logs/ there first)
#SBATCH --output=logs/%x_%j.out
# Task 007 Phase 6 pass: the 100 sealed tasks on the frozen system. One job:
#   1. hpc/seal_regen.py (row 21) in nav116.sif --nv, imports from the seal-005 tree $PROJ/dev/seal005/code, reads the
#      sealed file $PROJ/sealed/test_100.jsonl (put there with scp; nothing else reads it), writes
#      <out>/verified.jsonl; any guard failure or mismatch -> the job stops before vLLM.
#   2. vLLM exactly as hpc/phase3.sbatch (the vllm.sif image, Qwen3.8-27B, same flags).
#   3. hpc/phase6_batch.sh: every verified task via `sim.agent_run --verified` (row 20), N workers, code = the submit
#      dir (the run commit, $PROJ/code), perceived poses.
# Submit (positional args only, never --export):   cd $PROJ/code && sbatch hpc/phase6.sbatch <run commit, full sha> pass1 16
#   <run commit> = the commit hpc/freeze_check.py passed; the job stops before regen if $CODE/.git/HEAD differs.
# Output: $PROJ/runs/007/<tag>/ (verified.jsonl, <task_id>/..., job.log, logs/). job.log carries task ids and
# pass / fail + label only; seal_regen's stdout (counts + sha256 only) goes into job.log, its stderr to logs/.
# Restart-safe as phase3 (hpc/RESTART.md): every (re)start re-runs the regeneration, which must give the same
# verified sha256 as the first start (verified.sha256); tasks with result.json are skipped, partial dirs moved aside.
# The seal meta ($PROJ/sealed/test_100.meta.json, no tasks) ships beside the sealed file: seal_regen checks the tree's
# dependency hashes against it.
# --time: 006 counted run 2, 40 tasks / 32 cpus = 527-976 s -> 100 tasks ~ 45 min, + regen ~10 min + load ~3 min, x ~2.
set -uo pipefail
IMAGE_DIR=${IMAGE_DIR:?set IMAGE_DIR to the folder holding g1.sif / vllm.sif}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the folder holding the Qwen3.8-27B weights}
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
CODE=${SLURM_SUBMIT_DIR:?run under sbatch}
JOB=${SLURM_JOB_ID:?run under sbatch}; RESTART=${SLURM_RESTART_COUNT:-0}
RUN_COMMIT=${1:?freeze-checked run commit (full sha)}; TAG=${2:?tag}; N=${3:-16}
OUT=$PROJ/runs/007/$TAG
SEAL_TREE=$PROJ/dev/seal005/code; SEALED=$PROJ/sealed/test_100.jsonl; META=$PROJ/sealed/test_100.meta.json
VERIFIED=$OUT/verified.jsonl
VLLM_SIF=$IMAGE_DIR/vllm.sif; NAV_SIF=$PROJ/images/nav116.sif; MODEL=$MODEL_DIR
STOP=$OUT/stop/${JOB}_${RESTART}
T0=$(date +%s)
mkdir -p "$OUT/stop" "$OUT/logs" "$PROJ/cache/vllm" "$PROJ/cache/hf" "$PROJ/cache/tmp/vllm_$JOB"
log() { echo "[$(date -Is)] [$(( $(date +%s) - T0 ))s] $*" | tee -a "$OUT/job.log"; }
VPID=""; BPID=""
cleanup() {
    [ -z "$BPID" ] || kill "$BPID" 2>/dev/null
    [ -z "$VPID" ] || kill "$VPID" 2>/dev/null
    sleep 3; tail -n 200 "$OUT/logs/vllm_${JOB}_${RESTART}.log" > "$OUT/logs/vllm_tail_${JOB}_${RESTART}.log" 2>/dev/null
    [ -z "$VPID" ] || kill -9 "$VPID" 2>/dev/null
}
trap cleanup EXIT
trap 'log "USR1: stop requested"; touch "$STOP"' USR1
[ -d "$CODE/.git" ] && [ -f "$CODE/sim/agent_run.py" ] && [ -f "$CODE/hpc/seal_regen.py" ] || { log "bad code dir $CODE"; exit 2; }
[ -f "$SEAL_TREE/.git/HEAD" ] && [ -f "$SEALED" ] && [ -f "$META" ] || { log "missing seal-005 tree $SEAL_TREE or sealed file / meta in $PROJ/sealed"; exit 2; }
HEAD_SHA=$(cat "$CODE/.git/HEAD")
log "job $JOB restart $RESTART node $(hostname) cpus $SLURM_CPUS_PER_TASK code $HEAD_SHA run_commit $RUN_COMMIT seal005 tree $(cat "$SEAL_TREE/.git/HEAD") tag $TAG N $N $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
[ "${#RUN_COMMIT}" = 40 ] && [ "$HEAD_SHA" = "$RUN_COMMIT" ] || { log "code HEAD $HEAD_SHA is not the freeze-checked run commit $RUN_COMMIT"; exit 2; }
[ "$(nvidia-smi -L | wc -l)" -eq 1 ] || { log "expected 1 GPU"; exit 1; }
[ -d "$MODEL" ] && [ -f "$VLLM_SIF" ] && [ -f "$NAV_SIF" ] || { log "missing sif/model"; exit 1; }

TR=$(date +%s)
REGEN=$(apptainer exec --nv --cleanenv \
    --env MUJOCO_GL=egl,PROJ=$PROJ,MENAGERIE_DIR=$PROJ/ext/mujoco_menagerie,PYTHONDONTWRITEBYTECODE=1,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=1 \
    -B "$PROJ" --pwd "$SEAL_TREE" "$NAV_SIF" \
    python "$CODE/hpc/seal_regen.py" --worktree "$SEAL_TREE" --sealed "$SEALED" --meta "$META" --out "$VERIFIED" \
    2> "$OUT/logs/seal_regen_${JOB}_${RESTART}.err")
RC=$?
log "seal_regen rc=$RC wall_s $(( $(date +%s) - TR )): $REGEN"
[ "$RC" = 0 ] || { log "seal-005 regeneration failed: no task runs (stderr in logs/seal_regen_${JOB}_${RESTART}.err)"; exit 1; }
SHA=$(sha256sum "$VERIFIED" | cut -d' ' -f1)
[ -f "$OUT/verified.sha256" ] || echo "$SHA" > "$OUT/verified.sha256"
[ "$SHA" = "$(cat "$OUT/verified.sha256")" ] || { log "verified sha256 $SHA differs from the first start's $(cat "$OUT/verified.sha256")"; exit 1; }
log "verified $SHA"

export APPTAINERENV_VLLM_API_KEY=$(openssl rand -hex 24)
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TV=$(date +%s)
apptainer exec --nv --cleanenv -B "$PROJ" -B "$IMAGE_DIR:$IMAGE_DIR:ro" -B "$MODEL_DIR:$MODEL_DIR:ro" \
    --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HOME=$PROJ/cache/hf,VLLM_CACHE_ROOT=$PROJ/cache/vllm,TMPDIR=$PROJ/cache/tmp/vllm_$JOB,VLLM_NO_USAGE_STATS=1,DO_NOT_TRACK=1,PYTHONUNBUFFERED=1 \
    "$VLLM_SIF" vllm serve "$MODEL" \
    --served-model-name qwen3.8-27b \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.75 --max-num-seqs 16 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image": 4}' > "$OUT/logs/vllm_${JOB}_${RESTART}.log" 2>&1 &
VPID=$!
for _ in $(seq 240); do   # bounded 20 min
    kill -0 "$VPID" 2>/dev/null || { log "vllm exited before healthy"; exit 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] && break
    [ -e "$STOP" ] && break
    sleep 5
done
[ -e "$STOP" ] && { log "stop during vLLM load"; scontrol requeue "$JOB" && log "requeued"; exit 0; }
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] || { log "not healthy after 20 min"; exit 1; }
log "vllm healthy: load_time_s $(( $(date +%s) - TV )) (launch -> /health 200)"

bash "$CODE/hpc/phase6_batch.sh" "$CODE" "$VERIFIED" "$OUT" "$N" "http://127.0.0.1:$PORT/v1" "$STOP" &
BPID=$!
while kill -0 "$BPID" 2>/dev/null; do wait "$BPID"; RC=$?; done   # wait returns early when USR1 arrives
BPID=""
log "batch rc=$RC"
if [ "$RC" = 75 ]; then
    scontrol requeue "$JOB" && log "requeued"
fi
exit 0
