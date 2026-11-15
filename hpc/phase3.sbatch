#!/bin/bash
#SBATCH --job-name=p3_116
#SBATCH --partition=gpu_short
# account: submit with sbatch -A "$PERUN_ACCOUNT"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --open-mode=append
# (path below is relative to the submit dir; create logs/ there first)
#SBATCH --output=logs/%x_%j.out
# Task 004 Phase 3 batch: one job = vLLM (Qwen3.8-27B, the vllm.sif image, read-only, flags as hpc/colour_check.sbatch
# + --max-num-seqs 16) + a task list run by N parallel sim workers in nav116.sif (hpc/phase3_batch.sh).
# Submit (positional args only, never --export); several tasksets: join them with commas.
#   cd $PROJ/code && sbatch hpc/phase3.sbatch tasksets/phase3_tuning.json tune1 8
# Output: $PROJ/runs/004/<tag>/<task_id>/ (log.jsonl, photos/, tools.json, result.json last), job.log, logs/.
# Restart-safe (hpc/RESTART.md): USR1 90 s before the wall -> stop file -> workers killed (their dirs stay partial)
# -> scontrol requeue. On every (re)start a task with result.json is skipped, a partial dir is moved aside and re-run.
# --time: 20 tasks x ~10 min / 8 workers = 3 rounds ~ 30 min, + vLLM load <= 20 min (measured 2.5-3 min), x ~2.5.
set -uo pipefail
IMAGE_DIR=${IMAGE_DIR:?set IMAGE_DIR to the folder holding g1.sif / vllm.sif}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the folder holding the Qwen3.8-27B weights}
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
CODE=${SLURM_SUBMIT_DIR:?run under sbatch}
JOB=${SLURM_JOB_ID:?run under sbatch}; RESTART=${SLURM_RESTART_COUNT:-0}
TASKSET=${1:?taskset json (relative to the code clone)}; TAG=${2:?tag}; N=${3:-8}; POSES=${4:-perceived}
OUT=$PROJ/runs/004/$TAG
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
[ -d "$CODE/.git" ] && [ -f "$CODE/sim/agent_run.py" ] && for f in ${TASKSET//,/ }; do [ -f "$CODE/$f" ] || { log "missing taskset $f"; exit 2; }; done; true || { log "bad code dir / taskset $CODE $TASKSET"; exit 2; }
log "job $JOB restart $RESTART node $(hostname) cpus $SLURM_CPUS_PER_TASK code $(cd "$CODE" && cat .git/HEAD | cut -c1-12) taskset $TASKSET tag $TAG N $N poses $POSES $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
[ "$(nvidia-smi -L | wc -l)" -eq 1 ] || { log "expected 1 GPU"; exit 1; }
[ -d "$MODEL" ] && [ -f "$VLLM_SIF" ] && [ -f "$NAV_SIF" ] || { log "missing sif/model"; exit 1; }

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

TS_ABS=$(for f in ${TASKSET//,/ }; do printf "%s," "$CODE/$f"; done); TS_ABS=${TS_ABS%,}
POSES=$POSES bash "$CODE/hpc/phase3_batch.sh" "$CODE" "$TS_ABS" "$OUT" "$N" "http://127.0.0.1:$PORT/v1" "$STOP" &
BPID=$!
while kill -0 "$BPID" 2>/dev/null; do wait "$BPID"; RC=$?; done   # wait returns early when USR1 arrives
BPID=""
log "batch rc=$RC"
if [ "$RC" = 75 ]; then
    scontrol requeue "$JOB" && log "requeued"
fi
exit 0
