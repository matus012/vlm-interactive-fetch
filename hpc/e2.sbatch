#!/bin/bash
#SBATCH --job-name=e2_116
#SBATCH --partition=gpu_short
# account: submit with sbatch -A "$PERUN_ACCOUNT"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --open-mode=append
# (path below is relative to the submit dir; create logs/ there first)
#SBATCH --output=logs/%x_%j.out
# Task 011 E2 counted pass: house B (NAV116_HOUSE=B in every container), the frozen agent, seal-011. One job:
#   1. hpc/seal_regen.py from the seal-011 tree $PROJ/dev/seal011/code (house B, easings on) -> <out>/verified.jsonl;
#      any guard failure or mismatch -> stop before vLLM.
#   2. vLLM exactly as hpc/phase3.sbatch; hpc/phase6_batch.sh (no extra flag) -> <out>/<task_id>/ = the counted pass.
#   3. Only after step 2 has written every result (task 011 step 5): agent/e3.py byte-unchanged, E3a then E3b, on the
#      same verified tasks -> <out>/e3a/, <out>/e3b/ (vLLM stopped first).
# Submit: cd $PROJ/code && sbatch hpc/e2.sbatch <run commit> <tag> <N> <seal-011 commit> <E2 seal sha256>
# Output: $PROJ/runs/011/<tag>/. Restart-safe as phase6.sbatch.
set -uo pipefail
IMAGE_DIR=${IMAGE_DIR:?set IMAGE_DIR to the folder holding g1.sif / vllm.sif}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the folder holding the Qwen3.8-27B weights}
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
CODE=${SLURM_SUBMIT_DIR:?run under sbatch}
JOB=${SLURM_JOB_ID:?run under sbatch}; RESTART=${SLURM_RESTART_COUNT:-0}
export NAV116_HOUSE=B   # task 011: house B in every container (phase6_batch.sh passes it on)
RUN_COMMIT=${1:?freeze-checked run commit (full sha)}; TAG=${2:?tag}; N=${3:-16}
SEAL_COMMIT=${4:?seal-011 commit (full sha)}; SEAL_SHA=${5:?E2 seal sha256}
OUT=$PROJ/runs/011/$TAG
SEAL_TREE=$PROJ/dev/seal011/code; SEALED=$PROJ/sealed/e2_100.jsonl; META=$PROJ/sealed/e2_100.meta.json
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
[ -f "$SEAL_TREE/.git/HEAD" ] && [ -f "$SEALED" ] && [ -f "$META" ] || { log "missing seal-011 tree $SEAL_TREE or sealed file / meta in $PROJ/sealed"; exit 2; }
HEAD_SHA=$(cat "$CODE/.git/HEAD")
log "job $JOB restart $RESTART node $(hostname) cpus $SLURM_CPUS_PER_TASK code $HEAD_SHA run_commit $RUN_COMMIT seal011 tree $(cat "$SEAL_TREE/.git/HEAD") seal_sha ${SEAL_SHA:0:16} tag $TAG N $N $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
[ "${#RUN_COMMIT}" = 40 ] && [ "$HEAD_SHA" = "$RUN_COMMIT" ] || { log "code HEAD $HEAD_SHA is not the freeze-checked run commit $RUN_COMMIT"; exit 2; }
[ "$(nvidia-smi -L | wc -l)" -eq 1 ] || { log "expected 1 GPU"; exit 1; }
[ -d "$MODEL" ] && [ -f "$VLLM_SIF" ] && [ -f "$NAV_SIF" ] || { log "missing sif/model"; exit 1; }

TR=$(date +%s)
REGEN=$(apptainer exec --nv --cleanenv \
    --env MUJOCO_GL=egl,PROJ=$PROJ,MENAGERIE_DIR=$PROJ/ext/mujoco_menagerie,PYTHONDONTWRITEBYTECODE=1,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=1,NAV116_HOUSE=B \
    -B "$PROJ" --pwd "$SEAL_TREE" "$NAV_SIF" \
    python "$CODE/hpc/seal_regen.py" --worktree "$SEAL_TREE" --sealed "$SEALED" --meta "$META" --out "$VERIFIED" \
    --seal-commit "$SEAL_COMMIT" --sha256 "$SEAL_SHA" \
    2> "$OUT/logs/seal_regen_${JOB}_${RESTART}.err")
RC=$?
log "seal_regen rc=$RC wall_s $(( $(date +%s) - TR )): $REGEN"
[ "$RC" = 0 ] || { log "seal-011 regeneration failed: no task runs (stderr in logs/seal_regen_${JOB}_${RESTART}.err)"; exit 1; }
SHA=$(sha256sum "$VERIFIED" | cut -d' ' -f1)
[ -f "$OUT/verified.sha256" ] || echo "$SHA" > "$OUT/verified.sha256"
[ "$SHA" = "$(cat "$OUT/verified.sha256")" ] || { log "verified sha256 $SHA differs from the first start's $(cat "$OUT/verified.sha256")"; exit 1; }
log "verified $SHA"

# review 011 F2: the run code builds the worlds; its seal dependency files (house B's included) must hash as in
# the seal's meta before any VLM call
python3 - "$CODE" "$META" >> "$OUT/job.log" 2>&1 <<'PY' || { log "run code's seal dependency files differ from the meta: no runs"; exit 1; }
import hashlib, json, sys
code, meta = sys.argv[1], json.load(open(sys.argv[2]))
bad = [f for f, h in meta["dep_sha256"].items() if hashlib.sha256(open(f"{code}/{f}", "rb").read()).hexdigest() != h]
print("dep check:", len(meta["dep_sha256"]), "files,", "differ:", bad)
sys.exit(1 if bad else 0)
PY
log "run code's seal dependency files = meta"

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
    scontrol requeue "$JOB" && log "requeued"; exit 0
fi
LEFT=$(python3 -c 'import json,sys,os
ids=[json.loads(x)["task_id"] for x in open(sys.argv[1]) if x.strip()]
print(sum(not os.path.exists(os.path.join(sys.argv[2], i, "result.json")) for i in ids))' "$VERIFIED" "$OUT")
[ "$LEFT" = 0 ] || { log "VLM pass incomplete ($LEFT without result.json): no script runs"; exit 0; }
kill "$VPID" 2>/dev/null; VPID=""
log "VLM pass written (100/100); scripts next (task 011 step 5)"
for S in e3a e3b; do
    TS=$(date +%s)
    bash "$CODE/hpc/phase6_batch.sh" "$CODE" "$VERIFIED" "$OUT/$S" 24 "http://unused" "$STOP" "--script $S" &
    BPID=$!
    while kill -0 "$BPID" 2>/dev/null; do wait "$BPID"; RC=$?; done
    BPID=""
    log "batch $S rc=$RC wall_s $(( $(date +%s) - TS ))"
    if [ "$RC" = 75 ]; then scontrol requeue "$JOB" && log "requeued"; exit 0; fi
done
exit 0
