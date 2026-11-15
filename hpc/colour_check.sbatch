#!/bin/bash
#SBATCH --job-name=col116
#SBATCH --partition=gpu_short
# account: submit with sbatch -A "$PERUN_ACCOUNT"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:40:00
# (path below is relative to the submit dir; create logs/ there first)
#SBATCH --output=logs/%x_%j.out
# G1-colour (Qwen3.8-27B names the box colours in <=12 look photos) + MuJoCo sims-beside-vLLM scaling (N=1,4,8).
# Submit from the code clone: cd $PROJ/dev/1v/code && sbatch hpc/colour_check.sbatch <photos_dir>   (never --export)
# the image and model folders are read-only here; all writes go under $PROJ.
set -uo pipefail
IMAGE_DIR=${IMAGE_DIR:?set IMAGE_DIR to the folder holding g1.sif / vllm.sif}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the folder holding the Qwen3.8-27B weights}
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
CODE=${SLURM_SUBMIT_DIR:?run under sbatch}
JOB=${SLURM_JOB_ID:?run under sbatch}
PHOTOS=${1:?photos dir (contains truth.json + <id>.png)}
OUT=$PROJ/runs/1v/$JOB
VLLM_SIF=$IMAGE_DIR/vllm.sif; NAV_SIF=$PROJ/images/nav116.sif; MODEL=$MODEL_DIR
T0=$(date +%s)
mkdir -p "$OUT" "$PROJ/cache/vllm" "$PROJ/cache/hf" "$PROJ/cache/tmp/vllm_$JOB"
log() { echo "[$(( $(date +%s) - T0 ))s] $*" | tee -a "$OUT/job.log"; }
VPID=""; LPID=""
cleanup() {
    [ -z "$LPID" ] || kill "$LPID" 2>/dev/null
    [ -z "$VPID" ] || kill "$VPID" 2>/dev/null
    sleep 3; tail -n 200 "$OUT/vllm.log" > "$OUT/vllm_tail.log" 2>/dev/null
    [ -z "$VPID" ] || kill -9 "$VPID" 2>/dev/null
}
trap cleanup EXIT
[ -d "$CODE/.git" ] && [ -f "$CODE/hpc/colour_check.py" ] && [ -f "$PHOTOS/truth.json" ] || { log "bad code/photos dir $CODE $PHOTOS"; exit 2; }
log "job $JOB node $(hostname) cpus $SLURM_CPUS_PER_TASK code $(cd "$CODE" && cat .git/HEAD | cut -c1-12) photos $PHOTOS $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
[ "$(nvidia-smi -L | wc -l)" -eq 1 ] || { log "expected 1 GPU"; exit 1; }
[ -d "$MODEL" ] && [ -f "$VLLM_SIF" ] && [ -f "$NAV_SIF" ] || { log "missing sif/model"; exit 1; }
cp -r "$PHOTOS" "$OUT/photos"

export APPTAINERENV_VLLM_API_KEY=$(openssl rand -hex 24)
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TV=$(date +%s)
apptainer exec --nv --cleanenv -B "$PROJ" -B "$IMAGE_DIR:$IMAGE_DIR:ro" -B "$MODEL_DIR:$MODEL_DIR:ro" \
    --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HOME=$PROJ/cache/hf,VLLM_CACHE_ROOT=$PROJ/cache/vllm,TMPDIR=$PROJ/cache/tmp/vllm_$JOB,VLLM_NO_USAGE_STATS=1,DO_NOT_TRACK=1,PYTHONUNBUFFERED=1 \
    "$VLLM_SIF" vllm serve "$MODEL" \
    --served-model-name qwen3.8-27b \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.75 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image": 4}' > "$OUT/vllm.log" 2>&1 &
VPID=$!
for _ in $(seq 240); do   # bounded 20 min
    kill -0 "$VPID" 2>/dev/null || { log "vllm exited before healthy"; exit 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] && break
    sleep 5
done
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] || { log "not healthy after 20 min"; exit 1; }
log "healthy: load_time_s $(( $(date +%s) - TV )) (vllm launch -> /health 200); $(( $(date +%s) - T0 )) s from job start"
echo "{\"load_time_s_from_job_start\": $(( $(date +%s) - T0 )), \"load_time_s_from_vllm_launch\": $(( $(date +%s) - TV ))}" > "$OUT/load_time.json"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee "$OUT/gpu_mem.txt"

# 1) the scored pass (alone on the GPU server)
apptainer exec --cleanenv -B "$PROJ" -B "$IMAGE_DIR:$IMAGE_DIR:ro" -B "$MODEL_DIR:$MODEL_DIR:ro" --pwd "$CODE" "$VLLM_SIF" \
    python3 hpc/colour_check.py vlm "http://127.0.0.1:$PORT/v1" "$OUT/photos" "$OUT/colour_check.json" 2>&1 | tee "$OUT/client.log"
RC=${PIPESTATUS[0]}
# 2) sims beside vLLM: a background client loops the same calls (one stream) while N = 1, 4, 8 sims run 45 s each
# (steps/s per sim in sims_<N>/sim_<i>.json; load_calls.jsonl has the call latencies/tokens during the phases)
mkdir -p "$OUT/load" "$OUT/sims_alone"   # baseline: 1 sim, vLLM idle
apptainer exec --nv --cleanenv --env PROJ=$PROJ,MENAGERIE_DIR=$PROJ/ext/mujoco_menagerie -B "$PROJ" --pwd "$CODE" "$NAV_SIF" \
    python hpc/colour_check.py sim "$OUT/sims_alone" 20 1 > "$OUT/sims_alone/sim_1.log" 2>&1
apptainer exec --cleanenv -B "$PROJ" -B "$IMAGE_DIR:$IMAGE_DIR:ro" -B "$MODEL_DIR:$MODEL_DIR:ro" --pwd "$CODE" "$VLLM_SIF" \
    python3 hpc/colour_check.py load "http://127.0.0.1:$PORT/v1" "$OUT/photos" "$OUT/load" > "$OUT/load/client.log" 2>&1 &
LPID=$!
sleep 10
(
for N in 1 4 8; do
    mkdir -p "$OUT/sims_$N"; PIDS=""
    for i in $(seq "$N"); do
        apptainer exec --nv --cleanenv --env PROJ=$PROJ,MENAGERIE_DIR=$PROJ/ext/mujoco_menagerie -B "$PROJ" --pwd "$CODE" "$NAV_SIF" \
            python hpc/colour_check.py sim "$OUT/sims_$N" 45 "$i" > "$OUT/sims_$N/sim_$i.log" 2>&1 &
        PIDS="$PIDS $!"
    done
    wait $PIDS
done
); echo "$(date +%s)" > "$OUT/sims_end_epoch"
touch "$OUT/load/STOP"; wait "$LPID" 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee "$OUT/gpu_mem_after.txt"
log "client rc=$RC"
exit $RC
