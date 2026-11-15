#!/bin/bash
#SBATCH --job-name=vlm116
#SBATCH --partition=gpu_short
# account: submit with sbatch -A "$PERUN_ACCOUNT"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:40:00
# (path below is relative to the submit dir; create logs/ there first)
#SBATCH --output=logs/%x_%j.out
# vLLM (Qwen3.8-27B, 0.75 GPU mem) + a MuJoCo sim stepping beside it + 3 tool-call checks.
# Submit from the code clone: cd $PROJ/dev/0b/code && sbatch hpc/vlm_check.sbatch   (never --export)
# the image and model folders are read-only here; all writes go under $PROJ.
set -uo pipefail
IMAGE_DIR=${IMAGE_DIR:?set IMAGE_DIR to the folder holding g1.sif / vllm.sif}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the folder holding the Qwen3.8-27B weights}
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
CODE=${SLURM_SUBMIT_DIR:?run under sbatch}
JOB=${SLURM_JOB_ID:?run under sbatch}
OUT=$PROJ/runs/0b/$JOB
VLLM_SIF=$IMAGE_DIR/vllm.sif; G1_SIF=$IMAGE_DIR/g1.sif; MODEL=$MODEL_DIR
T0=$(date +%s)
mkdir -p "$OUT" "$PROJ/cache/vllm" "$PROJ/cache/hf" "$PROJ/cache/tmp/vllm_$JOB"
log() { echo "[$(( $(date +%s) - T0 ))s] $*" | tee -a "$OUT/job.log"; }
VPID=""; SPID=""
cleanup() {
    touch "$OUT/STOP"
    [ -z "$SPID" ] || { wait "$SPID" 2>/dev/null; }
    [ -z "$VPID" ] || kill "$VPID" 2>/dev/null
    sleep 3; tail -n 200 "$OUT/vllm.log" > "$OUT/vllm_tail.log" 2>/dev/null
    [ -z "$VPID" ] || kill -9 "$VPID" 2>/dev/null
}
trap cleanup EXIT
[ -d "$CODE/.git" ] && [ -f "$CODE/hpc/vlm_check.py" ] || { log "bad code dir $CODE"; exit 2; }
log "job $JOB node $(hostname) code $(cd "$CODE" && cat .git/HEAD | cut -c1-12) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
[ "$(nvidia-smi -L | wc -l)" -eq 1 ] || { log "expected 1 GPU"; exit 1; }
[ -d "$MODEL" ] && [ -f "$VLLM_SIF" ] && [ -f "$G1_SIF" ] || { log "missing sif/model"; exit 1; }

# sim: starts first, alone for ~20 s (baseline steps/s), keeps stepping while vLLM loads and the client runs
apptainer exec --nv --cleanenv -B "$PROJ" -B "$IMAGE_DIR:$IMAGE_DIR:ro" -B "$MODEL_DIR:$MODEL_DIR:ro" --pwd "$CODE" \
    --env MUJOCO_GL=egl,MENAGERIE_DIR=$PROJ/ext/mujoco_menagerie,PYTHONUNBUFFERED=1 \
    "$G1_SIF" /opt/venv-g1/bin/python hpc/sim_beside.py "$OUT" > "$OUT/sim.log" 2>&1 &
SPID=$!
for _ in $(seq 60); do [ -f "$OUT/photo.png" ] && break; kill -0 "$SPID" 2>/dev/null || break; sleep 2; done
[ -f "$OUT/photo.png" ] || { log "photo not rendered"; cat "$OUT/sim.log"; exit 3; }
log "photo ready"
sleep 20
TV=$(date +%s)   # vLLM starts here; sim steps/s before this = alone
echo "$TV" > "$OUT/vllm_start_epoch"

export APPTAINERENV_VLLM_API_KEY=$(openssl rand -hex 24)
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
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
LOAD=$(( $(date +%s) - T0 ))
log "healthy: load_time_s $LOAD (job start -> /health 200); vllm-only $(( $(date +%s) - TV ))s"
echo "{\"load_time_s_from_job_start\": $LOAD, \"load_time_s_from_vllm_launch\": $(( $(date +%s) - TV ))}" > "$OUT/load_time.json"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee "$OUT/gpu_mem.txt"
# fresh EGL render by the sim process while vLLM holds its GPU memory; the 3 calls use this photo
touch "$OUT/render_now"
for _ in $(seq 60); do [ -f "$OUT/photo_after.png" ] && break; kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
[ -f "$OUT/photo_after.png" ] || { log "no photo_after.png (sim render failed beside vLLM)"; cat "$OUT/sim.log"; exit 3; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee "$OUT/gpu_mem_after_render.txt"
log "render beside vLLM: $(cat "$OUT/render_after.json")"

# client with the python inside vllm.sif (openai + PIL present); key via env
apptainer exec --cleanenv -B "$PROJ" -B "$IMAGE_DIR:$IMAGE_DIR:ro" -B "$MODEL_DIR:$MODEL_DIR:ro" --pwd "$CODE" "$VLLM_SIF" \
    python3 hpc/vlm_check.py "http://127.0.0.1:$PORT/v1" "$OUT/photo_after.png" "$OUT/vlm_check.json" "$OUT/sim_steps.jsonl" \
    2>&1 | tee "$OUT/client.log"
RC=${PIPESTATUS[0]}
nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee "$OUT/gpu_mem_after.txt"
touch "$OUT/STOP"; wait "$SPID" 2>/dev/null; SPID=""
# steps/s alone (before vLLM launched) vs while loading vLLM vs during client calls
python3 - "$OUT" <<'PY' | tee "$OUT/sim_rates.txt"
import json, sys
o = sys.argv[1]
tv = float(open(o + "/vllm_start_epoch").read())
pts = [json.loads(x) for x in open(o + "/sim_steps.jsonl") if x.strip()]
def rate(a, b):
    w = [p for p in pts if a <= p["t"] <= b]
    return round((w[-1]["steps"] - w[0]["steps"]) / (w[-1]["t"] - w[0]["t"])) if len(w) > 1 else None
print("alone steps/s", rate(0, tv)); print("vllm loading steps/s", rate(tv, pts[-1]["t"]))
PY
log "client rc=$RC"
exit $RC
