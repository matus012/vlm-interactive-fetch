#!/bin/bash
#SBATCH --job-name=e4_116
#SBATCH --partition=gpu_short
# account: submit with sbatch -A "$PERUN_ACCOUNT"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --open-mode=append
# (path below is relative to the submit dir; create logs/ there first)
#SBATCH --output=logs/%x_%j.out
# Task 014 E4 counted pass: seal-014 (both houses), the frozen agent, then the E3b and E3c scripts. One job:
#   1. sim/e4_seal.py regen from the run code, per (house, labels) group present in the seal: the code's dependency
#      files must hash as the seal's meta, every task must regenerate equal -> <out>/verified_<H>.jsonl; any failure ->
#      stop before vLLM.
#   2. vLLM as hpc/e2.sbatch; hpc/phase6_batch.sh per house (NAV116_HOUSE=<H>) -> <out>/<task_id>/ = the counted pass.
#   3. Only after every VLM result is written: E3b (agent/e3.py byte-unchanged) then E3c on the same verified tasks ->
#      <out>/e3b/, <out>/e3c/ (vLLM stopped first).
# Submit: cd $PROJ/code && sbatch hpc/e4.sbatch <run commit> <tag> <N> <E4 seal sha256>
# Output: $PROJ/runs/014/<tag>/. Restart-safe as phase6.sbatch (finished tasks are skipped).
set -uo pipefail
IMAGE_DIR=${IMAGE_DIR:?set IMAGE_DIR to the folder holding g1.sif / vllm.sif}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the folder holding the Qwen3.8-27B weights}
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
CODE=${SLURM_SUBMIT_DIR:?run under sbatch}
JOB=${SLURM_JOB_ID:?run under sbatch}; RESTART=${SLURM_RESTART_COUNT:-0}
RUN_COMMIT=${1:?freeze-checked run commit (full sha)}; TAG=${2:?tag}; N=${3:-16}; SEAL_SHA=${4:?E4 seal sha256}
OUT=$PROJ/runs/014/$TAG
SEALED=${E4_SEALED:-$PROJ/sealed/e4_100.jsonl}; META=${E4_META:-$PROJ/sealed/e4_100.meta.json}   # smoke: a dev dry-run seal via env
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
[ -d "$CODE/.git" ] && [ -f "$CODE/sim/e4_seal.py" ] || { log "bad code dir $CODE"; exit 2; }
[ -f "$SEALED" ] && [ -f "$META" ] || { log "missing sealed file / meta in $PROJ/sealed"; exit 2; }
HEAD_SHA=$(cat "$CODE/.git/HEAD")
log "job $JOB restart $RESTART node $(hostname) code $HEAD_SHA run_commit $RUN_COMMIT seal_sha ${SEAL_SHA:0:16} tag $TAG N $N $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
[ "${#RUN_COMMIT}" = 40 ] && [ "$HEAD_SHA" = "$RUN_COMMIT" ] || { log "code HEAD $HEAD_SHA is not the run commit $RUN_COMMIT"; exit 2; }
[ "$(sha256sum "$SEALED" | cut -d' ' -f1)" = "$SEAL_SHA" ] || { log "sealed file sha256 differs from the given seal sha"; exit 2; }
[ "$(nvidia-smi -L | wc -l)" -eq 1 ] || { log "expected 1 GPU"; exit 1; }
[ -d "$MODEL" ] && [ -f "$VLLM_SIF" ] && [ -f "$NAV_SIF" ] || { log "missing sif/model"; exit 1; }

nav() {  # nav <house> <cmd...>: one container with that house
    local H=$1; shift
    apptainer exec --nv --cleanenv \
        --env MUJOCO_GL=egl,PROJ=$PROJ,MENAGERIE_DIR=$PROJ/ext/mujoco_menagerie,PYTHONDONTWRITEBYTECODE=1,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=1,NAV116_HOUSE=$H \
        -B "$PROJ" --pwd "$CODE" "$NAV_SIF" "$@"
}
HOUSES=$(python3 -c 'import json,sys; print(" ".join(sorted({json.loads(x).get("house","A") for x in open(sys.argv[1]) if x.strip()})))' "$SEALED")
for H in $HOUSES; do
    V=$OUT/verified_$H.jsonl
    if [ ! -f "$V" ]; then
        rm -f "$OUT/regen_${H}"_*.jsonl
        for L in plain labels; do
            NL=$(python3 -c 'import json,sys; print(sum(json.loads(x).get("house","A")==sys.argv[2] and bool(json.loads(x).get("labels"))==(sys.argv[3]=="labels") for x in open(sys.argv[1]) if x.strip()))' "$SEALED" "$H" "$L")
            [ "$NL" = 0 ] && continue
            FLAG=""; [ "$L" = labels ] && FLAG="--labels"
            TR=$(date +%s)
            R=$(nav "$H" python -m sim.e4_seal regen --sealed "$SEALED" --meta "$META" --out "$OUT/regen_${H}_$L.jsonl" $FLAG 2> "$OUT/logs/regen_${H}_${L}_${JOB}_${RESTART}.err")
            RC=$?
            log "regen house $H $L rc=$RC wall_s $(( $(date +%s) - TR )): $R"
            [ "$RC" = 0 ] || { log "regeneration failed: no runs (stderr in logs/regen_${H}_${L}_*.err)"; exit 1; }
        done
        cat "$OUT/regen_${H}"_*.jsonl > "$V.tmp" && mv "$V.tmp" "$V"
    fi
    log "verified_$H $(wc -l < "$V") tasks $(sha256sum "$V" | cut -c1-16)"
done
TOTAL=$(cat "$OUT"/verified_*.jsonl | wc -l)
[ "$TOTAL" = "$(grep -c . "$SEALED")" ] || { log "verified $TOTAL != sealed $(grep -c . "$SEALED")"; exit 1; }

export APPTAINERENV_VLLM_API_KEY=$(openssl rand -hex 24)
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
left() { python3 -c 'import json,sys,os,glob
ids=[json.loads(x)["task_id"] for f in glob.glob(os.path.join(sys.argv[1],"verified_*.jsonl")) for x in open(f) if x.strip()]
print(sum(not os.path.exists(os.path.join(sys.argv[2], i, "result.json")) for i in ids))' "$OUT" "$1"; }
if [ "$(left "$OUT")" != 0 ]; then
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
    for _ in $(seq 240); do
        kill -0 "$VPID" 2>/dev/null || { log "vllm exited before healthy"; exit 1; }
        [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] && break
        [ -e "$STOP" ] && break
        sleep 5
    done
    [ -e "$STOP" ] && { log "stop during vLLM load"; scontrol requeue "$JOB" && log "requeued"; exit 0; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] || { log "not healthy after 20 min"; exit 1; }
    log "vllm healthy: load_time_s $(( $(date +%s) - TV ))"
    for H in $HOUSES; do
        TB=$(date +%s)
        NAV116_HOUSE=$H bash "$CODE/hpc/phase6_batch.sh" "$CODE" "$OUT/verified_$H.jsonl" "$OUT" "$N" "http://127.0.0.1:$PORT/v1" "$STOP" &
        BPID=$!
        while kill -0 "$BPID" 2>/dev/null; do wait "$BPID"; RC=$?; done
        BPID=""
        log "VLM batch house $H rc=$RC wall_s $(( $(date +%s) - TB ))"
        if [ "$RC" = 75 ]; then scontrol requeue "$JOB" && log "requeued"; exit 0; fi
    done
    kill "$VPID" 2>/dev/null; VPID=""
fi
L=$(left "$OUT")
[ "$L" = 0 ] || { log "VLM pass incomplete ($L without result.json): no script runs"; exit 0; }
log "VLM pass written ($TOTAL/$TOTAL); scripts next"
for S in e3b e3c; do
    for H in $HOUSES; do
        TS=$(date +%s)
        NAV116_HOUSE=$H bash "$CODE/hpc/phase6_batch.sh" "$CODE" "$OUT/verified_$H.jsonl" "$OUT/$S" 14 "http://unused" "$STOP" "--script $S" &
        BPID=$!
        while kill -0 "$BPID" 2>/dev/null; do wait "$BPID"; RC=$?; done
        BPID=""
        log "batch $S house $H rc=$RC wall_s $(( $(date +%s) - TS ))"
        if [ "$RC" = 75 ]; then scontrol requeue "$JOB" && log "requeued"; exit 0; fi
    done
done
log "done: VLM $TOTAL, e3b $(ls "$OUT"/e3b/*/result.json 2>/dev/null | wc -l), e3c $(ls "$OUT"/e3c/*/result.json 2>/dev/null | wc -l)"
exit 0
