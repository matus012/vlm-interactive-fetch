#!/bin/bash
# Task 004: run every task of a taskset with N parallel sim workers (nav116.sif) against a running vLLM. Host bash.
#   phase3_batch.sh <code> <taskset.json> <out> <N> <base_url> <stop_file>     (called by hpc/phase3.sbatch)
# Per task: <out>/<task_id>/result.json exists -> skip; dir exists without it -> moved aside to
# <out>/aside/<task_id>.<epoch> and re-run. Logs: <out>/logs/<task_id>.<epoch>.log; per-task start/end in <out>/job.log.
# Stop file appears -> no new task, running workers killed (their dirs stay partial), exit 75. All done -> exit 0.
set -uo pipefail
CODE=$1; TASKSET=$2; OUT=$3; N=$4; URL=$5; STOP=$6
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
NAV_SIF=$PROJ/images/nav116.sif
mkdir -p "$OUT/aside" "$OUT/logs"
IDS=()
log() { echo "[$(date -Is)] $*" >> "$OUT/job.log"; }
# TASKSET may be several taskset files joined by commas (task 005: one job runs every type's set); ids are unique
# across sets (type-seed). SET[id] = the file the id comes from.
declare -A SET=()
while IFS=$'\t' read -r i f; do IDS+=("$i"); SET[$i]=$f; done < <(python3 -c 'import json,sys
for f in sys.argv[1].split(","):
    for t in json.load(open(f))["tasks"]: print(t["task_id"], f, sep="\t")' "$TASKSET")
declare -A RUN=()   # pid -> task id (=() : an unassigned array is unbound under set -u)
reap() {
    local p rc id   # local: the caller's loop variable is also called id
    for p in "${!RUN[@]}"; do
        if ! kill -0 "$p" 2>/dev/null; then
            wait "$p"; rc=$?
            id=${RUN[$p]}; unset "RUN[$p]"
            log "end $id rc=$rc $([ -f "$OUT/$id/result.json" ] && python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("pass" if r["pass"] else "fail " + str(r["label"]), "steps", r["steps_used"], "wall_s", r["timing"]["run_wall_s"])' "$OUT/$id/result.json" || echo incomplete)"
        fi
    done
}
stop_all() {
    log "stop file seen: killing ${#RUN[@]} running task(s): ${RUN[*]}"
    for p in "${!RUN[@]}"; do kill "$p" 2>/dev/null; done
    sleep 5
    for p in "${!RUN[@]}"; do kill -9 "$p" 2>/dev/null; wait "$p" 2>/dev/null; done
    exit 75
}
log "batch start: ${#IDS[@]} tasks, N=$N, taskset $TASKSET"
for id in "${IDS[@]}"; do
    if [ -f "$OUT/$id/result.json" ]; then log "skip $id (result.json)"; continue; fi
    while [ "${#RUN[@]}" -ge "$N" ]; do [ -e "$STOP" ] && stop_all; sleep 5; reap; done
    [ -e "$STOP" ] && stop_all
    E=$(date +%s)
    if [ -e "$OUT/$id" ]; then mv "$OUT/$id" "$OUT/aside/$id.$E"; log "moved partial $id -> aside/$id.$E"; fi
    apptainer exec --nv --cleanenv \
        --env MUJOCO_GL=egl,PROJ=$PROJ,MENAGERIE_DIR=$PROJ/ext/mujoco_menagerie,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONUNBUFFERED=1,NAV116_HOUSE=${NAV116_HOUSE:-A} \
        -B "$PROJ" --pwd "$CODE" "$NAV_SIF" \
        python -m sim.agent_run --taskset "${SET[$id]}" --task-id "$id" --out "$OUT/$id" --base-url "$URL" \
        --model-name qwen3.8-27b --poses "${POSES:-perceived}" > "$OUT/logs/$id.$E.log" 2>&1 &
    RUN[$!]=$id
    log "start $id pid $!"
done
while [ "${#RUN[@]}" -gt 0 ]; do [ -e "$STOP" ] && stop_all; sleep 5; reap; done
left=0
for id in "${IDS[@]}"; do [ -f "$OUT/$id/result.json" ] || left=$((left + 1)); done
log "batch end: $left task(s) without result.json"
exit 0
