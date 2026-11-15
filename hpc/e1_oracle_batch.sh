#!/bin/bash
# Task 009 step 4 (E1 oracle split), inside nav116.sif on a GPU node, after the counted pass is written:
#   bash hpc/e1_oracle_batch.sh <verified.jsonl> <out> <N>
# via hpc/batch_run.sbatch (gpu_short): one `python -m sim.e1_oracle` per task, N at a time; a task with
# <out>/<id>/oracle.json is skipped, a partial dir moved to <out>/aside/. Per-task line in <out>/oracle_job.log.
set -uo pipefail
VERIFIED=$1; OUT=$2; N=$3
mkdir -p "$OUT/aside" "$OUT/logs"
python3 -c 'import json,sys
for x in open(sys.argv[1]):
    if x.strip(): print(json.loads(x)["task_id"])' "$VERIFIED" > "$OUT/ids.txt"
one() {
    id=$1
    [ -f "$OUT/$id/oracle.json" ] && return 0
    [ -e "$OUT/$id" ] && mv "$OUT/$id" "$OUT/aside/$id.$(date +%s)"
    python -m sim.e1_oracle --verified "$VERIFIED" --task-id "$id" --out "$OUT/$id" > "$OUT/logs/$id.log" 2>&1
    echo "[$(date -Is)] $id rc=$? $(tail -1 "$OUT/logs/$id.log")" >> "$OUT/oracle_job.log"
}
export -f one; export VERIFIED OUT
xargs -P "$N" -I{} bash -c 'one {}' < "$OUT/ids.txt"
echo "[$(date -Is)] done: $(ls "$OUT"/*/oracle.json 2>/dev/null | wc -l) / $(wc -l < "$OUT/ids.txt") with oracle.json" >> "$OUT/oracle_job.log"
