#!/bin/bash
#SBATCH --job-name=b116
# account: submit with sbatch -A "$PERUN_ACCOUNT"
#SBATCH --open-mode=append
# (path below is relative to the submit dir; create logs/ there first)
#SBATCH --output=logs/%x_%j.out
# Generic sim batch on PERUN (ruling 2026-11-13: batch sim work runs on PERUN, the laptop only pytest + short dev runs).
# One command, run in nav116.sif from a synced code clone (hpc/sync_code.sh <code_dir>), outputs under $PROJ.
#   ssh <login-node> "sbatch -p gpu_short --gres=gpu:1 -c 16 --mem 64G -t 01:00:00 $PROJ/bin/batch_run.sbatch <code_dir> <out_dir> '<cmd>'"
#   ssh <login-node> "sbatch -p cpu_short -c 32 --mem 64G -t 01:00:00 $PROJ/bin/batch_run.sbatch <code_dir> <out_dir> '<cmd>'"
# RENDERER RULE: CPU nodes render with Mesa llvmpipe, whose segmentation px differ from NVIDIA EGL (hall_px seed 1000:
# red 138 vs 130). Anything whose result depends on rendered px (task draws / sampler / hall_px / locate / look /
# script gates / cover) runs on a GPU node; cpu_short only for physics-only work. The job logs which renderer it got.
# Never sbatch --export. Record every job in your track's runs/<track>/perun_jobs.txt (id, partition, cpus, wall).
set -uo pipefail
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
CODE=${1:?code dir}; OUT=${2:?out dir}; CMD=${3:?command}
[[ $CODE == "$PROJ"/* && $OUT == "$PROJ"/* ]] || { echo "code and out must be under $PROJ"; exit 2; }
mkdir -p "$OUT"
NV=""; command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && NV=--nv
T0=$(date +%s)
echo "[$(date -Is)] job ${SLURM_JOB_ID:-?} node $(hostname) part ${SLURM_JOB_PARTITION:-?} cpus ${SLURM_CPUS_PER_TASK:-?} gpu ${NV:-none} code $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null) cmd: $CMD" | tee -a "$OUT/batch_job.log"
apptainer exec $NV --cleanenv \
    --env MUJOCO_GL=egl,PROJ=$PROJ,MENAGERIE_DIR=$PROJ/ext/mujoco_menagerie,PYTHONPATH=.,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=1 \
    -B "$PROJ" --pwd "$CODE" "$PROJ/images/nav116.sif" bash -c "$CMD" >> "$OUT/batch_stdout.txt" 2>&1
RC=$?
echo "[$(date -Is)] job ${SLURM_JOB_ID:-?} rc=$RC wall_s=$(( $(date +%s) - T0 ))" | tee -a "$OUT/batch_job.log"
exit $RC
