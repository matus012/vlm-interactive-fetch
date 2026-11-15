#!/bin/bash
#SBATCH --job-name=img116
#SBATCH --partition=gpu_short
# account: submit with sbatch -A "$PERUN_ACCOUNT"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
# (path below is relative to the submit dir; create logs/ there first)
#SBATCH --output=logs/%x_%j.out
# 0d check: nav116.sif EGL renderer + one TidyBot frame. Submit from a synced clone: cd $PROJ/dev/0d/code && sbatch hpc/image_check.sbatch
set -uo pipefail
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
REPO=${SLURM_SUBMIT_DIR:?}
OUT=$PROJ/runs/0d/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "$(date -Is) node $(hostname) cpus $(nproc) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" | tee "$OUT/job.log"
apptainer exec --nv --cleanenv --env PROJ=$PROJ -B $PROJ $PROJ/images/nav116.sif \
  python $REPO/hpc/image_check.py "$OUT" 2>&1 | tee -a "$OUT/job.log"
echo "$(date -Is) done rc=${PIPESTATUS[0]}" | tee -a "$OUT/job.log"
