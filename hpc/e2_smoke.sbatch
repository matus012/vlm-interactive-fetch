#!/bin/bash
#SBATCH --job-name=e2s_116
#SBATCH --partition=gpu_short
# account: submit with sbatch -A "$PERUN_ACCOUNT"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
# (path below is relative to the submit dir; create logs/ there first)
#SBATCH --output=logs/%x_%j.out
# Task 011 step 2: the VLM smoke in house B (<= 3 dev-seed runs, plumbing only) = hpc/phase3.sbatch with
# NAV116_HOUSE=B exported (hpc/phase3_batch.sh passes it into each sim container). Same positional args:
#   cd $PROJ/code && sbatch hpc/e2_smoke.sbatch tasksets/e2_smoke.json e2smoke 3
export NAV116_HOUSE=B
exec bash "${SLURM_SUBMIT_DIR:?run under sbatch}/hpc/phase3.sbatch" "$@"
