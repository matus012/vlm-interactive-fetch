#!/bin/bash
#SBATCH --job-name=body0a
#SBATCH --partition=gpu_short
# account: submit with sbatch -A "$PERUN_ACCOUNT"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
# (path below is relative to the submit dir; create logs/ there first)
#SBATCH --output=logs/%x_%j.out
# 0a body check on an H200: G0a-cam/img/base/grasp + MuJoCo steps/s. Code: hpc/sync_code.sh $PROJ/dev/0a/code
# Submit from <login-node>:  sbatch $PROJ/dev/0a/code/hpc/body_check.sbatch   (never --export)
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
CODE=${CODE:-$PROJ/dev/0a/code}
SIF=${IMAGE_DIR:?set IMAGE_DIR to the folder holding g1.sif / vllm.sif}/g1.sif
OUT=$PROJ/runs/0a/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "start $(date -Is) node $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) code $(cat "$CODE/.git/HEAD" 2>/dev/null)"
cd "$CODE"
apptainer exec --nv --cleanenv --env MUJOCO_GL=egl,MENAGERIE_DIR=$PROJ/ext/mujoco_menagerie -B "$PROJ" -B "$IMAGE_DIR" "$SIF" \
    /opt/venv-g1/bin/python -m sim.body_check --out "$OUT"
echo "rc=$? end $(date -Is) out $OUT"
