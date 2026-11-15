#!/bin/bash
#SBATCH --job-name=restart116
#SBATCH --partition=gpu_short
# account: submit with sbatch -A "$PERUN_ACCOUNT"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --open-mode=append
# (path below is relative to the submit dir; create logs/ there first)
#SBATCH --output=logs/%x_%j.out
# Ledger pattern for every PERUN job (see hpc/RESTART.md). Submit:
#   sbatch hpc/restart_check.sbatch <ledger_dir> [requeue|successor] [n_items]
# Ledger dir and mode are POSITIONAL args (never --export: the job goes HELD).
set -u
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
CODE=$PROJ/dev/0c/code
LEDGER=$1; MODE=${2:-requeue}; N=${3:-30}
ME=${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}
echo "start $(date -Is) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) cpus=$SLURM_CPUS_PER_TASK mode=$MODE"
python3 $CODE/hpc/ledger.py init $LEDGER $N
# USR1 (90 s before the wall): tell the worker to finish its current item and not start another
trap 'echo "USR1 $(date -Is)"; touch $LEDGER/stop/$ME' USR1
apptainer exec --nv --cleanenv --env SLURM_JOB_ID=$SLURM_JOB_ID,SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0} -B "$PROJ" -B "$IMAGE_DIR" ${IMAGE_DIR:?set IMAGE_DIR to the folder holding g1.sif / vllm.sif}/g1.sif /opt/venv-g1/bin/python -I \
    -c "import sys; sys.path.insert(0,'$CODE/hpc'); sys.argv=['d','$LEDGER','$PROJ/ext/mujoco_menagerie/stanford_tidybot/scene.xml','30']; exec(open('$CODE/hpc/ledger_demo.py').read())" &
W=$!
while kill -0 $W 2>/dev/null; do wait $W; RC=$?; done
echo "worker rc=$RC $(date -Is)"
if [ "$RC" = 75 ]; then
    if [ "$MODE" = successor ]; then
        sbatch --dependency=afterany:$SLURM_JOB_ID $CODE/hpc/restart_check.sbatch $LEDGER $MODE $N
    else
        scontrol requeue $SLURM_JOB_ID && echo "requeued $(date -Is)"
    fi
fi
echo "end $(date -Is)"
