#!/bin/bash
# Runs only after a failed primary gate. Arms isolate consolidation from step size.
#SBATCH --job-name=exgrmt-v1p-diag
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --account=<cluster-user>
#SBATCH --qos=user_<cluster-user>
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --array=0-2

set -euo pipefail

PROJECT_DIR="${EX_GRMT_PROJECT_DIR:-<cluster-home>/Ex-GRMT-v1-fix-20260809}"
cd "$PROJECT_DIR"

source /public/software/modules/5.4.0/init/bash
module purge
module load anaconda3/python3.8
module load cuda/11.8
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_PREFIX:-<cluster-home>/.conda/envs/ex-grmt}"

SEED="${SEED:-42}"
ENVS_PER_GPU="${ENVS_PER_GPU:-1024}"
MAX_ITERS="${MAX_ITERS:-500}"
PIPELINE_DIR="$PROJECT_DIR/logs/pipeline/v1-proxy-seed${SEED}"
BASE_CHECKPOINT="$(<"$PIPELINE_DIR/base_checkpoint.txt")"
export EX_GRMT_STRATIFIED_MANIFEST="$(<"$PIPELINE_DIR/stratified_manifest.txt")"
export EX_GRMT_MASTERED_MANIFEST="$(<"$PIPELINE_DIR/mastered_manifest.txt")"
export EX_GRMT_CHALLENGING_MANIFEST="$(<"$PIPELINE_DIR/challenging_manifest.txt")"

case "$SLURM_ARRAY_TASK_ID" in
  0) TASK=ExGRMT-Stage2-Flat-Unitree-G1; LR=1e-3; ARM=full-lr1e3 ;;
  1) TASK=ExGRMT-Stage2-NoCon-Flat-Unitree-G1; LR=1e-3; ARM=nocon-lr1e3 ;;
  2) TASK=ExGRMT-Stage2-Flat-Unitree-G1; LR=1e-4; ARM=full-lr1e4 ;;
  *) echo "unexpected array arm $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

export MUJOCO_GL=egl
export EX_GRMT_DATA="${EX_GRMT_DATA:-$PROJECT_DIR/data}"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

/opt/gridview/slurm/bin/srun train "$TASK" \
  --env.scene.num-envs "$ENVS_PER_GPU" \
  --env.seed "$SEED" \
  --agent.seed "$SEED" \
  --agent.max-iterations "$MAX_ITERS" \
  --agent.run-name "v1-proxy-diagnostic-${ARM}-seed${SEED}" \
  --agent.logger tensorboard \
  --agent.algorithm.base-checkpoint "$BASE_CHECKPOINT" \
  --agent.algorithm.learning-rate "$LR" \
  --agent.algorithm.schedule adaptive \
  --env.commands.motion.recovery-probability 0 \
  --gpu-ids all
