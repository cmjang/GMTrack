#!/bin/bash
# 2607.20110v1 Stage-I public-data proxy. Starts from scratch; no recovery proxy.
#SBATCH --job-name=gmtrack-v1p-s1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --account=<cluster-user>
#SBATCH --qos=user_<cluster-user>
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=5-00:00:00

set -euo pipefail

PROJECT_DIR="${GMTRACK_PROJECT_DIR:?set GMTRACK_PROJECT_DIR to the checkout to run from}"
cd "$PROJECT_DIR"

source /public/software/modules/5.4.0/init/bash
module purge
module load anaconda3/python3.8
module load cuda/11.8
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_PREFIX:?set CONDA_ENV_PREFIX to the conda environment prefix}"

SEED="${SEED:-42}"
ENVS_PER_GPU="${ENVS_PER_GPU:-1024}"
MAX_ITERS="${MAX_ITERS:-100000}"
NUM_GPUS=4
RUN_NAME="${RUN_NAME:-v1-final-backflip-seed${SEED}}"
DATA_DIR="${GMTRACK_DATA:-$PROJECT_DIR/data}"
MANIFEST="${GMTRACK_STAGE1_MANIFEST:-$DATA_DIR/current/stage1_paper_mix_final_backflip_cartwheel_balanced_grounded_no_fall_getup.json}"
PIPELINE_DIR="$PROJECT_DIR/logs/pipeline/v1-final-backflip-seed${SEED}"

mkdir -p "$PROJECT_DIR/logs/slurm" "$PIPELINE_DIR"
test -f "$MANIFEST"
echo "55cca5c02f16c659e4ab3baf08d9ad1fb69865f37cfba084958ade0911cf51fe  2607.20110v1.pdf" | sha256sum -c -

export MUJOCO_GL=egl
export GMTRACK_DATA="$DATA_DIR"
export GMTRACK_STAGE1_MANIFEST="$MANIFEST"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "v1 final-backflip Stage I: seed=$SEED total_envs=$((ENVS_PER_GPU * NUM_GPUS)) iterations=$MAX_ITERS recovery=0"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

ARGS=(
  GMTrack-Stage1-Flat-Unitree-G1
  --env.scene.num-envs "$ENVS_PER_GPU"
  --env.seed "$SEED"
  --agent.seed "$SEED"
  --agent.max-iterations "$MAX_ITERS"
  --agent.run-name "$RUN_NAME"
  --agent.logger tensorboard
  --agent.actor.fsq-levels 32
  --agent.algorithm.learning-rate 1e-3
  --agent.algorithm.schedule adaptive
  --env.commands.motion.manifest "$MANIFEST"
  --env.commands.motion.recovery-probability 0
  --gpu-ids all
)

/opt/gridview/slurm/bin/srun train "${ARGS[@]}"

RUN_DIR="$(find logs/rsl_rl/gmtrack_stage1 -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_NAME}" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
CHECKPOINT="$RUN_DIR/model_$((MAX_ITERS - 1)).pt"
test -f "$CHECKPOINT"
realpath "$CHECKPOINT" > "$PIPELINE_DIR/base_checkpoint.txt"
echo "Stage-I handoff: $CHECKPOINT"
