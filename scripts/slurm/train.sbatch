#!/bin/bash
#SBATCH --job-name=ex-grmt
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#
# Stage-I generalist training at paper scale.
#
#   sbatch scripts/slurm/train.sbatch
#   sbatch --export=ALL,TASK=ExGRMT-Stage2-Flat-Unitree-G1,BASE_CKPT=/path/model.pt \
#          scripts/slurm/train.sbatch
#
# For multi-GPU, raise --gres and set NUM_GPUS; mjlab dispatches through torchrunx.

set -euo pipefail

TASK="${TASK:-ExGRMT-Stage1-Flat-Unitree-G1}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-30000}"
NUM_GPUS="${NUM_GPUS:-1}"

# Offscreen rendering: cluster nodes have no display.
export MUJOCO_GL=egl
# Keep the motion library off the shared home quota if EX_GRMT_DATA is exported.
: "${EX_GRMT_DATA:=$PWD/data}"
export EX_GRMT_DATA

mkdir -p logs/slurm

echo "task=$TASK num_envs=$NUM_ENVS max_iters=$MAX_ITERS gpus=$NUM_GPUS"
nvidia-smi --query-gpu=name,memory.total --format=csv

ARGS=(
  "$TASK"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.max-iterations "$MAX_ITERS"
  --agent.logger tensorboard
)
if [[ -n "${BASE_CKPT:-}" ]]; then
  ARGS+=(--agent.algorithm.base-checkpoint "$BASE_CKPT")
fi
if [[ "$NUM_GPUS" -gt 1 ]]; then
  ARGS+=(--gpu-ids "$(seq -s, 0 $((NUM_GPUS - 1)))")
fi

srun uv run train "${ARGS[@]}"
