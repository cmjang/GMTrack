#!/bin/bash
# Stage-I for the online-causal + heading task.
#
# The actor sees the same 11 pure-past reference tokens as the causal baseline and
# therefore has zero future-frame lookahead. Each 38D reference token is extended by
# the existing Heading task's 6D robot-to-reference root-orientation error; training
# resets receive the same yaw-only +/-0.2 rad perturbation. Future reference remains
# confined to the training-only critic and intent reconstruction target.
# Feet terminate at 0.20 m pelvis-XY/yaw-realigned horizontal error or 0.15 m Z error.
#SBATCH --job-name=gmtrack-csl-hdg-s1
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

PROJECT_DIR="${GMTRACK_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
test -d src/gmtrack || {
  echo "PROJECT_DIR=$PROJECT_DIR is not a GMTrack checkout (no src/gmtrack)." >&2
  exit 1
}

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
RUN_NAME="${RUN_NAME:-causal-heading-seed${SEED}}"
DATA_DIR="${GMTRACK_DATA:-$PROJECT_DIR/data}"
MANIFEST="${GMTRACK_STAGE1_MANIFEST:-$DATA_DIR/current/stage1_paper_mix_final_backflip_cartwheel_balanced_grounded_no_fall_getup.json}"
PIPELINE_DIR="$PROJECT_DIR/logs/pipeline/causal-heading-seed${SEED}"
RECON_COEF="${RECON_COEF:-0.5}"
KL_COEF="${KL_COEF:-0.0005}"

mkdir -p "$PROJECT_DIR/logs/slurm" "$PIPELINE_DIR"
test -f "$MANIFEST"

export MUJOCO_GL=egl
export GMTRACK_DATA="$DATA_DIR"
export GMTRACK_STAGE1_MANIFEST="$MANIFEST"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "causal Heading Stage I: seed=$SEED total_envs=$((ENVS_PER_GPU * NUM_GPUS)) iterations=$MAX_ITERS"
echo "  actor future frames: none | heading feedback: 6D | reset yaw: +/-0.2 rad"
echo "  foot termination: pelvis-realigned XY=0.20 m | Z=0.15 m"
echo "  intent aux: recon=$RECON_COEF kl=$KL_COEF"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

ARGS=(
  GMTrack-Stage1-Causal-Flat-Unitree-G1
  --env.scene.num-envs "$ENVS_PER_GPU"
  --env.seed "$SEED"
  --agent.seed "$SEED"
  --agent.max-iterations "$MAX_ITERS"
  --agent.run-name "$RUN_NAME"
  --agent.logger tensorboard
  --agent.actor.fsq-levels 32
  --agent.algorithm.learning-rate 1e-3
  --agent.algorithm.schedule adaptive
  --agent.algorithm.intent-reconstruction-coef "$RECON_COEF"
  --agent.algorithm.intent-kl-coef "$KL_COEF"
  --env.commands.motion.manifest "$MANIFEST"
  --env.commands.motion.recovery-probability 0
  --gpu-ids all
)

/opt/gridview/slurm/bin/srun train "${ARGS[@]}"

RUN_DIR="$(find logs/rsl_rl/gmtrack_stage1_causal_heading -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_NAME}" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
CHECKPOINT="$RUN_DIR/model_$((MAX_ITERS - 1)).pt"
test -f "$CHECKPOINT"
realpath "$CHECKPOINT" > "$PIPELINE_DIR/base_checkpoint.txt"
echo "causal Heading Stage-I handoff: $CHECKPOINT"
echo "Next: re-stratify with THIS checkpoint before any matching Stage II run."
