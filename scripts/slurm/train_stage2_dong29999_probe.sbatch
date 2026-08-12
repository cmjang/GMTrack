#!/bin/bash
# Current-code PACE+STAR gate over artifacts produced specifically from the
# imported <cluster-user> model_29999.pt. This is a diagnostic, not a formal v1 result.
#SBATCH --job-name=exgrmt-d29999-s2gate
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
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_DIR="${EX_GRMT_PROJECT_DIR:-<cluster-home>/Ex-GRMT-v1-fix-20260809}"
DATA_DIR="${EX_GRMT_DATA:-<cluster-home>/Ex-GRMT-v1-fix-20260809-data}"
SEED="${SEED:-42}"
PROBE_TAG="${PROBE_TAG:-dong29999-latest-s2-probe-seed${SEED}}"
PIPELINE_DIR="$PROJECT_DIR/logs/pipeline/$PROBE_TAG"
ENVS_PER_GPU="${ENVS_PER_GPU:-1024}"
MAX_ITERS="${MAX_ITERS:-500}"
BASE_CHECKPOINT="$(<"$PIPELINE_DIR/base_checkpoint.txt")"
STRATIFIED="$(<"$PIPELINE_DIR/stratified_manifest.txt")"
MASTERED="$(<"$PIPELINE_DIR/mastered_manifest.txt")"
CHALLENGING="$(<"$PIPELINE_DIR/challenging_manifest.txt")"
RUN_NAME="${RUN_NAME:-$PROBE_TAG}"

EXPECTED_BASE_SHA256=e6e978dd124de45a90d41212b462a4b5fac351b539e9552026bd680db7369aef

cd "$PROJECT_DIR"
mkdir -p logs/slurm "$PIPELINE_DIR"
for path in "$BASE_CHECKPOINT" "$STRATIFIED" "$MASTERED" "$CHALLENGING"; do
  test -f "$path"
done
echo "$EXPECTED_BASE_SHA256  $BASE_CHECKPOINT" | sha256sum -c -

source /public/software/modules/5.4.0/init/bash
module purge
module load anaconda3/python3.8
module load cuda/11.8
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_PREFIX:-<cluster-home>/.conda/envs/ex-grmt}"

export MUJOCO_GL=egl
export EX_GRMT_DATA="$DATA_DIR"
export EX_GRMT_STRATIFIED_MANIFEST="$STRATIFIED"
export EX_GRMT_MASTERED_MANIFEST="$MASTERED"
export EX_GRMT_CHALLENGING_MANIFEST="$CHALLENGING"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

ARGS=(
  ExGRMT-Stage2-Flat-Unitree-G1
  --env.scene.num-envs "$ENVS_PER_GPU"
  --env.seed "$SEED"
  --agent.seed "$SEED"
  --agent.max-iterations "$MAX_ITERS"
  --agent.run-name "$RUN_NAME"
  --agent.logger tensorboard
  --agent.algorithm.base-checkpoint "$BASE_CHECKPOINT"
  --agent.algorithm.learning-rate 1e-3
  --agent.algorithm.schedule adaptive
  --env.commands.motion.recovery-probability 0
  --gpu-ids all
)

/opt/gridview/slurm/bin/srun train "${ARGS[@]}"

RUN_DIR="$(find logs/rsl_rl/ex_grmt_stage2 -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_NAME}" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
test -d "$RUN_DIR"
python -m ex_grmt.scripts.check_stage2_gate \
  --run-dir "$RUN_DIR" \
  --out "$PIPELINE_DIR/gate_report.json" \
  --expected-iterations "$MAX_ITERS"
