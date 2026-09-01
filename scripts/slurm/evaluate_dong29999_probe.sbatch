#!/bin/bash
# Nominal one-rollout evaluation over the strict challenging artifact. Set
# EVAL_PHASE=stage1 for the imported baseline or stage2 for the latest gate model.
# The JSON includes per-source metrics, so seed-backflip can be read directly.
#SBATCH --job-name=gmtrack-d29999-eval
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --account=<cluster-user>
#SBATCH --qos=user_<cluster-user>
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${GMTRACK_PROJECT_DIR:?set GMTRACK_PROJECT_DIR to the checkout to run from}"
DATA_DIR="${GMTRACK_DATA:-$PROJECT_DIR/data}"
SEED="${SEED:-42}"
PROBE_TAG="${PROBE_TAG:-dong29999-latest-s2-probe-seed${SEED}}"
PIPELINE_DIR="$PROJECT_DIR/logs/pipeline/$PROBE_TAG"
EVAL_PHASE="${EVAL_PHASE:-stage1}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-$PROBE_TAG}"
STRATIFIED="$(<"$PIPELINE_DIR/stratified_manifest.txt")"
MASTERED="$(<"$PIPELINE_DIR/mastered_manifest.txt")"
CHALLENGING="$(<"$PIPELINE_DIR/challenging_manifest.txt")"

case "$EVAL_PHASE" in
  stage1)
    CHECKPOINT="$(<"$PIPELINE_DIR/base_checkpoint.txt")"
    ;;
  stage2)
    RUN_DIR="$(find "$PROJECT_DIR/logs/rsl_rl/gmtrack_stage2" -mindepth 1 -maxdepth 1 \
      -type d -name "*_${EVAL_RUN_NAME}" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
    test -d "$RUN_DIR"
    CHECKPOINT="$(find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' \
      -printf '%f %p\n' | sort -t_ -k2,2n | tail -1 | cut -d' ' -f2-)"
    ;;
  *)
    echo "EVAL_PHASE must be stage1 or stage2, got: $EVAL_PHASE" >&2
    exit 2
    ;;
esac

cd "$PROJECT_DIR"
mkdir -p logs/slurm "$PIPELINE_DIR"
for path in "$CHECKPOINT" "$STRATIFIED" "$MASTERED" "$CHALLENGING"; do
  test -f "$path"
done

source /public/software/modules/5.4.0/init/bash
module purge
module load anaconda3/python3.8
module load cuda/11.8
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_PREFIX:?set CONDA_ENV_PREFIX to the conda environment prefix}"

export MUJOCO_GL=egl
export GMTRACK_DATA="$DATA_DIR"
export GMTRACK_STRATIFIED_MANIFEST="$STRATIFIED"
export GMTRACK_MASTERED_MANIFEST="$MASTERED"
export GMTRACK_CHALLENGING_MANIFEST="$CHALLENGING"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CHECKPOINT_ITER="$(basename "$CHECKPOINT" .pt)"
OUT="$PIPELINE_DIR/eval_${EVAL_PHASE}_${CHECKPOINT_ITER}.json"
echo "[INFO] phase=$EVAL_PHASE checkpoint=$CHECKPOINT output=$OUT"
/opt/gridview/slurm/bin/srun python -m gmtrack.scripts.evaluate \
  --checkpoint "$CHECKPOINT" \
  --training-seed "$SEED" \
  --manifest "$CHALLENGING" \
  --task GMTrack-Stage2-Flat-Unitree-G1 \
  --num-envs 256 \
  --rollouts-per-clip 1 \
  --eval-mode nominal \
  --device cuda:0 \
  --out "$OUT"
