#!/bin/bash
# 2607.20110v1 Sec. IV-C: <=10 s clips, five randomized rollouts, >=80% mastered.
#SBATCH --job-name=gmtrack-v1p-strat
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
#SBATCH --time=1-00:00:00

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
PIPELINE_DIR="$PROJECT_DIR/logs/pipeline/v1-final-backflip-seed${SEED}"
DATA_DIR="${GMTRACK_DATA:-$PROJECT_DIR/data}"
OUT_DIR="$PROJECT_DIR/logs/legacy_manifests/v1-final-backflip-seed${SEED}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$(<"$PIPELINE_DIR/base_checkpoint.txt")}"
SOURCE_MANIFEST="$DATA_DIR/current/stage1_paper_mix_final_backflip_cartwheel_balanced_grounded_no_fall_getup.json"

mkdir -p "$PROJECT_DIR/logs/slurm" "$OUT_DIR"
test -f "$BASE_CHECKPOINT"
export MUJOCO_GL=egl
export GMTRACK_DATA="$DATA_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

/opt/gridview/slurm/bin/srun python -m gmtrack.scripts.stratify \
  --checkpoint "$BASE_CHECKPOINT" \
  --manifest "$SOURCE_MANIFEST" \
  --stratified-manifest "$OUT_DIR/stratified.json" \
  --out-dir "$OUT_DIR" \
  --clip-seconds 10 \
  --trials 5 \
  --threshold 0.8 \
  --num-envs 256 \
  --device cuda:0 \
  --seed "$SEED" \
  --dataset-label v1-proxy

printf '%s\n' "$OUT_DIR/stratified.json" > "$PIPELINE_DIR/stratified_manifest.txt"
printf '%s\n' "$OUT_DIR/mastered.json" > "$PIPELINE_DIR/mastered_manifest.txt"
printf '%s\n' "$OUT_DIR/challenging.json" > "$PIPELINE_DIR/challenging_manifest.txt"
