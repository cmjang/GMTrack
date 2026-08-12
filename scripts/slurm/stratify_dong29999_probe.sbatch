#!/bin/bash
# Diagnostic handoff of <cluster-user>'s completed Flat/FSQ32 Stage-I checkpoint into
# the current 2607.20110v1 stratification implementation. This probe has its own
# pipeline and manifest directories and cannot overwrite the formal seed-42 run.
#SBATCH --job-name=exgrmt-d29999-strat
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

PROJECT_DIR="${EX_GRMT_PROJECT_DIR:-<cluster-home>/Ex-GRMT-v1-fix-20260809}"
DATA_DIR="${EX_GRMT_DATA:-<cluster-home>/Ex-GRMT-v1-fix-20260809-data}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$PROJECT_DIR/imported_checkpoints/<cluster-user>_stage1_norecovery_29999/model_29999.pt}"
SEED="${SEED:-42}"
PROBE_TAG="${PROBE_TAG:-dong29999-latest-s2-probe-seed${SEED}}"
PIPELINE_DIR="$PROJECT_DIR/logs/pipeline/$PROBE_TAG"
OUT_DIR="$PROJECT_DIR/logs/legacy_manifests/$PROBE_TAG"
SOURCE_MANIFEST="${SOURCE_MANIFEST:?Set SOURCE_MANIFEST for this retired checkpoint}"

EXPECTED_PAPER_SHA256=55cca5c02f16c659e4ab3baf08d9ad1fb69865f37cfba084958ade0911cf51fe
EXPECTED_MANIFEST_SHA256=f66013b20625307aa24ef6d60ba05450d1a0d8a658d0eda5dea86f2ca33692a8
EXPECTED_BASE_SHA256=e6e978dd124de45a90d41212b462a4b5fac351b539e9552026bd680db7369aef

cd "$PROJECT_DIR"
mkdir -p logs/slurm "$PIPELINE_DIR" "$OUT_DIR"
test -f "$BASE_CHECKPOINT"
test -f "$SOURCE_MANIFEST"
echo "$EXPECTED_PAPER_SHA256  2607.20110v1.pdf" | sha256sum -c -
echo "$EXPECTED_MANIFEST_SHA256  $SOURCE_MANIFEST" | sha256sum -c -
echo "$EXPECTED_BASE_SHA256  $BASE_CHECKPOINT" | sha256sum -c -

source /public/software/modules/5.4.0/init/bash
module purge
module load anaconda3/python3.8
module load cuda/11.8
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_PREFIX:-<cluster-home>/.conda/envs/ex-grmt}"

export MUJOCO_GL=egl
export EX_GRMT_DATA="$DATA_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

realpath "$BASE_CHECKPOINT" > "$PIPELINE_DIR/base_checkpoint.txt"
printf '%s\n' "$EXPECTED_BASE_SHA256" > "$PIPELINE_DIR/base_checkpoint.sha256"

/opt/gridview/slurm/bin/srun python -m ex_grmt.scripts.stratify \
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
