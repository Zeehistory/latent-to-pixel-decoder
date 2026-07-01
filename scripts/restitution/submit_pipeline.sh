#!/bin/bash
# Submit the restitution (bounce) pipeline on Bouchet.
#
#   export BASE_DIR=$HOME/vjepa-latent-physics
#   cd $BASE_DIR
#   ENCODER=vjepa2_large bash scripts/restitution/submit_pipeline.sh
#
# Pilot: PILOT=1 ENCODER=vjepa2_large bash scripts/restitution/submit_pipeline.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_restitution_env.sh"
cd "$BASE_DIR"
mkdir -p logs

GPU_PART=${GPU_PART:-$(scripts/_pick_partition.sh 0 scavenge_gpu gpu_devel)}
CPU_PART=${CPU_PART:-$(scripts/_pick_partition.sh 256000 day week bigmem)}

TRAIN_CLIPS=4000
TEST_CLIPS=800
if [ "${PILOT:-0}" = "1" ]; then
  TRAIN_CLIPS=400
  TEST_CLIPS=80
  MAX_STEPS=1500
  NUM_SCENES=10
  echo "[rest_submit] PILOT: $TRAIN_CLIPS train / $TEST_CLIPS test clips"
fi

echo "[rest_submit] ENCODER=$ENCODER GPU=$GPU_PART CPU=$CPU_PART"

J_TRAIN_EX=$(sbatch --parsable --partition="$GPU_PART" \
  --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",SPLIT=train,NUM_CLIPS="$TRAIN_CLIPS" \
  "$SCRIPT_DIR/extract.sh")
J_TEST_EX=$(sbatch --parsable --partition="$GPU_PART" \
  --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",SPLIT=test,NUM_CLIPS="$TEST_CLIPS" \
  --dependency=afterok:"$J_TRAIN_EX" \
  "$SCRIPT_DIR/extract.sh")

J_SUB=$(sbatch --parsable --partition="$CPU_PART" --mem=320G \
  --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR" \
  --dependency=afterok:"$J_TRAIN_EX":"$J_TEST_EX" \
  "$SCRIPT_DIR/subspace.sh")
J_CMD=$(sbatch --parsable --partition="$CPU_PART" --mem=320G \
  --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR" \
  --dependency=afterok:"$J_SUB" \
  "$SCRIPT_DIR/fit_command.sh")

J_TRAIN=$(sbatch --parsable --partition="$GPU_PART" --mem="$TRAIN_MEM" \
  --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",MAX_STEPS="${MAX_STEPS:-8000}" \
  --dependency=afterok:"$J_TRAIN_EX" \
  "$SCRIPT_DIR/train_decoder.sh")

J_STEER=$(sbatch --parsable --partition="$GPU_PART" \
  --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",NUM_SCENES="${NUM_SCENES:-40}" \
  --dependency=afterok:"$J_SUB":"$J_CMD":"$J_TRAIN" \
  "$SCRIPT_DIR/steer.sh")

cat <<EOF

Submitted restitution pipeline for ENCODER=$ENCODER
  extract train : $J_TRAIN_EX
  extract test  : $J_TEST_EX
  subspace      : $J_SUB
  fit command   : $J_CMD
  train decoder : $J_TRAIN
  steer         : $J_STEER

Results: $STEER_DIR/steer_restitution_summary.json
         $SUBSPACE_DIR/subspace_summary.json
         $STEER_DIR/cmd_gain_calibration.json

EOF
