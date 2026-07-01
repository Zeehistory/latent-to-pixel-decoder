#!/bin/bash
# Submit the full ViT-L / ViT-H ablation pipeline on Bouchet with SLURM dependencies.
#
#   export BASE_DIR=$HOME/vjepa-latent-physics
#   cd $BASE_DIR
#
#   # ViT-H ablation (Andy's ask):
#   ENCODER=vjepa2_huge bash scripts/ablation/submit_pipeline.sh
#
#   # ViT-L baseline for side-by-side comparison:
#   ENCODER=vjepa2_large bash scripts/ablation/submit_pipeline.sh
#
# Optional fast smoke:  PILOT=1 ENCODER=vjepa2_huge bash scripts/ablation/submit_pipeline.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_encoder_env.sh"

cd "$BASE_DIR"
mkdir -p logs

GPU_PART=$(scripts/_pick_partition.sh 0 gpu_devel scavenge_gpu)
CPU_PART=$(scripts/_pick_partition.sh 256000 bigmem day week)

TRAIN_CLIPS=4000
TEST_CLIPS=800
if [ "${PILOT:-0}" = "1" ]; then
    TRAIN_CLIPS=400
    TEST_CLIPS=80
    MAX_STEPS=1500
    NUM_SCENES=10
    echo "[submit] PILOT mode: $TRAIN_CLIPS train / $TEST_CLIPS test clips, $MAX_STEPS steps"
fi

echo "[submit] ENCODER=$ENCODER BASE_DIR=$BASE_DIR"
echo "[submit] GPU partition=$GPU_PART  CPU partition=$CPU_PART"

J_TRAIN_EX=$(sbatch --parsable --partition="$GPU_PART" \
    --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",SPLIT=train,NUM_CLIPS="$TRAIN_CLIPS" \
    "$SCRIPT_DIR/extract_v2d.sh")
J_TEST_EX=$(sbatch --parsable --partition="$GPU_PART" \
    --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",SPLIT=test,NUM_CLIPS="$TEST_CLIPS" \
    "$SCRIPT_DIR/extract_v2d.sh")

J_SUB=$(sbatch --parsable --partition="$CPU_PART" --mem=256G \
    --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR" \
    --dependency=afterok:"$J_TRAIN_EX":"$J_TEST_EX" \
    "$SCRIPT_DIR/subspace_v2d.sh")
J_CMD=$(sbatch --parsable --partition="$CPU_PART" --mem=256G \
    --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR" \
    --dependency=afterok:"$J_SUB" \
    "$SCRIPT_DIR/fit_command.sh")
J_TRAIN=$(sbatch --parsable --partition="$GPU_PART" --mem=192G \
    --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",MAX_STEPS="${MAX_STEPS:-8000}" \
    --dependency=afterok:"$J_TRAIN_EX" \
    "$SCRIPT_DIR/train_decoder.sh")
J_STEER=$(sbatch --parsable --partition="$GPU_PART" \
    --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",NUM_SCENES="${NUM_SCENES:-40}" \
    --dependency=afterok:"$J_SUB":"$J_CMD":"$J_TRAIN" \
    "$SCRIPT_DIR/steer_v2d.sh")

cat <<EOF

Submitted ablation pipeline for ENCODER=$ENCODER
  extract train : $J_TRAIN_EX
  extract test  : $J_TEST_EX
  subspace      : $J_SUB   (after extract)
  fit command   : $J_CMD   (after subspace)
  train decoder : $J_TRAIN (after train extract)
  steer+decode  : $J_STEER (after subspace+cmd+train)

Watch:  squeue -u \$USER
Results: $STEER_DIR/steer2d_summary.json
         $SUBSPACE_DIR/subspace_summary.json
         $STEER_DIR/cmd_gain_calibration.json

Compare ViT-L vs ViT-H in steer2d_summary.json -> results:
  full_delta.angle_err_deg    (ceiling; ~1-2 deg on L)
  subspace_U8.angle_err_deg   (global subspace; ~21 deg on L)
  ridge_global.angle_err_deg  (linear transfer; ~34 deg on L)
  cmd_U8_s2.angle_err_deg     (command-only; ~6-7 deg on L)

EOF
