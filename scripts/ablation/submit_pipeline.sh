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
#
# Bouchet gpu_devel has a 1-GPU-job/user cap — this script serializes GPU steps automatically.
# Override partitions if needed:
#   GPU_PART=scavenge_gpu CPU_PART=day ENCODER=vjepa2_huge bash scripts/ablation/submit_pipeline.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_encoder_env.sh"

cd "$BASE_DIR"
mkdir -p logs

# Prefer scavenge_gpu for multi-step pipelines (no 1-job/user cap). gpu_devel is fine for single jobs.
GPU_PART=${GPU_PART:-$(scripts/_pick_partition.sh 0 scavenge_gpu gpu_devel)}
CPU_PART=${CPU_PART:-$(scripts/_pick_partition.sh 256000 day week bigmem)}
SERIAL_GPU=0
if [ "$GPU_PART" = "gpu_devel" ]; then SERIAL_GPU=1; fi

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
echo "[submit] GPU partition=$GPU_PART  CPU partition=$CPU_PART  SERIAL_GPU=$SERIAL_GPU"

J_TRAIN_EX=$(sbatch --parsable --partition="$GPU_PART" \
    --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",SPLIT=train,NUM_CLIPS="$TRAIN_CLIPS" \
    "$SCRIPT_DIR/extract_v2d.sh")

if [ "$SERIAL_GPU" = "1" ]; then
    # gpu_devel: one GPU job per user — chain extract test after train extract.
    J_TEST_EX=$(sbatch --parsable --partition="$GPU_PART" \
        --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",SPLIT=test,NUM_CLIPS="$TEST_CLIPS" \
        --dependency=afterok:"$J_TRAIN_EX" \
        "$SCRIPT_DIR/extract_v2d.sh")
else
    J_TEST_EX=$(sbatch --parsable --partition="$GPU_PART" \
        --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",SPLIT=test,NUM_CLIPS="$TEST_CLIPS" \
        "$SCRIPT_DIR/extract_v2d.sh")
fi

J_SUB=$(sbatch --parsable --partition="$CPU_PART" --mem=320G \
    --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR" \
    --dependency=afterok:"$J_TRAIN_EX":"$J_TEST_EX" \
    "$SCRIPT_DIR/subspace_v2d.sh")
J_CMD=$(sbatch --parsable --partition="$CPU_PART" --mem=320G \
    --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR" \
    --dependency=afterok:"$J_SUB" \
    "$SCRIPT_DIR/fit_command.sh")

if [ "$SERIAL_GPU" = "1" ]; then
    # Decoder train also waits for test extract so only one GPU job runs at a time.
    J_TRAIN=$(sbatch --parsable --partition="$GPU_PART" --mem="$TRAIN_MEM" \
        --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",MAX_STEPS="${MAX_STEPS:-8000}" \
        --dependency=afterok:"$J_TEST_EX" \
        "$SCRIPT_DIR/train_decoder.sh")
else
    J_TRAIN=$(sbatch --parsable --partition="$GPU_PART" --mem="$TRAIN_MEM" \
        --export=ALL,ENCODER="$ENCODER",BASE_DIR="$BASE_DIR",MAX_STEPS="${MAX_STEPS:-8000}" \
        --dependency=afterok:"$J_TRAIN_EX" \
        "$SCRIPT_DIR/train_decoder.sh")
fi

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
