#!/bin/bash
#SBATCH --job-name=rest_extract
#SBATCH --partition=scavenge_gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rest_extract_%j.out
#SBATCH --error=logs/rest_extract_%j.err

source "${SLURM_SUBMIT_DIR}/scripts/restitution/_job_init.sh"

SPLIT=${SPLIT:-train}
case "$SPLIT" in
  train) NUM_CLIPS=${NUM_CLIPS:-4000}; SEED=${SEED:-0} ;;
  test)  NUM_CLIPS=${NUM_CLIPS:-800};  SEED=${SEED:-2} ;;
  *) echo "unknown SPLIT=$SPLIT"; exit 1 ;;
esac

OUTPUT_DIR="$LATENT_ROOT/$SPLIT/$ENCODER"
echo "[rest_extract] ENCODER=$ENCODER SPLIT=$SPLIT -> $OUTPUT_DIR"
python scripts/extract_latents.py \
  --config "$TRAIN_CONFIG" \
  --encoder "$ENCODER" \
  --layers "$ENCODER_LAYERS" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$EXTRACT_BATCH" \
  --shard_size 128 \
  data.scenario=scene_restitution \
  data.clips_per_scene=8 \
  data.num_clips="$NUM_CLIPS" \
  data.seed="$SEED" \
  "data.speed_range=[0.018,0.032]" \
  "data.restitution_range=[0.35,0.95]" \
  "data.radius_range=[0.11,0.11]"
