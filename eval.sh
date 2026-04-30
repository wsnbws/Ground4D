#!/bin/bash
# Ground4D Evaluation Script
# Example: bash eval.sh

# ─── User configuration ───────────────────────────────────────
IMAGE_DIR="/path/to/orad/training"          # Dataset root
CKPT_PATH="logs/ground4d/ckpt/model_latest.pt"  # Fine-tuned checkpoint
DGGT_CKPT_PATH="/path/to/pretrained/dggt.pt"    # Pretrained backbone
TRACK_CKPT="/path/to/pretrained/tapip3d_final.pth"  # TAPIP3D weights
OUTPUT_DIR="results/ground4d_eval"
DATASET_TYPE="orad"   # "orad" or "rellis"
INTERVAL=20
NUM_INTER_FRAMES=3
MODE=3                # 2=static, 3=dynamic+interpolation
VOXEL_SIZE=0.002
CUDA_ID=0
# ──────────────────────────────────────────────────────────────

TORCH_CUDA_ARCH_LIST="8.9" \
CUDA_VISIBLE_DEVICES=$CUDA_ID \
python custom_inference.py \
  --image_dir       $IMAGE_DIR            \
  --dataset_type    $DATASET_TYPE         \
  --ckpt_path       $CKPT_PATH            \
  --dggt_ckpt_path  $DGGT_CKPT_PATH       \
  --model_type      voxel_v2              \
  --sequence_length 4                     \
  --output_dir      $OUTPUT_DIR           \
  --interval        $INTERVAL             \
  --n_inter_frames  $NUM_INTER_FRAMES     \
  --track_ckpt      $TRACK_CKPT           \
  --mode            $MODE                 \
  --voxel_size      $VOXEL_SIZE           \
  --fusion_version  v1                    \
  --save_images
