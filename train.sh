#!/bin/bash
# Ground4D Training Script
# Example: bash train.sh

# ─── User configuration ───────────────────────────────────────
IMAGE_DIR="/path/to/orad/training"          # Path to ORAD-3D training set
CKPT_PATH="/path/to/pretrained/dggt.pt"    # Pretrained VGGT/DGGT backbone
LOG_DIR="logs/ground4d"                     # Output directory for logs & checkpoints
# ──────────────────────────────────────────────────────────────

TORCH_CUDA_ARCH_LIST="8.9" \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=12386 ground4d_train.py \
  --log_dir          $LOG_DIR              \
  --ckpt_path        $CKPT_PATH            \
  --image_dir        $IMAGE_DIR            \
  --voxel_size       0.002                 \
  --feature_dim      64                    \
  --hidden_dim       64                    \
  --max_epoch        2000                  \
  --save_image       100                   \
  --save_ckpt        100                   \
  --sequence_length  4                     \
  --drop_view_num    1                     \
  --drop_middle_view_prob 0.5              \
  --fusion_version   v1                    \
  --auto_resume

# ── Optional: enable surface-normal supervision ────────────────
# Add the following flags to the command above to activate the
# normal-regularization branch described in Section 3.2:
#
#   --use_normal_supervision                \
#   --normal_key          normals           \
#   --normal_pred_weight  0.05              \
#   --normal_gs_weight    0.02              \
#   --normal_loss_warmup_steps 1000        \
#   --normal_static_only_warmup_epochs 100 \
#   --normal_cond_scale   0.07             \
#   --normal_softmin_temp 10.0             \
#   --gs_anisotropy_weight 0.05            \
#   --gs_anisotropy_target_ratio 0.3       \
#   --quat_format xyzw                     \
#   --normal_static_only
