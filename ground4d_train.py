"""
Voxel-based Training V2: Temporal-aware Gaussian Fusion

This version implements a more sophisticated approach where:
1. Gaussians are grouped into voxels spatially
2. At render time, Gaussians within each voxel are dynamically fused based on temporal proximity
3. Each voxel produces a single representative Gaussian for rendering
4. This reduces rendering cost while maintaining temporal consistency
"""

import os
import argparse
import random
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import lpips
import logging
import datetime

from ground4d.models.vggt import VGGT
from ground4d.utils.pose_enc import pose_encoding_to_extri_intri
from ground4d.utils.geometry import unproject_depth_map_to_point_map
from ground4d.utils.gs import concat_list, get_split_gs
from ground4d.normal_modules import (
    fetch_normal_gt,
    fetch_normal_valid_mask,
    compute_pred_normal_loss,
    compute_gs_normal_loss_against_gt,
    compute_gs_anisotropy_loss,
)
from gsplat.rendering import rasterization
from datasets.orad_dataset import OradDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import new voxelize_v2 modules
from ground4d.voxelize_v2 import GaussianVoxelizerV2, TemporalVoxelFusion, TemporalVoxelFusionV3


def setup_logger(log_dir):
    """Set up logger to save logs to file and print to console"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger("VoxelTrain")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def create_optimizer(model, fusion_module, args):
    """Create optimizer with different learning rates for different components"""
    param_groups = [
        {'params': model.module.gs_head.parameters(), 'lr': 4e-5, 'name': 'gs_head'},
        {'params': model.module.instance_head.parameters(), 'lr': 4e-5, 'name': 'instance_head'},
        {'params': model.module.sky_model.parameters(), 'lr': 1e-4, 'name': 'sky_model'},
    ]
    if getattr(model.module, 'normal_head', None) is not None:
        param_groups.append({'params': model.module.normal_head.parameters(), 'lr': 1e-4, 'name': 'normal_head'})
    if fusion_module is not None:
        fusion_params = [p for p in fusion_module.parameters() if p.requires_grad]
        if len(fusion_params) > 0:
            param_groups.append({'params': fusion_params, 'lr': 1e-4, 'name': 'fusion_module'})
    optimizer = AdamW(param_groups, weight_decay=1e-4)
    return optimizer

def create_scheduler(optimizer, dataloader, args):
    """Create learning rate scheduler with warmup and cosine annealing"""
    warmup_iterations = 1000
    steps_per_epoch = len(dataloader)
    total_steps = args.max_epoch * steps_per_epoch
    
    def get_lr_lambda(current_step):
        if current_step < warmup_iterations:
            return float(current_step) / float(max(1, warmup_iterations))
        else:
            progress = (current_step - warmup_iterations) / max(1, total_steps - warmup_iterations)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda=get_lr_lambda)

class CheckpointManager:
    """Manage pretrained init and trainable-only resume checkpoints."""

    def __init__(self, log_dir, local_rank):
        self.log_dir = log_dir
        self.local_rank = local_rank

    @staticmethod
    def _unwrap(module):
        return module.module if hasattr(module, "module") else module

    @staticmethod
    def _extract_model_payload(ckpt_obj):
        if isinstance(ckpt_obj, dict):
            for key in ("model", "state_dict"):
                if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                    return ckpt_obj[key]
        return ckpt_obj if isinstance(ckpt_obj, dict) else None

    @staticmethod
    def _filter_compatible_state_dict(model_state, candidate_state):
        """Keep only keys that exist and have identical tensor shapes."""
        filtered = {}
        skipped = []
        for key, value in candidate_state.items():
            if key in model_state and model_state[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(key)
        return filtered, skipped

    @staticmethod
    def _merge_with_partial_channel_copy(model_state, candidate_state):
        """
        Merge checkpoint into current state:
        - Exact shape match: fully load.
        - Same-rank tensors with matching non-channel dims: copy overlapping out-channels.
        """
        merged = {}
        skipped = []
        partial = []

        for key, src in candidate_state.items():
            if key not in model_state:
                skipped.append(key)
                continue

            dst = model_state[key]
            if dst.shape == src.shape:
                merged[key] = src
                continue

            # Partial copy for shape-changed heads (e.g. output channel expansion).
            if dst.dim() == src.dim() and dst.dim() >= 1:
                # Match all dims except dim 0 (out channel / bias length).
                same_tail = tuple(dst.shape[1:]) == tuple(src.shape[1:])
                if same_tail:
                    new_tensor = dst.clone()
                    copy_n = min(dst.shape[0], src.shape[0])
                    new_tensor[:copy_n] = src[:copy_n].to(new_tensor.dtype)
                    merged[key] = new_tensor
                    partial.append((key, copy_n, int(dst.shape[0]), int(src.shape[0])))
                    continue

            skipped.append(key)

        return merged, skipped, partial

    def load_pretrained(self, model, ckpt_path, skip_gs_sky_for_train=False):
        """
        skip_gs_sky_for_train: if True (e.g. when use_normal_supervision), do not load
            gs_head and sky_model from checkpoint so they are trained from scratch.
        """
        model_raw = self._unwrap(model)
        ckpt_obj = torch.load(ckpt_path, map_location="cpu")
        payload = self._extract_model_payload(ckpt_obj)
        if payload is None:
            raise ValueError(f"Unsupported pretrained checkpoint format: {ckpt_path}")
        filtered, skipped = self._filter_compatible_state_dict(model_raw.state_dict(), payload)
        if skip_gs_sky_for_train:
            filtered = {k: v for k, v in filtered.items()
                        if not (k.startswith("gs_head.") or k.startswith("sky_model."))}
        model_raw.load_state_dict(filtered, strict=False)
        if self.local_rank == 0:
            logger = logging.getLogger("VoxelTrain")
            logger.info(f"Loaded pretrained weights: {ckpt_path}" +
                        (" (gs_head and sky_model not loaded, trained from scratch)" if skip_gs_sky_for_train else ""))
            if skipped:
                logger.info(f"Skipped {len(skipped)} incompatible pretrained keys.")

    def save_trainable(self, model, fusion_module, optimizer, scheduler, epoch, global_step):
        model_raw = self._unwrap(model)
        fusion_raw = self._unwrap(fusion_module) if fusion_module is not None else None
        ckpt_path = os.path.join(self.log_dir, "ckpt", "model_latest.pt")
        trainable_dict = {
            "gs_head": model_raw.gs_head.state_dict(),
            "instance_head": model_raw.instance_head.state_dict(),
            "sky_model": model_raw.sky_model.state_dict(),
            "fusion_module": fusion_raw.state_dict() if fusion_raw is not None else None,
        }
        if getattr(model_raw, "normal_head", None) is not None:
            trainable_dict["normal_head"] = model_raw.normal_head.state_dict()
        torch.save(
            {
                "trainable": trainable_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            ckpt_path,
        )
        return ckpt_path

    def resolve_resume_path(self, resume_ckpt, auto_resume):
        resume_path = resume_ckpt.strip()
        if auto_resume and not resume_path:
            candidate = os.path.join(self.log_dir, "ckpt", "model_latest.pt")
            if os.path.exists(candidate):
                resume_path = candidate
        return resume_path

    def load_resume(self, model, fusion_module, optimizer, scheduler, resume_path):
        if not resume_path:
            return 0, 0
        if not os.path.exists(resume_path):
            if self.local_rank == 0:
                logging.getLogger("VoxelTrain").warning(f"Resume checkpoint not found: {resume_path}")
            return 0, 0

        model_raw = self._unwrap(model)
        fusion_raw = self._unwrap(fusion_module) if fusion_module is not None else None
        ckpt = torch.load(resume_path, map_location="cpu")

        if "trainable" in ckpt:
            trainable = ckpt["trainable"]
            if "gs_head" in trainable:
                model_raw.gs_head.load_state_dict(trainable["gs_head"], strict=False)
            if "instance_head" in trainable:
                model_raw.instance_head.load_state_dict(trainable["instance_head"], strict=False)
            if "sky_model" in trainable:
                model_raw.sky_model.load_state_dict(trainable["sky_model"], strict=False)
            if "normal_head" in trainable and getattr(model_raw, "normal_head", None) is not None:
                model_raw.normal_head.load_state_dict(trainable["normal_head"], strict=False)
            if "fusion_module" in trainable and trainable["fusion_module"] is not None and fusion_raw is not None:
                fusion_raw.load_state_dict(trainable["fusion_module"], strict=False)
        else:
            # Backward compatibility with full-model checkpoints.
            if "model" in ckpt:
                merged, skipped, partial = self._merge_with_partial_channel_copy(
                    model_raw.state_dict(), ckpt["model"]
                )
                model_raw.load_state_dict(merged, strict=False)
                if self.local_rank == 0:
                    logger = logging.getLogger("VoxelTrain")
                    if partial:
                        logger.info(f"Applied partial channel copy for {len(partial)} keys from old resume model.")
                    if skipped:
                        logger.info(f"Skipped {len(skipped)} incompatible keys from old resume model.")
            if "fusion_module" in ckpt and ckpt["fusion_module"] is not None and fusion_raw is not None:
                fusion_raw.load_state_dict(ckpt["fusion_module"], strict=False)

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        if self.local_rank == 0:
            logger = logging.getLogger("VoxelTrain")
            logger.info(f"Resumed from: {resume_path}")
            logger.info(f"Start epoch: {start_epoch}, global_step: {global_step}")
        return start_epoch, global_step

def extract_gaussian_features(point_map, gs_map, gs_conf, dy_map, timestamps, bg_mask, drop_frame_indices=None):
    """
    Extract and concatenate all Gaussian features into a single tensor.
    
    Returns:
        gaussian_features: [N, 15] tensor with all Gaussian attributes
        static_points: [N, 3] positions
        frame_idx: [N] frame index for each Gaussian
        static_gs_conf: [N] GS confidence/lifespan parameter aligned with gaussian_features
    """
    N_frames = point_map.shape[1]
    device = point_map.device
    
    static_mask = torch.ones_like(bg_mask)
    if drop_frame_indices is not None:
        for drop_idx in drop_frame_indices:
            if 0 <= drop_idx < N_frames:
                static_mask[:, drop_idx] = False
    static_points = point_map[static_mask].reshape(-1, 3)
    static_gs_conf = gs_conf[static_mask].reshape(-1)
    
    # Extract Gaussian attributes
    gs_dynamic_list = dy_map[static_mask].sigmoid()
    static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
    static_opacity = static_opacity * (1 - gs_dynamic_list)
    
    # Frame indices and timestamps
    frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
    gs_timestamps = timestamps[frame_idx].float()
    
    # Concatenate all features: [pos(3), rgb(3), opa(1), scale(3), rot(4), time(1)]
    gaussian_features = torch.cat([
        static_points,                                                          # [N, 3]
        static_rgbs,                                                            # [N, 3]
        static_opacity if static_opacity.dim() == 2 else static_opacity.unsqueeze(-1),  # [N, 1]
        static_scales,                                                          # [N, 3]
        static_rotations,                                                       # [N, 4]
        gs_timestamps.unsqueeze(-1) if gs_timestamps.dim() == 1 else gs_timestamps      # [N, 1]
    ], dim=-1)
    
    return gaussian_features, static_points, frame_idx, static_gs_conf

def unpack_gaussian_features(fused_gaussians):
    """
    Unpack fused Gaussian features into individual components.
    
    Args:
        fused_gaussians: [M, 15] fused Gaussian features
    
    Returns:
        Tuple of (points, rgbs, opacity, scales, rotations)
    """
    points = fused_gaussians[:, 0:3]
    rgbs = fused_gaussians[:, 3:6]
    opacity = fused_gaussians[:, 6:7].squeeze(-1)  # [M] to match dynamic_opacity
    scales = fused_gaussians[:, 7:10]
    rotations = fused_gaussians[:, 10:14]
    
    return points, rgbs, opacity, scales, rotations

def render_frame(fused_static_gs, dynamic_gs, extrinsic, intrinsic, H, W):
    """
    Render a single frame with fused static Gaussians and dynamic Gaussians.
    
    Args:
        fused_static_gs: Tuple of (points, rgbs, opacity, scales, rotations) for static
        dynamic_gs: Tuple of (points, rgbs, opacity, scales, rotations) for dynamic (or None)
        extrinsic, intrinsic: Camera parameters
        H, W: Image dimensions
    
    Returns:
        Rendered image and alpha channel
    """
    if dynamic_gs is not None:
        # Concatenate static and dynamic Gaussians
        world_points, rgbs, opacity, scales, rotation = concat_list(
            list(fused_static_gs),
            list(dynamic_gs)
        )
    else:
        world_points, rgbs, opacity, scales, rotation = fused_static_gs
    
    # Rasterize
    renders, alphas, _ = rasterization(
        means=world_points,
        quats=rotation,
        scales=scales,
        opacities=opacity,
        colors=rgbs,
        viewmats=extrinsic[None],
        Ks=intrinsic[None],
        width=W,
        height=H,
    )
    
    return renders, alphas

def compute_losses(rendered_image, target_image, alphas, sky_mask, dy_map,
                   dynamic_masks, lpips_loss_fn, global_step, normal_pred_loss=None,
                   normal_pred_weight=0.0, normal_gs_loss=None, normal_gs_weight=0.0,
                   gs_anisotropy_loss=None, gs_anisotropy_weight=0.0):
    """
    Compute all training losses.
    
    Returns:
        Total loss and individual loss components
    """
    # L1 reconstruction loss
    l1_loss = F.l1_loss(rendered_image, target_image)
    
    # Sky mask loss
    sky_mask_loss = F.l1_loss(alphas, 1 - sky_mask[..., 0][..., None])
    
    # Dynamic mask loss
    if dynamic_masks is not None:
        binary_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        dynamic_loss = binary_loss_fn(dy_map, dynamic_masks.float())
    else:
        binary_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        dynamic_loss = binary_loss_fn(dy_map, torch.zeros_like(dy_map))
    
    # LPIPS perceptual loss (gradually increase weight)
    lpips_val = lpips_loss_fn(rendered_image, target_image)
    lpips_weight = 0.05 * min(global_step / 1000, 1.0)
    lpips_loss = lpips_weight * lpips_val.mean()
    
    # Total loss
    total_loss = l1_loss + sky_mask_loss + 0.5 * dynamic_loss + lpips_loss
    normal_pred_item = 0.0
    if normal_pred_loss is not None and normal_pred_weight > 0:
        total_loss = total_loss + normal_pred_weight * normal_pred_loss
        normal_pred_item = normal_pred_loss.item()
    normal_gs_item = 0.0
    if normal_gs_loss is not None and normal_gs_weight > 0:
        total_loss = total_loss + normal_gs_weight * normal_gs_loss
        normal_gs_item = normal_gs_loss.item()
    gs_anisotropy_item = 0.0
    if gs_anisotropy_loss is not None and gs_anisotropy_weight > 0:
        total_loss = total_loss + gs_anisotropy_weight * gs_anisotropy_loss
        gs_anisotropy_item = gs_anisotropy_loss.item()
    return total_loss, {
        'l1': l1_loss.item(),
        'sky_mask': sky_mask_loss.item(),
        'dynamic': dynamic_loss.item(),
        'lpips': lpips_loss.item(),
        'normal_pred': normal_pred_item,
        'normal_gs': normal_gs_item,
        'gs_anisotropy': gs_anisotropy_item,
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Voxel-based training with temporal fusion')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to training images')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to pretrained model checkpoint')
    parser.add_argument('--log_dir', type=str, default='logs/voxel_exp_v2', help='Directory to save logs')
    parser.add_argument('--sequence_length', type=int, default=4, help='Number of frames per sequence')
    parser.add_argument('--max_epoch', type=int, default=50000, help='Maximum training epochs')
    parser.add_argument('--save_image', type=int, default=100, help='Save visualization every N epochs')
    parser.add_argument('--save_ckpt', type=int, default=100, help='Save checkpoint every N epochs')
    parser.add_argument('--resume_ckpt', type=str, default='',
                        help='Checkpoint path to resume training from')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Auto resume from <log_dir>/ckpt/model_latest.pt if exists')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (should be 1 for now)')
    parser.add_argument('--voxel_size', type=float, default=0.02, help='Voxel size in world coordinates')
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension for fusion module')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for time encoding')
    parser.add_argument('--fusion_version', type=str, default='v1', choices=['v1', 'v3'],
                        help='Temporal voxel fusion version')
    parser.add_argument('--fusion_gamma1', type=float, default=0.1,
                        help='gamma1 used by alpha_t-style temporal prior in fusion v3')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    # 法向量参数
    parser.add_argument('--use_normal_supervision', action='store_true',
                        help='Enable normal supervision loss for ablation')
    parser.add_argument('--normal_pred_weight', type=float, default=0.1,
                        help='Weight for predicted-normal vs GT supervision')
    parser.add_argument('--normal_gs_weight', type=float, default=0.0,
                        help='Weight for GS-derived-normal vs GT supervision')
    parser.add_argument('--normal_loss_warmup_steps', type=int, default=2000,
                        help='Linearly ramp normal supervision weights in early steps (<=0 to disable)')
    parser.add_argument('--normal_key', type=str, default='normals', help='Batch key for normal ground truth')
    parser.add_argument('--normal_static_only', action='store_true', help='Apply normal loss only on static background')
    parser.add_argument('--normal_static_only_warmup_epochs', type=int, default=50,
                        help='Delay static-only masking to avoid noisy dynamic masks at early epochs')
    parser.add_argument('--normal_softmin_temp', type=float, default=10.0,
                        help='Softmin temperature for GS-derived normal consistency')
    parser.add_argument('--quat_format', type=str, default='xyzw', choices=['xyzw', 'wxyz'],
                        help='Quaternion format used in gs_map rotations')
    parser.add_argument('--normal_cond_scale', type=float, default=0.05,
                        help='Condition scale for injecting normal features into gs_head')
    parser.add_argument('--normal_skip_gs_sky_pretrained', action='store_true',
                        help='When enabled, do not load pretrained gs_head/sky_model (usually not recommended)')
    parser.add_argument('--gs_anisotropy_weight', type=float, default=0.0,
                        help='Weight for GS anisotropy loss (encourage flat Gaussians for better normal estimation)')
    parser.add_argument('--gs_anisotropy_target_ratio', type=float, default=0.3,
                        help='Target ratio s_min/mean(s_mid,s_max) for anisotropy loss')
    parser.add_argument('--drop_middle_view_prob', type=float, default=1.0,
                        help='Probability to drop middle views before voxel fusion during training')
    parser.add_argument('--drop_view_num', type=int, default=1,
                        help='Number of middle views to drop')
    
    # 消融实验参数
    parser.add_argument('--ab_use_voxelize', type=int, default=1, choices=[0, 1],
                        help='Ablation switch: 1 to use voxelization, 0 to disable voxel module')
    parser.add_argument('--ab_use_ta', type=int, default=1, choices=[0, 1],
                        help='Ablation switch: 1 to use temporal attention in voxel fusion')
    parser.add_argument('--ab_use_in', type=int, default=1, choices=[0, 1],
                        help='Ablation switch: 1 to use voxel-wise normalization (IN) after TA')
    return parser.parse_args()

def main(args):
    use_voxelize = bool(args.ab_use_voxelize)
    use_ta = bool(args.ab_use_ta)
    use_in = bool(args.ab_use_in)
    if not use_voxelize and (use_ta or use_in):
        raise ValueError("Invalid ablation config: ab_use_ta/ab_use_in require ab_use_voxelize=1.")

    # Setup
    local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    dtype = torch.float32
    
    # Create directories
    if local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "ckpt"), exist_ok=True)
        logger = setup_logger(args.log_dir)
        logger.info(f"Logging to: {args.log_dir}")
        logger.info(f"Voxel size: {args.voxel_size}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Feature dim: {args.feature_dim}")
        logger.info(f"Hidden dim: {args.hidden_dim}")
        logger.info(f"Fusion version: {args.fusion_version}")
        logger.info(f"Fusion gamma1: {args.fusion_gamma1}")
        logger.info(f"Use normal supervision: {args.use_normal_supervision}")
        logger.info(f"Normal pred weight: {args.normal_pred_weight}")
        logger.info(f"Normal GS weight: {args.normal_gs_weight}")
        logger.info(f"Normal loss warmup steps: {args.normal_loss_warmup_steps}")
        logger.info(f"Normal key: {args.normal_key}")
        logger.info(f"Normal static-only warmup epochs: {args.normal_static_only_warmup_epochs}")
        logger.info(f"Normal condition scale: {args.normal_cond_scale}")
        logger.info(f"Skip pretrained gs/sky: {args.normal_skip_gs_sky_pretrained}")
        logger.info(f"Normal softmin temp: {args.normal_softmin_temp}")
        logger.info(f"GS anisotropy weight: {args.gs_anisotropy_weight}")
        logger.info(f"GS anisotropy target ratio: {args.gs_anisotropy_target_ratio}")
        logger.info(f"Quaternion format: {args.quat_format}")
        logger.info(f"Ablation - use_voxelize: {use_voxelize}")
        logger.info(f"Ablation - use_ta: {use_ta}")
        logger.info(f"Ablation - use_in: {use_in}")
    else:
        logger = None
    
    # Load dataset
    dataset = OradDataset(
        args.image_dir,
        sequence_length=args.sequence_length,
        use_normals=args.use_normal_supervision
    )
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    
    # Load pretrained model
    model = VGGT(
        inject_normal_to_gs=args.use_normal_supervision,
        normal_cond_scale=args.normal_cond_scale,
    ).to(device)
    ckpt_manager = CheckpointManager(args.log_dir, local_rank)
    ckpt_manager.load_pretrained(
        model,
        args.ckpt_path,
        skip_gs_sky_for_train=args.normal_skip_gs_sky_pretrained,
    )
    model.train()
    
    # Initialize voxelizer and fusion module (optional for ablation)
    voxelizer = GaussianVoxelizerV2(voxel_size=args.voxel_size) if use_voxelize else None
    fusion_module = None
    if use_voxelize:
        fusion_cls = TemporalVoxelFusionV3 if args.fusion_version == 'v3' else TemporalVoxelFusion
        if args.fusion_version == 'v3':
            fusion_module = fusion_cls(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                gamma1=args.fusion_gamma1,
            ).to(device)
        else:
            fusion_module = fusion_cls(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                use_temporal_attention=use_ta,
                use_voxel_norm=use_in,
            ).to(device)

    # Freeze backbone, only train heads
    if local_rank == 0:
        logger.info("\n[Model Configuration] Frozen backbone, training the following heads:")
    
    for param in model.parameters():
        param.requires_grad = False
        
    trainable_params = []
    head_names = ["gs_head", "instance_head", "sky_model"]
    if getattr(model, "normal_head", None) is not None:
        head_names.append("normal_head")
    for head_name in head_names:
        head = getattr(model, head_name)
        for name, param in head.named_parameters():
            param.requires_grad = True
            trainable_params.append(f"{head_name}.{name}")
        if local_rank == 0:
            logger.info(f"  - {head_name}")
            
    if local_rank == 0:
        if fusion_module is not None:
            logger.info(f"  - fusion_module (TemporalVoxelFusion)")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fusion_module is not None:
            total_params += sum(p.numel() for p in fusion_module.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}\n")
    
    model = DDP(model, device_ids=[local_rank])
    model._set_static_graph()
    
    if fusion_module is not None:
        fusion_has_trainable = any(p.requires_grad for p in fusion_module.parameters())
        if fusion_has_trainable:
            fusion_module = DDP(fusion_module, device_ids=[local_rank])
            fusion_module._set_static_graph()
        elif local_rank == 0:
            logger.info("Fusion module has no trainable parameters, skip DDP wrapping.")
    
    # Loss functions
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # Optimizer and scheduler
    optimizer = create_optimizer(model, fusion_module, args)
    scheduler = create_scheduler(optimizer, dataloader, args)

    # Optional resume from trainable-only checkpoint (overrides pretrained on trainable parts).
    resume_path = ckpt_manager.resolve_resume_path(args.resume_ckpt, args.auto_resume)
    start_epoch, global_step = ckpt_manager.load_resume(
        model=model,
        fusion_module=fusion_module,
        optimizer=optimizer,
        scheduler=scheduler,
        resume_path=resume_path,
    )
    
    # Training loop
    if local_rank == 0:
        logger.info("Starting training...")
    
    for epoch in tqdm(range(start_epoch, args.max_epoch), disable=(local_rank != 0)):
        sampler.set_epoch(epoch)
        
        for batch in dataloader:
            # Load batch data
            images = batch['images'].to(device)
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch['timestamps'][0].to(device)
            dynamic_masks = batch.get('dynamic_mask', None)
            if dynamic_masks is not None:
                dynamic_masks = dynamic_masks.to(device)[:, :, 0, :, :]
            normal_gt = fetch_normal_gt(batch, args.normal_key, device) if args.use_normal_supervision else None
            normal_valid_mask = fetch_normal_valid_mask(batch, device) if args.use_normal_supervision else None
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(dtype=dtype):
                # Forward pass through VGGT
                predictions = model(images)
                H, W = images.shape[-2:]
                
                # Extract camera parameters
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]
                
                # Unproject depth to 3D points
                depth_map = predictions["depth"][0]
                point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None, ...]
                point_map = torch.from_numpy(point_map).to(device).float()
                
                # Extract Gaussian parameters
                gs_map = predictions["gs_map"]
                gs_conf = predictions["gs_conf"]
                gs_normal_map = predictions.get("gs_normal_map", None)
                dy_map = predictions["dynamic_conf"].squeeze(-1)
                
                # Prepare static Gaussian features
                drop_frame_indices = None
                num_views = images.shape[1]
                if (
                    num_views >= 3
                    and args.drop_middle_view_prob > 0
                    and random.random() < args.drop_middle_view_prob
                ):
                    available_indices = list(range(1, num_views - 1))
                    num_to_drop = min(args.drop_view_num, len(available_indices))
                    if num_to_drop > 0:
                        drop_frame_indices = random.sample(available_indices, num_to_drop)
                        if local_rank == 0 and global_step % 100 == 0:
                            logger.info(f"[Temporal Aug] Dropping middle view indices {drop_frame_indices} before voxel fusion")

                gaussian_features, static_points, frame_idx, static_gs_conf = extract_gaussian_features(
                    point_map, gs_map, gs_conf, dy_map, timestamps, bg_mask, drop_frame_indices=drop_frame_indices
                )
                
                if use_voxelize:
                    # Voxelize: Group Gaussians spatially
                    voxel_indices, num_voxels, voxel_counts = voxelizer.voxelize(
                        static_points, gaussian_features
                    )

                    if local_rank == 0 and global_step % 100 == 0:
                        logger.info(
                            f"\n[Voxelization] {static_points.shape[0]} Gaussians -> {num_voxels} Voxels "
                            f"(compression: {static_points.shape[0]/num_voxels:.1f}x)"
                        )
                
                # Extract dynamic Gaussians for each frame
                dynamic_gs_per_frame = []
                for i in range(dy_map.shape[1]):
                    point_map_i = point_map[:, i]
                    bg_mask_i = bg_mask[:, i]
                    if bg_mask_i.sum() > 0:
                        dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
                        dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(
                            gs_map[:, i], bg_mask_i
                        )
                        gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid()
                        dynamic_opacity = dynamic_opacity * gs_dynamic_list_i
                        dynamic_gs_per_frame.append(
                            (dynamic_point, dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation)
                        )
                    else:
                        dynamic_gs_per_frame.append(None)
                
                # Render each frame with temporally-fused voxels
                chunked_renders = []
                chunked_alphas = []
                
                # Detach gaussian_features to save memory (don't backprop through it)
                gaussian_features_detached = gaussian_features
                
                for idx in range(extrinsic.shape[0]):
                    t_current = timestamps[idx].float()
                    
                    if use_voxelize:
                        # Fuse Gaussians within each voxel based on temporal proximity to t_current
                        if args.fusion_version == 'v3':
                            fused_gaussians, attn_weights = fusion_module(
                                gaussian_features_detached,
                                voxel_indices,
                                num_voxels,
                                t_current,
                                gs_conf=static_gs_conf,
                            )
                        else:
                            fused_gaussians, attn_weights = fusion_module(
                                gaussian_features_detached,
                                voxel_indices,
                                num_voxels,
                                t_current,
                            )
                        fused_static_gs = unpack_gaussian_features(fused_gaussians)
                    else:
                        # No voxel baseline: directly render all concatenated static Gaussians.
                        fused_static_gs = unpack_gaussian_features(gaussian_features_detached)
                    
                    # Render with fused static + dynamic Gaussians
                    renders, alphas = render_frame(
                        fused_static_gs,
                        dynamic_gs_per_frame[idx],
                        extrinsic[idx],
                        intrinsic[idx],
                        H, W
                    )
                    
                    chunked_renders.append(renders)
                    chunked_alphas.append(alphas)
                    
                    # Clear cache after each frame to prevent fragmentation
                    if idx % 2 == 0:
                        torch.cuda.empty_cache()
                
                # Composite renders
                renders = torch.cat(chunked_renders, dim=0)
                alphas = torch.cat(chunked_alphas, dim=0)
                bg_render = model.module.sky_model(images, extrinsic, intrinsic)
                renders = alphas * renders + (1 - alphas) * bg_render
                
                rendered_image = renders.permute(0, 3, 1, 2)
                target_image = images[0]
                
                # Compute losses
                normal_pred_loss = None
                normal_gs_loss = None
                gs_anisotropy_loss = None
                static_only_for_normal = (
                    args.normal_static_only and epoch >= args.normal_static_only_warmup_epochs
                )
                if args.normal_loss_warmup_steps > 0:
                    normal_warmup_scale = min(1.0, float(global_step) / float(args.normal_loss_warmup_steps))
                else:
                    normal_warmup_scale = 1.0
                normal_pred_weight_eff = args.normal_pred_weight * normal_warmup_scale
                normal_gs_weight_eff = args.normal_gs_weight * normal_warmup_scale
                gs_anisotropy_weight_eff = args.gs_anisotropy_weight * normal_warmup_scale
                if normal_gt is not None and gs_normal_map is not None:
                    normal_pred_loss = compute_pred_normal_loss(
                        gs_normal_map, normal_gt, normal_valid_mask, bg_mask, dy_map,
                        static_only=static_only_for_normal,
                    )
                    if normal_gs_weight_eff > 0:
                        normal_gs_loss = compute_gs_normal_loss_against_gt(
                            gs_map=gs_map,
                            gt_normals=normal_gt,
                            extrinsics=extrinsics,
                            bg_mask=bg_mask,
                            dy_map=dy_map,
                            gt_valid_mask=normal_valid_mask,
                            static_only=static_only_for_normal,
                            softmin_temp=args.normal_softmin_temp,
                            quat_format=args.quat_format,
                        )
                if args.use_normal_supervision and gs_anisotropy_weight_eff > 0:
                    gs_anisotropy_loss = compute_gs_anisotropy_loss(
                        gs_map=gs_map,
                        bg_mask=bg_mask,
                        dy_map=dy_map,
                        gt_valid_mask=normal_valid_mask,
                        static_only=static_only_for_normal,
                        target_ratio=args.gs_anisotropy_target_ratio,
                    )
                loss, loss_dict = compute_losses(
                    rendered_image, target_image, alphas, sky_mask[0],
                    dy_map[0], dynamic_masks[0] if dynamic_masks is not None else None,
                    lpips_loss_fn, global_step,
                    normal_pred_loss=normal_pred_loss,
                    normal_pred_weight=normal_pred_weight_eff,
                    normal_gs_loss=normal_gs_loss,
                    normal_gs_weight=normal_gs_weight_eff,
                    gs_anisotropy_loss=gs_anisotropy_loss,
                    gs_anisotropy_weight=gs_anisotropy_weight_eff,
                )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
        
        # Logging
        if local_rank == 0 and epoch % 1 == 0:
            logger.info(f"[Epoch {epoch}/{args.max_epoch}] "
                  f"Total: {loss.item():.4f} | "
                  f"L1: {loss_dict['l1']:.4f} | "
                  f"LPIPS: {loss_dict['lpips']:.4f} | "
                  f"Sky: {loss_dict['sky_mask']:.4f} | "
                  f"Dyn: {loss_dict['dynamic']:.4f} | "
                  f"NPred: {loss_dict['normal_pred']:.4f} | "
                  f"NGS: {loss_dict['normal_gs']:.4f} | "
                  f"Aniso: {loss_dict['gs_anisotropy']:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save visualization
        if local_rank == 0 and epoch % args.save_image == 0:
            random_frame_idx = random.randint(0, rendered_image.shape[0] - 1)
            rendered = rendered_image[random_frame_idx].detach().cpu().clamp(0, 1)
            target = target_image[random_frame_idx].detach().cpu().clamp(0, 1)
            dy_map_sigmoid = torch.sigmoid(dy_map[0, random_frame_idx]).detach().cpu()
            dy_map_rgb = dy_map_sigmoid.unsqueeze(0).repeat(3, 1, 1)
            sem_rgb = alphas[random_frame_idx, ..., 0].unsqueeze(0).repeat(3, 1, 1).cpu()
            combined = torch.cat([target, rendered, dy_map_rgb, sem_rgb], dim=-1)
            T.ToPILImage()(combined).save(
                os.path.join(args.log_dir, "images", f"epoch_{epoch:05d}.png")
            )
            if args.use_normal_supervision and normal_gt is not None and gs_normal_map is not None:
                mask_i = bg_mask[:, random_frame_idx]
                if static_only_for_normal:
                    mask_i = mask_i & (dy_map[:, random_frame_idx].sigmoid() < 0.5)
                if normal_valid_mask is not None:
                    mask_i = mask_i & normal_valid_mask[:, random_frame_idx]

                h_i, w_i = mask_i.shape[-2:]
                pred_normal_map = torch.zeros((h_i, w_i, 3), device=device, dtype=gs_map.dtype)
                if mask_i.sum() > 0:
                    pred_cam_i = F.normalize(gs_normal_map[:, random_frame_idx][mask_i].reshape(-1, 3), dim=-1, eps=1e-8)
                    pred_normal_map[mask_i[0]] = pred_cam_i

                gt_normal_map = normal_gt[0, random_frame_idx].detach().clone()
                if normal_valid_mask is not None:
                    gt_normal_map[~normal_valid_mask[0, random_frame_idx]] = 0.0

                pred_normal_vis = (pred_normal_map.permute(2, 0, 1).detach().clamp(-1, 1) * 0.5 + 0.5).cpu()
                gt_normal_vis = (gt_normal_map.permute(2, 0, 1).clamp(-1, 1) * 0.5 + 0.5).cpu()
                valid_vis = mask_i[0].float().unsqueeze(0).repeat(3, 1, 1).detach().cpu()
                normal_combined = torch.cat([target, gt_normal_vis, pred_normal_vis, valid_vis], dim=-1).clamp(0, 1)
                T.ToPILImage()(normal_combined).save(
                    os.path.join(args.log_dir, "images", f"epoch_{epoch:05d}_normal.png")
                )
        
        # Save checkpoint
        if local_rank == 0 and epoch % args.save_ckpt == 0:
            ckpt_path = ckpt_manager.save_trainable(
                model=model,
                fusion_module=fusion_module,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
            )
            logger.info(f"[Checkpoint] Saved at epoch {epoch}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
