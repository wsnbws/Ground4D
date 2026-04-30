import argparse
import os
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torchvision.transforms as T
from PIL import Image

# DGGT components
from ground4d.models.vggt import VGGT
from ground4d.utils.pose_enc import pose_encoding_to_extri_intri
from ground4d.utils.geometry import unproject_depth_map_to_point_map
from ground4d.utils.gs import concat_list, get_split_gs
from gsplat.rendering import rasterization
from datasets.orad_dataset import OradDataset
from datasets.multi_view_infer_dataset import MultiViewInferDataset

# Mode 3 components
from third_party.TAPIP3D.utils.inference_utils import load_model
from utils.interplation import interp_all

# Voxel components
from ground4d.voxelize_v2 import GaussianVoxelizerV2, TemporalVoxelFusion, TemporalVoxelFusionV3

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # Ensure dataloader is deterministic
    torch.use_deterministic_algorithms(False) # Some kernels don't support it

def alpha_t(t, t0, alpha, gamma0=1, gamma1=0.1):
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma*(t0-t)**2)
    alpha_ = alpha * conf
    return alpha_.float()

def extract_gaussian_features(point_map, gs_map, dy_map, timestamps, bg_mask):
    """Extract Gaussian features using bg_mask for consistency with inference.py"""
    static_mask = bg_mask
    static_points = point_map[static_mask].reshape(-1, 3)
    
    gs_dynamic_list = dy_map[static_mask].sigmoid()
    static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
    static_opacity = static_opacity * (1 - gs_dynamic_list)
    
    frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
    gs_timestamps = timestamps[frame_idx].float()
    
    gaussian_features = torch.cat([
        static_points,                                                          # [N, 3]
        static_rgbs,                                                            # [N, 3]
        static_opacity if static_opacity.dim() == 2 else static_opacity.unsqueeze(-1),  # [N, 1]
        static_scales,                                                          # [N, 3]
        static_rotations,                                                       # [N, 4]
        gs_timestamps.unsqueeze(-1) if gs_timestamps.dim() == 1 else gs_timestamps      # [N, 1]
    ], dim=-1)
    
    return gaussian_features, static_points, frame_idx

def unpack_gaussian_features(fused_gaussians):
    """Unpack fused Gaussian features as in voxel_train_v2.py"""
    points = fused_gaussians[:, 0:3]
    rgbs = fused_gaussians[:, 3:6]
    opacity = fused_gaussians[:, 6:7].squeeze(-1)
    scales = fused_gaussians[:, 7:10]
    rotations = fused_gaussians[:, 10:14]
    return points, rgbs, opacity, scales, rotations


def save_gaussians_npy(gaussian_dir, frame_idx, points, rgbs, opacity, scales, rotations, timestamp=None):
    """Save pre-rasterization Gaussians for external visualization."""
    os.makedirs(gaussian_dir, exist_ok=True)
    save_path = os.path.join(gaussian_dir, f"gaussians_frame_{frame_idx:04d}.npy")

    if timestamp is None:
        timestamp_val = None
    elif torch.is_tensor(timestamp):
        timestamp_val = float(timestamp.detach().cpu().item())
    else:
        timestamp_val = float(timestamp)

    payload = {
        "points": points.detach().float().cpu().numpy(),          # [N, 3]
        "rgbs": rgbs.detach().float().cpu().numpy(),              # [N, 3]
        "opacity": opacity.detach().float().reshape(-1, 1).cpu().numpy(),  # [N, 1]
        "scales": scales.detach().float().cpu().numpy(),          # [N, 3]
        "rotations": rotations.detach().float().cpu().numpy(),    # [N, 4]
        "timestamp": timestamp_val,
    }
    np.save(save_path, payload, allow_pickle=True)

def compute_metrics(img1, img2, lpips_fn):
    """Compute PSNR, SSIM, and LPIPS between two image tensors [S, 3, H, W]"""
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    psnr_list, ssim_list, lpips_list = [], [], []
    for i in range(img1.shape[0]):
        im1 = img1[i].cpu().permute(1, 2, 0).numpy()
        im2 = img2[i].cpu().permute(1, 2, 0).numpy()
        
        # PSNR
        psnr = peak_signal_noise_ratio(im2, im1, data_range=1.0)
        psnr_list.append(psnr)
        
        # SSIM
        ssim = structural_similarity(im2, im1, channel_axis=2, data_range=1.0)
        ssim_list.append(ssim)
        
        # LPIPS
        lp = lpips_fn(img1[i:i+1] * 2 - 1, img2[i:i+1] * 2 - 1).item()
        lpips_list.append(lp)
        
    return np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)


def _load_backbone_state(model, checkpoint, strict=False):
    """Load full-model state if present, otherwise fallback safely."""
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                model.load_state_dict(checkpoint[key], strict=strict)
                return
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=strict)
    else:
        raise ValueError("Unsupported checkpoint format")


def _checkpoint_has_full_model(checkpoint):
    """Return True if checkpoint contains full model state (backbone + heads)."""
    if not isinstance(checkpoint, dict):
        return False
    for key in ("model", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state = checkpoint[key]
            # Full DGGT has many keys (backbone + heads); trainable-only has only trainable keys
            if len(state) > 20:
                return True
    return False


def _load_trainable_heads(model, checkpoint):
    """Load head-only checkpoint format from voxel_train_v2 CheckpointManager."""
    trainable = checkpoint.get("trainable", None) if isinstance(checkpoint, dict) else None
    if not isinstance(trainable, dict):
        return False
    if "gs_head" in trainable:
        model.gs_head.load_state_dict(trainable["gs_head"], strict=False)
    if "instance_head" in trainable:
        model.instance_head.load_state_dict(trainable["instance_head"], strict=False)
    if "sky_model" in trainable:
        model.sky_model.load_state_dict(trainable["sky_model"], strict=False)
    return True


def _load_fusion_state(fusion_module, checkpoint):
    if fusion_module is None or not isinstance(checkpoint, dict):
        return
    if "trainable" in checkpoint and isinstance(checkpoint["trainable"], dict):
        trainable = checkpoint["trainable"]
        if "fusion_module" in trainable:
            fusion_module.load_state_dict(trainable["fusion_module"], strict=False)
            return
    if "fusion_module" in checkpoint and isinstance(checkpoint["fusion_module"], dict):
        fusion_module.load_state_dict(checkpoint["fusion_module"], strict=False)

def main():
    parser = argparse.ArgumentParser(description="Custom inference for DGGT and voxel models.")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to dataset root")
    parser.add_argument('--dataset_type', type=str, default='orad', choices=['orad', 'rellis'],
                        help="Dataset type: 'orad' or 'rellis'")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to model checkpoint (for voxel_v2: trainable heads + optional fusion)")
    parser.add_argument('--dggt_ckpt_path', type=str, default=None,
                        help="Path to pretrained DGGT full checkpoint. Required for voxel_v2 when ckpt_path only contains trainable heads (as saved by voxel_train_v2).")
    parser.add_argument('--model_type', type=str, choices=['dggt', 'voxel_v2'], required=True, help="Type of model to test")
    parser.add_argument('--mode', type=int, default=2, choices=[2, 3], help="Inference mode: 2 for reconstruction, 3 for interpolation")
    parser.add_argument('--sequence_length', type=int, default=4, help="Sequence length for testing")
    parser.add_argument('--use_mode3_multiview_dataset', action='store_true',
                        help="Use MultiViewInferDataset for mode=3 testing (fixed targets + 4/7/10 context ablation).")
    parser.add_argument('--start_idx', type=int, default=0, help="Start index for testing sequence")
    parser.add_argument('--interval', type=int, default=1, help="Frame interval for testing sequence (mode 2) or interpolation interval (mode 3)")
    parser.add_argument('--n_inter_frames', type=int, default=None, help="Number of intermediate frames for interpolation (mode 3)")
    parser.add_argument('--voxel_size', type=float, default=0.002, help="Voxel size for voxel mode")
    parser.add_argument('--fusion_version', type=str, default='v1', choices=['v1', 'v3'],
                        help="Temporal fusion version for voxel inference")
    parser.add_argument('--feature_dim', type=int, default=64, help="Fusion feature dimension")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Fusion hidden dimension")
    parser.add_argument('--ab_use_voxelize', type=int, default=1, choices=[0, 1],
                        help="Ablation switch: 1 to use voxelization, 0 to disable voxel module")
    parser.add_argument('--ab_use_ta', type=int, default=1, choices=[0, 1],
                        help="Ablation switch: 1 to use temporal attention in voxel fusion")
    parser.add_argument('--ab_use_in', type=int, default=1, choices=[0, 1],
                        help="Ablation switch: 1 to use voxel-wise normalization (IN) after TA")
    parser.add_argument('--track_ckpt', type=str, default='checkpoints/tapir_checkpoint.npy', help="Path to tracking model checkpoint for mode 3")
    parser.add_argument('--output_dir', type=str, default='results/custom_inference', help="Output directory")
    parser.add_argument('--save_images', action='store_true', help="Save rendered images (and interpolation video in mode 3)")
    parser.add_argument('--save_gaussians_npy', action='store_true',
                        help="Save pre-rasterization Gaussians as .npy for visualization")
    parser.add_argument('--save_gaussians_all_frames', action='store_true',
                        help="When saving Gaussians, dump all frames instead of only frame 0")
    parser.add_argument('--save_gaussians_first_n_batches', type=int, default=-1,
                        help="Only save Gaussians for first N batches (batch_size is 1). -1 means no limit.")
    parser.add_argument('--max_batches', type=int, default=-1,
                        help="Run inference on at most N batches (batch_size is 1). -1 means no limit.")
    parser.add_argument('--profile', action='store_true',
                        help="Profile average FPS, average GPU peak memory, and average Gaussian count.")
    args = parser.parse_args()
    use_voxelize = bool(args.ab_use_voxelize)
    use_ta = bool(args.ab_use_ta)
    use_in = bool(args.ab_use_in)
    if not use_voxelize and (use_ta or use_in):
        raise ValueError("Invalid ablation config: ab_use_ta/ab_use_in require ab_use_voxelize=1.")
    if args.save_gaussians_first_n_batches < -1:
        raise ValueError("--save_gaussians_first_n_batches must be -1 or a non-negative integer.")
    if args.max_batches < -1 or args.max_batches == 0:
        raise ValueError("--max_batches must be -1 or a positive integer.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed for reproducibility (dataset sampling)
    set_seed(42)

    # Load Dataset
    print(f"Loading {args.dataset_type.upper()} dataset from {args.image_dir}...")
    if args.mode == 3 and args.use_mode3_multiview_dataset:
        dataset = MultiViewInferDataset(
            args.image_dir,
            dataset_type=args.dataset_type,
            num_context_views=args.sequence_length,
        )
    else:
        dataset = OradDataset(
            args.image_dir,
            sequence_length=args.sequence_length,
            mode=args.mode,
            start_idx=args.start_idx,
            interval=args.interval,
            n_inter_frames=args.n_inter_frames,
            dataset_type=args.dataset_type
        )
    # Use num_workers=0 to ensure deterministic random sampling in __getitem__
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize Model
    print(f"Initializing {args.model_type} model...")
    model = VGGT().to(device)
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    
    # Track model for mode 3
    track_model = None
    if args.mode == 3:
        if not os.path.exists(args.track_ckpt):
            print(f"Warning: Track checkpoint not found at {args.track_ckpt}. Mode 3 may fail.")
        track_model = load_model(args.track_ckpt)
        track_model.to(device)
        track_model.seq_len = 2

    fusion_module = None
    voxelizer = None
    
    if args.model_type == 'voxel_v2':
        # voxel_train_v2 only saves trainable heads (gs_head, instance_head, sky_model) + fusion_module.
        # So we must first load the full DGGT backbone, then overlay the trained heads from ckpt_path.
        if _checkpoint_has_full_model(checkpoint):
            _load_backbone_state(model, checkpoint, strict=False)
            print("Loaded full model from checkpoint.")
        else:
            # Trainable-only checkpoint: load original DGGT first, then overlay trainable.
            if not args.dggt_ckpt_path:
                raise ValueError(
                    "For voxel_v2 with a trainable-only checkpoint (as saved by voxel_train_v2), "
                    "you must provide --dggt_ckpt_path pointing to the pretrained DGGT checkpoint."
                )
            if not os.path.exists(args.dggt_ckpt_path):
                raise FileNotFoundError(f"dggt_ckpt_path not found: {args.dggt_ckpt_path}")
            dggt_ckpt = torch.load(args.dggt_ckpt_path, map_location=device, weights_only=False)
            _load_backbone_state(model, dggt_ckpt, strict=False)
            print(f"Loaded DGGT backbone from {args.dggt_ckpt_path}.")
        # Overlay trained heads (and fusion) from ckpt_path
        _load_trainable_heads(model, checkpoint)

        if use_voxelize:
            fusion_cls = TemporalVoxelFusionV3 if args.fusion_version == 'v3' else TemporalVoxelFusion
            fusion_module = fusion_cls(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                use_temporal_attention=use_ta,
                use_voxel_norm=use_in,
            ).to(device)
            _load_fusion_state(fusion_module, checkpoint)
            fusion_module.eval()
            voxelizer = GaussianVoxelizerV2(voxel_size=args.voxel_size)
            print(f"Using voxel fusion version: {args.fusion_version}")
            print(f"Ablation switches - Voxelize: {use_voxelize}, TA: {use_ta}, IN: {use_in}")
        else:
            print("Ablation switches - Voxelize: 0, TA: 0, IN: 0 (no voxel baseline)")
    else:
        # Standard DGGT model
        _load_backbone_state(model, checkpoint, strict=True)
    
    model.eval()
    
    # Initialize LPIPS
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # Metrics storage
    psnr_results, ssim_results, lpips_results = [], [], []
    
    print(f"Starting inference on {len(dataloader)} scenes...")

    # Profiling accumulators (batch_size is 1 in this script)
    profile_enabled = bool(args.profile)
    total_time_sec = 0.0
    total_frames = 0
    profile_batches = 0
    total_gauss_sum = 0
    total_gauss_calls = 0
    avg_peak_allocated_bytes_sum = 0
    avg_peak_reserved_bytes_sum = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if args.max_batches > 0 and i >= args.max_batches:
                print(f"Reached --max_batches={args.max_batches}. Stop inference early.")
                break

            images = batch['images'].to(device) # [1, S, 3, H, W]
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2) # [1, S, H, W, 1]
            bg_mask = (sky_mask == 0).any(dim=-1) # [1, S, H, W]
            timestamps = batch['timestamps'][0].to(device) # [S]
            scene_name = batch.get('scene_name', [f"scene_{i:03d}"])[0]
            gaussian_dir = os.path.join(args.output_dir, scene_name, "gaussians")
            within_batch_limit = (
                args.save_gaussians_first_n_batches < 0 or i < args.save_gaussians_first_n_batches
            )
            should_save_gaussians_this_batch = args.save_gaussians_npy and within_batch_limit
            
            target_images = None
            if args.mode == 3:
                if 'targets' in batch:
                    target_images = batch['targets'].to(device) # [T, 3, H, W]
                    print(f"Debug: Loaded targets with shape {target_images.shape}")
                else:
                    print(f"Debug: 'targets' NOT found in batch. Keys: {batch.keys()}")

            # Per-batch profiling init
            if profile_enabled and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if profile_enabled and device.type == "cuda":
                torch.cuda.synchronize()
            batch_start_t = time.perf_counter() if profile_enabled else None
            gauss_sum_in_batch = 0
            gauss_calls_in_batch = 0

            with torch.cuda.amp.autocast():
                predictions = model(images)
                H, W = images.shape[-2:]
                
                # Camera parameters
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]

                # Point map
                depth_map = predictions["depth"][0]
                point_map_np = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None, ...]
                point_map = torch.from_numpy(point_map_np).to(device).float()
                
                # Gaussian parameters
                gs_map = predictions["gs_map"]
                gs_conf = predictions["gs_conf"]
                dy_map = predictions["dynamic_conf"].squeeze(-1) # [1, S, H, W]

                chunked_renders = []
                chunked_alphas = []
                S = extrinsic.shape[0]

                if args.model_type == 'dggt':
                    # Match original inference.py logic
                    
                    if args.mode == 2:
                        static_mask = (bg_mask & (dy_map < 0.5))
                        static_points = point_map[static_mask].reshape(-1, 3)
                        
                        gs_dynamic_list = dy_map[static_mask].sigmoid()
                        static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
                        static_opacity = static_opacity * (1 - gs_dynamic_list)
                        
                        static_gs_conf = gs_conf[static_mask]
                        frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
                        gs_timestamps = timestamps[frame_idx]

                        # Pre-extract dynamic Gaussians for each frame using bg_mask
                        dynamic_gs_list = []
                        for idx in range(S):
                            bg_mask_i = bg_mask[:, idx]
                            dynamic_point = point_map[:, idx][bg_mask_i].reshape(-1, 3)
                            dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, idx], bg_mask_i)
                            gs_dynamic_list_i = dy_map[:, idx][bg_mask_i].sigmoid()
                            dynamic_opacity = dynamic_opacity * gs_dynamic_list_i
                            dynamic_gs_list.append((dynamic_point, dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation))
                        
                        # Render loop
                        for idx in range(S):
                            t0 = timestamps[idx]
                            static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0=static_gs_conf)
                            static_gs = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                            
                            if dynamic_gs_list[idx][0].shape[0] > 0:
                                world_points, rgbs, opacity, scales, rotation = concat_list(
                                    static_gs,
                                    dynamic_gs_list[idx]
                                )
                            else:
                                world_points, rgbs, opacity, scales, rotation = static_gs

                            if profile_enabled:
                                # Approximate Gaussian count for this rasterization call.
                                # `world_points` is shaped [N, 3], so N ~= Gaussian number.
                                n_gauss = int(world_points.shape[0])
                                gauss_sum_in_batch += n_gauss
                                gauss_calls_in_batch += 1

                            if should_save_gaussians_this_batch and (args.save_gaussians_all_frames or idx == 0):
                                save_gaussians_npy(
                                    gaussian_dir, idx, world_points, rgbs, opacity, scales, rotation, t0
                                )
                            
                            renders, alphas, _ = rasterization(
                                means=world_points, quats=rotation, scales=scales, opacities=opacity, colors=rgbs,
                                viewmats=extrinsic[idx:idx+1], Ks=intrinsic[idx:idx+1], width=W, height=H
                            )
                            chunked_renders.append(renders)
                            chunked_alphas.append(alphas)

                    elif args.mode == 3:
                        # Interpolation logic
                        # target_images = None # Not available in current OradDataset for mode 3
                        depth_map_in = depth_map.unsqueeze(0)
                        
                        origin_extrinsic = extrinsic
                        origin_intrinsic = intrinsic
                        
                        (extrinsic, intrinsic, point_map, gs_map, dy_map, 
                        gs_conf, bg_mask, _, _, _, depth_interp) = interp_all(
                            extrinsic, intrinsic, point_map, gs_map, dy_map, 
                            gs_conf, bg_mask, images, target_images, depth_map_in, 
                            track_model, interval=args.n_inter_frames+1, views=1
                        )
                        
                        I = args.n_inter_frames+1
                        bg_point_map = point_map[:, ::I, ...]
                        bg_bg_mask = bg_mask[:, ::I, ...]
                        bg_gs_map = gs_map[:, ::I, ...]
                        bg_dy_map = dy_map[:, ::I, ...]
                        bg_gs_conf = gs_conf[:, ::I, ...]

                        static_mask = (bg_bg_mask & (bg_dy_map < 0.5))
                        gs_conf_static = bg_gs_conf[static_mask]
                        static_points = bg_point_map[static_mask].reshape(-1, 3)
                        gs_dynamic_list = bg_dy_map[static_mask].sigmoid()
                        static_rgbs, static_opacity, static_scales, static_rotation = get_split_gs(bg_gs_map, static_mask)
                        frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
                        gs_timestamps = timestamps[frame_idx]
                        static_opacity = static_opacity * (1 - gs_dynamic_list)
                        
                        dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
                        for i in range(dy_map.shape[1]):
                            point_map_i = point_map[:, i]
                            bg_mask_i = bg_mask[:, i]
                            
                            dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
                            dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rot_dyn = get_split_gs(gs_map[:, i], bg_mask_i)
                            gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid()
                            dynamic_opacity = dynamic_opacity * gs_dynamic_list_i

                            dynamic_points.append(dynamic_point)
                            dynamic_rgbs.append(dynamic_rgb)
                            dynamic_opacitys.append(dynamic_opacity)
                            dynamic_scales.append(dynamic_scale)
                            dynamic_rotations.append(dynamic_rot_dyn)

                        for idx in range(dy_map.shape[1]):
                            I = args.interval
                            t0 = timestamps[idx // I] # Use keyframe timestamp for alpha_t reference
                            # Note: timestamps might need to be interpolated if used for precise decay, 
                            # but inference.py uses index-based mapping: t0 = timestamps[idx//I]
                            
                            static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0=gs_conf_static, gamma1=0.1)

                            dynamic_gs_list_idx = [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]
                            
                            if dynamic_points[idx].shape[0] > 0:
                                world_points, rgbs, opacity, scales, rotation = concat_list(
                                    [static_points, static_rgbs, static_opacity_, static_scales, static_rotation],
                                    dynamic_gs_list_idx
                                )
                            else:
                                world_points, rgbs, opacity, scales, rotation = [static_points, static_rgbs, static_opacity_, static_scales, static_rotation]

                            if profile_enabled:
                                n_gauss = int(world_points.shape[0])
                                gauss_sum_in_batch += n_gauss
                                gauss_calls_in_batch += 1

                            if should_save_gaussians_this_batch and (args.save_gaussians_all_frames or idx == 0):
                                save_gaussians_npy(
                                    gaussian_dir, idx, world_points, rgbs, opacity, scales, rotation, t0
                                )

                            renders_chunk, alphas_chunk, _ = rasterization(
                                means=world_points,
                                quats=rotation,
                                scales=scales,
                                opacities=opacity,
                                colors=rgbs,
                                viewmats=extrinsic[idx:idx+1],
                                Ks=intrinsic[idx:idx+1],
                                width=W,
                                height=H,
                                render_mode='RGB+ED',  
                            )
                            chunked_renders.append(renders_chunk)
                            chunked_alphas.append(alphas_chunk)

                elif args.model_type == 'voxel_v2':

                    depth_map_in = depth_map.unsqueeze(0)
                    
                    origin_extrinsic = extrinsic
                    origin_intrinsic = intrinsic

                    (extrinsic, intrinsic, point_map, gs_map, dy_map, 
                    gs_conf, bg_mask, _, _, _, depth_interp) = interp_all(
                        extrinsic, intrinsic, point_map, gs_map, dy_map, 
                        gs_conf, bg_mask, images, target_images, depth_map_in, 
                        track_model, interval=args.n_inter_frames+1, views=1
                    )
                
                    # Extract static Gaussians from keyframes
                    I = args.n_inter_frames+1
                    bg_point_map = point_map[:, ::I, ...]
                    bg_bg_mask = bg_mask[:, ::I, ...]
                    bg_gs_map = gs_map[:, ::I, ...]
                    bg_dy_map = dy_map[:, ::I, ...]
                    bg_gs_conf = gs_conf[:, ::I, ...]
                    
                    static_mask = (bg_bg_mask & (bg_dy_map < 0.5))
                    gaussian_features, static_points, _ = extract_gaussian_features(
                        bg_point_map, bg_gs_map, bg_dy_map, timestamps, static_mask
                    )
                    
                    if use_voxelize:
                        # Voxelize static Gaussians
                        voxel_indices, num_voxels, _ = voxelizer.voxelize(static_points, gaussian_features)
                    
                    # Extract dynamic Gaussians for all interpolated frames
                    num_frames = dy_map.shape[1]
                    dynamic_gs_list = []
                    for idx in range(num_frames):
                        bg_mask_i = bg_mask[:, idx]
                        if bg_mask_i.sum() > 0:
                            dynamic_point = point_map[:, idx][bg_mask_i].reshape(-1, 3)
                            dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, idx], bg_mask_i)
                            gs_dynamic_list_i = dy_map[:, idx][bg_mask_i].sigmoid()
                            dynamic_opacity = dynamic_opacity * gs_dynamic_list_i
                            dynamic_gs_list.append((dynamic_point, dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation))
                        else:
                            dynamic_gs_list.append(None)
                    
                    # Build continuous timestamps for interpolated frames so
                    # temporal fusion is conditioned on the true in-between time.
                    interp_timestamps = []
                    for seg_idx in range(len(timestamps) - 1):
                        t_start = timestamps[seg_idx].float()
                        t_end = timestamps[seg_idx + 1].float()
                        for step_idx in range(I):
                            alpha = step_idx / float(I)
                            interp_timestamps.append((1.0 - alpha) * t_start + alpha * t_end)
                    interp_timestamps.append(timestamps[-1].float())
                    interp_timestamps = torch.stack(interp_timestamps)
                    
                    # Render all interpolated frames
                    for idx in range(num_frames):
                        # t_current = timestamps[idx // I].float()
                        t_current = interp_timestamps[idx]
                        
                        if use_voxelize:
                            # Temporal fusion of voxels
                            fused_gaussians, _ = fusion_module(gaussian_features, voxel_indices, num_voxels, t_current)
                            fused_static_gs = unpack_gaussian_features(fused_gaussians)
                        else:
                            # No voxel baseline: directly render all concatenated static Gaussians.
                            fused_static_gs = unpack_gaussian_features(gaussian_features)
                        
                        if dynamic_gs_list[idx] is not None:
                            world_points, rgbs, opacity, scales, rotation = concat_list(
                                list(fused_static_gs), 
                                list(dynamic_gs_list[idx])
                            )
                        else:
                            world_points, rgbs, opacity, scales, rotation = fused_static_gs

                        if profile_enabled:
                            n_gauss = int(world_points.shape[0])
                            gauss_sum_in_batch += n_gauss
                            gauss_calls_in_batch += 1

                        if should_save_gaussians_this_batch and (args.save_gaussians_all_frames or idx == 0):
                            save_gaussians_npy(
                                gaussian_dir, idx, world_points, rgbs, opacity, scales, rotation, t_current
                            )
                        
                        renders, alphas, _ = rasterization(
                            means=world_points, quats=rotation, scales=scales, opacities=opacity, colors=rgbs,
                            viewmats=extrinsic[idx:idx+1], Ks=intrinsic[idx:idx+1], width=W, height=H,
                            render_mode='RGB+ED'
                        )
                        chunked_renders.append(renders)
                        chunked_alphas.append(alphas)

                # Composite background
                renders = torch.cat(chunked_renders, dim=0)
                alphas = torch.cat(chunked_alphas, dim=0)
                
                if args.mode == 3:
                     # For mode 3, we need to generate background for new poses
                     # origin_extrinsic/intrinsic were saved in the mode 3 block
                     bg_render = model.sky_model.forward_with_new_pose(images, origin_extrinsic, origin_intrinsic, extrinsic, intrinsic)
                else:
                     bg_render = model.sky_model(images, extrinsic, intrinsic)
                
                # Special normalization for dggt model (from inference.py)
                if args.model_type == 'dggt':
                    bg_render = (bg_render - bg_render.min()) / (bg_render.max() - bg_render.min() + 1e-8)
                
                # Final composite
                renders = alphas * renders[..., :3] + (1 - alphas) * bg_render
                rendered_image = renders.permute(0, 3, 1, 2) # [S_new, 3, H, W]
                
                # Metrics
                if args.mode == 2 or (args.mode == 3 and target_images is not None):
                    if args.mode == 2:
                        target_image = images[0] # [S, 3, H, W]
                    else: # args.mode == 3
                        target_image = target_images[0] # [T, 3, H, W]
                    
                    print(f"Debug: Computing metrics. Rendered: {rendered_image.shape}, Target: {target_image.shape}")
                    
                    # Ensure shapes match before computing metrics
                    if rendered_image.shape == target_image.shape:
                        psnr, ssim, lpips_val = compute_metrics(rendered_image, target_image, lpips_fn)
                        psnr_results.append(psnr)
                        ssim_results.append(ssim)
                        lpips_results.append(lpips_val)
                    else:
                        print(f"Warning: Shape mismatch for metrics. Rendered: {rendered_image.shape}, Target: {target_image.shape}")
                else:
                    if args.mode == 3:
                        print("Debug: Skipping metrics because target_images is None")

                if args.save_images:
                    scene_dir = os.path.join(args.output_dir, scene_name)
                    os.makedirs(scene_dir, exist_ok=True)
                    S_out = rendered_image.shape[0]
                    for s in range(S_out):
                        img_path = os.path.join(scene_dir, f"frame_{s:02d}.png")
                        img = T.ToPILImage()(rendered_image[s].clamp(0, 1))
                        img.save(img_path)
                    
                    # Save video
                    if args.mode == 3:
                        video_path = os.path.join(scene_dir, "rendered_video.mp4")
                        import imageio
                        video_frames = (rendered_image.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                        imageio.mimwrite(video_path, video_frames, fps=8, codec="libx264")

            # Per-batch profiling finalize (must be after rendering/composition)
            if profile_enabled:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    peak_alloc_bytes = torch.cuda.max_memory_allocated(device)
                    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
                else:
                    peak_alloc_bytes = 0
                    peak_reserved_bytes = 0

                batch_elapsed = time.perf_counter() - batch_start_t
                frames_out = int(rendered_image.shape[0]) if 'rendered_image' in locals() else 0

                total_time_sec += float(batch_elapsed)
                total_frames += int(frames_out)
                profile_batches += 1
                if device.type == "cuda":
                    avg_peak_allocated_bytes_sum += int(peak_alloc_bytes)
                    avg_peak_reserved_bytes_sum += int(peak_reserved_bytes)
                total_gauss_sum += int(gauss_sum_in_batch)
                total_gauss_calls += int(gauss_calls_in_batch)

    # Summary
    print(f"\n========================================")
    print(f" Final Evaluation Summary: {args.model_type} Mode {args.mode}")
    print(f"========================================")
    if (args.mode == 2 or args.mode == 3) and psnr_results:
        print(f" PSNR:  {np.mean(psnr_results):.4f}")
        print(f" SSIM:  {np.mean(ssim_results):.4f}")
        print(f" LPIPS: {np.mean(lpips_results):.4f}")
    else:
        print(" Metrics not computed (no data or no targets).")
    print(f"========================================\n")

    if profile_enabled:
        # Average FPS = total rendered frames / total wall time.
        avg_fps = (total_frames / total_time_sec) if total_time_sec > 0 else 0.0
        avg_peak_alloc_mb = (avg_peak_allocated_bytes_sum / max(1, profile_batches)) / (1024.0 * 1024.0)
        avg_peak_reserved_mb = (avg_peak_reserved_bytes_sum / max(1, profile_batches)) / (1024.0 * 1024.0)
        avg_gauss = (total_gauss_sum / max(1, total_gauss_calls)) if total_gauss_calls > 0 else 0.0
        print("========================================")
        print(f" Profiling Summary (average):")
        print(f"  Avg FPS: {avg_fps:.3f}")
        if device.type == "cuda":
            print(f"  Avg GPU peak allocated: {avg_peak_alloc_mb:.1f} MB")
            print(f"  Avg GPU peak reserved:  {avg_peak_reserved_mb:.1f} MB")
        print(f"  Avg Gaussians per raster call: {avg_gauss:.0f}")
        print("========================================")

    # Save results to file
    result_path = os.path.join(args.output_dir, f"metrics_{args.model_type}_mode{args.mode}.txt")
    with open(result_path, "w") as f:
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Checkpoint: {args.ckpt_path}\n")
        if (args.mode == 2 or args.mode == 3) and psnr_results:
            f.write(f"PSNR:  {np.mean(psnr_results)}\n")
            f.write(f"SSIM:  {np.mean(ssim_results)}\n")
            f.write(f"LPIPS: {np.mean(lpips_results)}\n")
        else:
            f.write("Metrics not computed.\n")
    print(f"Metrics saved to {result_path}")

if __name__ == "__main__":
    main()