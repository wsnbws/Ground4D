import torch
import torch.nn.functional as F

from ground4d.utils.gs import get_split_gs


def fetch_normal_gt(batch, normal_key, device):
    """Fetch and normalize normal GT to shape [B, S, H, W, 3]."""
    candidates = [normal_key, "normal", "normals", "normal_map", "normal_gt"]
    normal_gt = None
    for key in candidates:
        if key in batch:
            normal_gt = batch[key]
            break
    if normal_gt is None:
        return None

    normal_gt = normal_gt.to(device)
    if normal_gt.dim() != 5:
        return None

    if normal_gt.shape[2] == 3:
        normal_gt = normal_gt.permute(0, 1, 3, 4, 2)
    elif normal_gt.shape[-1] != 3:
        return None

    if normal_gt.min() >= 0.0 and normal_gt.max() <= 1.0:
        normal_gt = normal_gt * 2.0 - 1.0

    return F.normalize(normal_gt, dim=-1, eps=1e-6)


def fetch_normal_valid_mask(batch, device):
    """Fetch normal valid mask to shape [B, S, H, W]."""
    candidates = ["normal_valid_mask", "normals_valid_mask", "normal_mask", "normal_valid"]
    valid_mask = None
    for key in candidates:
        if key in batch:
            valid_mask = batch[key]
            break
    if valid_mask is None:
        return None

    valid_mask = valid_mask.to(device)
    if valid_mask.dim() == 5:
        if valid_mask.shape[2] == 1:
            valid_mask = valid_mask[:, :, 0]
        elif valid_mask.shape[-1] == 1:
            valid_mask = valid_mask[..., 0]
        elif valid_mask.shape[2] == 3:
            valid_mask = valid_mask[:, :, 0]
        elif valid_mask.shape[-1] == 3:
            valid_mask = valid_mask[..., 0]
        else:
            return None
    elif valid_mask.dim() != 4:
        return None

    return valid_mask > 0.5


def compute_pred_normal_loss(
    pred_normals,
    gt_normals,
    gt_valid_mask,
    bg_mask,
    dy_map,
    static_only,
):
    """Cosine loss between predicted normals and GT normals in camera coordinates."""
    if pred_normals is None or gt_normals is None:
        return None

    pred_normals = F.normalize(pred_normals, dim=-1, eps=1e-8)
    gt_normals = F.normalize(gt_normals, dim=-1, eps=1e-8)

    total_loss = 0.0
    valid_frames = 0
    num_frames = pred_normals.shape[1]

    for i in range(num_frames):
        mask_i = bg_mask[:, i]
        if static_only:
            mask_i = mask_i & (dy_map[:, i].sigmoid() < 0.5)
        if gt_valid_mask is not None:
            mask_i = mask_i & gt_valid_mask[:, i]
        if mask_i.sum() == 0:
            continue

        pred_i = pred_normals[:, i][mask_i].reshape(-1, 3)
        gt_i = gt_normals[:, i][mask_i].reshape(-1, 3)
        valid = torch.isfinite(gt_i).all(dim=-1) & torch.isfinite(pred_i).all(dim=-1)
        if valid.sum() == 0:
            continue

        cos = F.cosine_similarity(pred_i[valid], gt_i[valid], dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        # cosine loss + angular L1 loss for stronger gradient near zero error
        angular_loss = torch.acos(cos).mean()
        total_loss = total_loss + 0.5 * (1.0 - cos).mean() + 0.5 * angular_loss
        valid_frames += 1

    if valid_frames == 0:
        return pred_normals.new_tensor(0.0)
    return total_loss / valid_frames


def compute_gs_anisotropy_loss(
    gs_map,
    bg_mask,
    dy_map,
    gt_valid_mask,
    static_only,
    target_ratio,
):
    """
    Encourage per-Gaussian anisotropy for stable normal prediction.
    Penalize only when s_min / mean(s_mid, s_max) exceeds target_ratio.
    """
    total_loss = 0.0
    valid_frames = 0
    num_frames = gs_map.shape[1]

    for i in range(num_frames):
        mask_i = bg_mask[:, i]
        if static_only:
            mask_i = mask_i & (dy_map[:, i].sigmoid() < 0.5)
        if gt_valid_mask is not None:
            mask_i = mask_i & gt_valid_mask[:, i]
        if mask_i.sum() == 0:
            continue

        _, _, scales_i, _ = get_split_gs(gs_map[:, i], mask_i)
        if scales_i.numel() == 0:
            continue

        scales_sorted, _ = torch.sort(scales_i, dim=-1)
        s0 = scales_sorted[:, 0]
        s12 = 0.5 * (scales_sorted[:, 1] + scales_sorted[:, 2])
        ratio = s0 / (s12 + 1e-8)
        total_loss = total_loss + F.relu(ratio - target_ratio).mean()
        valid_frames += 1

    if valid_frames == 0:
        return gs_map.new_tensor(0.0)
    return total_loss / valid_frames


def quat_rotate_vec(quat, vec, quat_format="xyzw"):
    """Rotate vectors by unit quaternions."""
    q = F.normalize(quat, dim=-1, eps=1e-8)
    if quat_format == "xyzw":
        q_xyz, q_w = q[:, :3], q[:, 3:4]
    else:
        q_w, q_xyz = q[:, :1], q[:, 1:4]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def estimate_normals_from_gs(scales, rotations, softmin_temp=10.0, quat_format="xyzw"):
    """Estimate normals from GS scales and rotations."""
    axis_weights = torch.softmax(-softmin_temp * scales, dim=-1)
    local_normal = F.normalize(axis_weights, dim=-1, eps=1e-8)
    world_normal = quat_rotate_vec(rotations, local_normal, quat_format=quat_format)
    return F.normalize(world_normal, dim=-1, eps=1e-8)


def compute_normal_consistency_loss(
    pred_normals,
    gs_map,
    extrinsics,
    bg_mask,
    dy_map,
    gt_valid_mask=None,
    static_only=False,
    softmin_temp=10.0,
    quat_format="xyzw",
):
    """
    Consistency loss between predicted normals and normals estimated from GS params.
    Predicted and estimated normals are compared in camera coordinates.
    """
    if pred_normals is None:
        return None

    total_loss = 0.0
    valid_frames = 0
    pred_normals = F.normalize(pred_normals, dim=-1, eps=1e-8)
    num_frames = pred_normals.shape[1]

    for i in range(num_frames):
        mask_i = bg_mask[:, i]
        if static_only:
            mask_i = mask_i & (dy_map[:, i].sigmoid() < 0.5)
        if gt_valid_mask is not None:
            mask_i = mask_i & gt_valid_mask[:, i]
        if mask_i.sum() == 0:
            continue

        pred_i = pred_normals[:, i][mask_i].reshape(-1, 3)
        _, _, scales_i, rot_i = get_split_gs(gs_map[:, i], mask_i)
        gs_world_i = estimate_normals_from_gs(
            scales_i, rot_i, softmin_temp=softmin_temp, quat_format=quat_format
        )

        rot_wc = extrinsics[:, i, :3, :3]
        h_i, w_i = mask_i.shape[-2:]
        rot_wc_masked = (
            rot_wc[:, None, None, :, :]
            .expand(-1, h_i, w_i, -1, -1)[mask_i]
            .reshape(-1, 3, 3)
        )
        gs_cam_i = torch.bmm(rot_wc_masked, gs_world_i.unsqueeze(-1)).squeeze(-1)
        gs_cam_i = F.normalize(gs_cam_i, dim=-1, eps=1e-8)

        valid = torch.isfinite(pred_i).all(dim=-1) & torch.isfinite(gs_cam_i).all(dim=-1)
        if valid.sum() == 0:
            continue

        cos = F.cosine_similarity(pred_i[valid], gs_cam_i[valid], dim=-1).abs()
        total_loss = total_loss + (1.0 - cos).mean()
        valid_frames += 1

    if valid_frames == 0:
        return pred_normals.new_tensor(0.0)
    return total_loss / valid_frames


def compute_gs_normal_loss_against_gt(
    gs_map,
    gt_normals,
    extrinsics,
    bg_mask,
    dy_map,
    gt_valid_mask=None,
    static_only=False,
    softmin_temp=10.0,
    quat_format="xyzw",
):
    """
    Supervise GS-derived normals with GT normals in camera coordinates.
    """
    if gs_map is None or gt_normals is None:
        return None

    gt_normals = F.normalize(gt_normals, dim=-1, eps=1e-8)
    total_loss = 0.0
    valid_frames = 0
    num_frames = gs_map.shape[1]

    for i in range(num_frames):
        mask_i = bg_mask[:, i]
        if static_only:
            mask_i = mask_i & (dy_map[:, i].sigmoid() < 0.5)
        if gt_valid_mask is not None:
            mask_i = mask_i & gt_valid_mask[:, i]
        if mask_i.sum() == 0:
            continue

        gt_i = gt_normals[:, i][mask_i].reshape(-1, 3)
        _, _, scales_i, rot_i = get_split_gs(gs_map[:, i], mask_i)
        gs_world_i = estimate_normals_from_gs(
            scales_i, rot_i, softmin_temp=softmin_temp, quat_format=quat_format
        )

        rot_wc = extrinsics[:, i, :3, :3]
        h_i, w_i = mask_i.shape[-2:]
        rot_wc_masked = (
            rot_wc[:, None, None, :, :]
            .expand(-1, h_i, w_i, -1, -1)[mask_i]
            .reshape(-1, 3, 3)
        )
        gs_cam_i = torch.bmm(rot_wc_masked, gs_world_i.unsqueeze(-1)).squeeze(-1)
        gs_cam_i = F.normalize(gs_cam_i, dim=-1, eps=1e-8)

        valid = torch.isfinite(gt_i).all(dim=-1) & torch.isfinite(gs_cam_i).all(dim=-1)
        if valid.sum() == 0:
            continue

        cos = F.cosine_similarity(gs_cam_i[valid], gt_i[valid], dim=-1).abs()
        total_loss = total_loss + (1.0 - cos).mean()
        valid_frames += 1

    if valid_frames == 0:
        return gs_map.new_tensor(0.0)
    return total_loss / valid_frames

