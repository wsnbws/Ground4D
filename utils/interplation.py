import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib

from third_party.TAPIP3D.utils.inference_utils import load_model, read_video, inference, get_grid_queries, resize_depth_bilinear
import numpy as np
import torch
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from collections import defaultdict
def project_point_cloud(point_cloud, other_extrinsic, intrinsic, H, W):
    
    device = point_cloud.device
    dtype = point_cloud.dtype
    N = point_cloud.shape[0]
    
    
    ones = torch.ones((N, 1), device=device, dtype=dtype)
    point_cloud_hom = torch.cat([point_cloud, ones], dim=-1)  # (N, 4)
    
    pts_cam_hom = (other_extrinsic @ point_cloud_hom.T).T  # (N, 4)
    pts_cam = pts_cam_hom[:, :3]  # (N, 3)

    depth = pts_cam[:, 2]  # (N,)
    valid = depth > 0
    if valid.sum() == 0:
        point_map = torch.full((H, W, 3), float('nan'), device=device, dtype=dtype)
        mask = torch.zeros((H, W), device=device, dtype=torch.bool)
        return point_map, mask

    valid_indices = valid.nonzero(as_tuple=True)[0]
    pts_cam_valid = pts_cam[valid_indices]           # (M, 3)
    world_valid   = point_cloud[valid_indices]         # (M, 3) 
    depth_valid   = pts_cam_valid[:, 2]                # (M,)

    pts_norm = pts_cam_valid[:, :2] / pts_cam_valid[:, 2:3]   # (M, 2)

    ones_norm = torch.ones((pts_norm.shape[0], 1), device=device, dtype=dtype)
    pts_norm_hom = torch.cat([pts_norm, ones_norm], dim=-1)     # (M, 3)
    uv_homog = (intrinsic @ pts_norm_hom.T).T  # (M, 3)
    u = uv_homog[:, 0]
    v = uv_homog[:, 1]

    u_int = torch.round(u).long()
    v_int = torch.round(v).long()

    inside = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    if inside.sum() == 0:
        point_map = torch.full((H, W, 3), float('nan'), device=device, dtype=dtype)
        mask = torch.zeros((H, W), device=device, dtype=torch.bool)
        return point_map, mask

    candidate_indices = inside.nonzero(as_tuple=True)[0]
    u_int       = u_int[candidate_indices]
    v_int       = v_int[candidate_indices]
    depth_valid = depth_valid[candidate_indices]
    world_valid = world_valid[candidate_indices]

    flat_pixel_idx = v_int * W + u_int  # (M_valid,)

    perm_depth = torch.argsort(depth_valid, stable=True)
    flat_pixel_idx = flat_pixel_idx[perm_depth]
    depth_valid    = depth_valid[perm_depth]
    world_valid    = world_valid[perm_depth]

    perm_idx = torch.argsort(flat_pixel_idx, stable=True)
    flat_pixel_idx = flat_pixel_idx[perm_idx]
    depth_valid    = depth_valid[perm_idx]
    world_valid    = world_valid[perm_idx]

    unique_flat_idx = torch.unique_consecutive(flat_pixel_idx)
    mask = torch.cat((torch.tensor([True], device=flat_pixel_idx.device), flat_pixel_idx[1:] != flat_pixel_idx[:-1]))
    unique_indices = mask.nonzero(as_tuple=True)[0]

    winner_world = world_valid[unique_indices]  # (num_pixels, 3)

    point_map_flat = torch.full((H * W, 3), float('nan'), device=device, dtype=dtype)
    mask_flat = torch.zeros((H * W,), device=device, dtype=torch.bool)

    point_map_flat[unique_flat_idx] = winner_world
    mask_flat[unique_flat_idx] = True

    point_map = point_map_flat.view(H, W, 3)
    mask = mask_flat.view(H, W)

    return point_map, mask

def filter_dense_points(points, radius=0.2, min_neighbors=10, min_total=50):
    if len(points) < min_total:
        return np.empty((0, 3))

    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    
    if len(points) < min_total:
        return np.empty((0, 3))
    
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius)
    mask = np.array([len(c) > min_neighbors for c in counts])
    filtered = points[mask]
    
    if len(filtered) < min_total:
        return np.empty((0, 3))
    
    return filtered


def interpolate_poses_intrinsics(extrinsics, intrinsics, interval, views, device=None):
    extrinsics_np = extrinsics.detach().cpu().numpy()
    intrinsics_np = intrinsics.detach().cpu().numpy()
    T = extrinsics.shape[0]
    if views == 1:
        # 单视角逻辑不变
        K = T
        extrinsics_out = []
        intrinsics_out = []
        for i in range(K - 1):
            pose0 = extrinsics_np[i]
            pose1 = extrinsics_np[i + 1]
            intr0 = intrinsics_np[i]
            intr1 = intrinsics_np[i + 1]
            R0 = pose0[:3, :3]
            R1 = pose1[:3, :3]
            t0 = pose0[:3, 3]
            t1 = pose1[:3, 3]
            r0 = scipy.spatial.transform.Rotation.from_matrix(R0)
            r1 = scipy.spatial.transform.Rotation.from_matrix(R1)
            slerp = scipy.spatial.transform.Slerp([0, 1], scipy.spatial.transform.Rotation.concatenate([r0, r1]))
            for j in range(interval):
                alpha = j / interval
                R_interp = slerp([alpha]).as_matrix()[0]
                t_interp = (1 - alpha) * t0 + alpha * t1
                pose_interp = np.eye(4)
                pose_interp[:3, :3] = R_interp
                pose_interp[:3, 3] = t_interp
                extrinsics_out.append(pose_interp)
                intr_interp = (1 - alpha) * intr0 + alpha * intr1
                intrinsics_out.append(intr_interp)
        extrinsics_out.append(extrinsics_np[-1])
        intrinsics_out.append(intrinsics_np[-1])
        extrinsics_interp = torch.tensor(extrinsics_out, dtype=torch.float32, device=device)
        intrinsics_interp = torch.tensor(intrinsics_out, dtype=torch.float32, device=device)
        return extrinsics_interp, intrinsics_interp
    else:
        frames_per_view = T // views
        extrinsics_out_views = [[] for _ in range(views)]
        intrinsics_out_views = [[] for _ in range(views)]
        for v in range(views):
            idxs = [v + i * views for i in range(frames_per_view)]
            ext_v = extrinsics_np[idxs]
            int_v = intrinsics_np[idxs]
            for i in range(frames_per_view - 1):
                pose0 = ext_v[i]
                pose1 = ext_v[i + 1]
                intr0 = int_v[i]
                intr1 = int_v[i + 1]
                R0 = pose0[:3, :3]
                R1 = pose1[:3, :3]
                t0 = pose0[:3, 3]
                t1 = pose1[:3, 3]
                r0 = scipy.spatial.transform.Rotation.from_matrix(R0)
                r1 = scipy.spatial.transform.Rotation.from_matrix(R1)
                slerp = scipy.spatial.transform.Slerp([0, 1], scipy.spatial.transform.Rotation.concatenate([r0, r1]))
                for j in range(interval):
                    alpha = j / interval
                    R_interp = slerp([alpha]).as_matrix()[0]
                    t_interp = (1 - alpha) * t0 + alpha * t1
                    pose_interp = np.eye(4)
                    pose_interp[:3, :3] = R_interp
                    pose_interp[:3, 3] = t_interp
                    extrinsics_out_views[v].append(pose_interp)
                    intr_interp = (1 - alpha) * intr0 + alpha * intr1
                    intrinsics_out_views[v].append(intr_interp)
            extrinsics_out_views[v].append(ext_v[-1])
            intrinsics_out_views[v].append(int_v[-1])
        extrinsics_out = []
        intrinsics_out = []
        for i in range(frames_per_view * interval + 1):
            for v in range(views):
                if i < len(extrinsics_out_views[v]):
                    extrinsics_out.append(extrinsics_out_views[v][i])
                    intrinsics_out.append(intrinsics_out_views[v][i])
        extrinsics_out = np.array(extrinsics_out)
        intrinsics_out = np.array(intrinsics_out)
        extrinsics_interp = torch.tensor(extrinsics_out, dtype=torch.float32, device=device)
        intrinsics_interp = torch.tensor(intrinsics_out, dtype=torch.float32, device=device)
        return extrinsics_interp, intrinsics_interp


def smooth_depth(depth_torch, kernel_size=7, sigma=1.0, iterations=2):
    device = depth_torch.device
    dtype = depth_torch.dtype

    mask = torch.isfinite(depth_torch)
    if mask.all():
        ks = kernel_size
        coords = torch.arange(ks, dtype=dtype, device=device) - (ks - 1) / 2.0
        gauss1d = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
        gauss1d = gauss1d / gauss1d.sum()
        kernel2d = gauss1d[:, None] * gauss1d[None, :]
        kernel = kernel2d.unsqueeze(0).unsqueeze(0)
        pad = ks // 2
        depth_in = depth_torch.unsqueeze(0).unsqueeze(0)
        sm = F.conv2d(depth_in, kernel.to(device=device, dtype=dtype), padding=pad)
        return sm.squeeze(0).squeeze(0)

    if mask.sum() == 0:
        return depth_torch
    depth0 = depth_torch.clone()
    depth0[~mask] = 0.0
    depth_in = depth0.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    mask_in = mask.float().unsqueeze(0).unsqueeze(0)
    ks = kernel_size
    coords = torch.arange(ks, dtype=dtype, device=device) - (ks - 1) / 2.0
    gauss1d = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    gauss1d = gauss1d / gauss1d.sum()
    kernel2d = gauss1d[:, None] * gauss1d[None, :]
    kernel = kernel2d.unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(device=device, dtype=dtype)
    pad = ks // 2

    eps = 1e-8
    for _ in range(max(1, iterations)):
        conv_d = F.conv2d(depth_in * mask_in, kernel, padding=pad)
        conv_m = F.conv2d(mask_in, kernel, padding=pad)
        filled = conv_d / (conv_m + eps)
        
        # Ensure dtype match
        if filled.dtype != depth_in.dtype:
            filled = filled.to(depth_in.dtype)

        missing = (mask_in == 0)
        depth_in = depth_in.clone()
        depth_in[missing] = filled[missing]
        mask_in = (mask_in + (conv_m > 0).float()).clamp(max=1.0)
    smoothed = F.conv2d(depth_in, kernel, padding=pad)
    return smoothed.squeeze(0).squeeze(0)

def interp_all(extrinsics, intrinsics, point_map, gs_map, dy_map, gs_conf, bg_mask, images,target_images, depth, track_model, interval=4,views=1):
    depth = depth.squeeze(0).squeeze(-1)
    extrinsics_interp, intrinsics_interp = interpolate_poses_intrinsics(
        extrinsics, intrinsics, interval,views, device=images.device
    )
    depth_interp = None
    
    dy_map_sigmoid = dy_map.sigmoid()
    dy_map_binary = dy_map_sigmoid > 0.95
    dy_map_0 = dy_map_binary[0, 0]

    if dy_map_0.count_nonzero() > 500:
        cluster_eps_local = 0.015
        cluster_min_samples_local = 10
        filter_radius_local = 0.15
        filter_min_neighbors_local = 10
        filter_min_total_local = 50
        sample_per_cluster = 50
        match_max_distance = 0.5

        B, T, H, W, _ = point_map.shape
        assert B == 1, " batch=1"
        device = point_map.device

        frames_clusters = [None] * T
        for i in range(T):
            dynamic_mask = dy_map_binary[0, i, ...]
            if isinstance(dynamic_mask, torch.Tensor):
                dynamic_mask_np = dynamic_mask.detach().cpu().numpy()
            else:
                dynamic_mask_np = np.asarray(dynamic_mask)

            ys, xs = np.nonzero(dynamic_mask_np)
            if ys.size == 0:
                frames_clusters[i] = {'clusters': [], 'pts': np.empty((0,3)), 'labels': np.array([]), 'pixel_idxs': np.array([], dtype=np.int64)}
                continue
            pts_all = point_map[0, i, ys, xs]
            if isinstance(pts_all, torch.Tensor):
                pts_np = pts_all.detach().cpu().numpy().reshape(-1, 3)
            else:
                pts_np = np.asarray(pts_all).reshape(-1, 3)

            flat_idxs = (ys * W + xs).astype(np.int64)

            if pts_np.shape[0] == 0:
                frames_clusters[i] = {'clusters': [], 'pts': np.empty((0,3)), 'labels': np.array([]), 'pixel_idxs': np.array([], dtype=np.int64)}
                continue
            filtered = filter_dense_points(pts_np,
                                           radius=filter_radius_local,
                                           min_neighbors=filter_min_neighbors_local,
                                           min_total=filter_min_total_local)
            to_cluster = filtered if (filtered is not None and filtered.shape[0] > 0) else pts_np
            if to_cluster.shape[0] == pts_np.shape[0]:
                pixel_idxs_for_to_cluster = flat_idxs
            else:
                kd = cKDTree(pts_np)
                dists, nn_idx = kd.query(to_cluster, k=1)
                pixel_idxs_for_to_cluster = flat_idxs[nn_idx]

            if to_cluster.shape[0] == 0:
                frames_clusters[i] = {'clusters': [], 'pts': pts_np, 'labels': np.array([]), 'pixel_idxs': np.array([], dtype=np.int64)}
                continue

            clustering = DBSCAN(eps=cluster_eps_local, min_samples=cluster_min_samples_local).fit(to_cluster)
            labels = clustering.labels_
            unique_labels = sorted(set(labels))
            clusters = []
            for lab in unique_labels:
                if lab == -1:
                    continue
                mask_lab = (labels == lab)
                cluster_pts = to_cluster[mask_lab]
                cluster_pixel_idxs = pixel_idxs_for_to_cluster[mask_lab]
                centroid = cluster_pts.mean(axis=0)
                clusters.append({'pts': cluster_pts, 'centroid': centroid, 'pixel_idxs': cluster_pixel_idxs})
            frames_clusters[i] = {'clusters': clusters, 'pts': to_cluster, 'labels': labels, 'pixel_idxs': pixel_idxs_for_to_cluster}
        tracked_objects = {}

        if track_model is not None and frames_clusters[0]['clusters']:
            first_frame_clusters = frames_clusters[0]['clusters']
            queries_list = []
            cluster_indices = []
            for cidx, cluster in enumerate(first_frame_clusters):
                cluster_pts = cluster['pts']
                k = min(sample_per_cluster, cluster_pts.shape[0])
                if k == 0:
                    continue
                choice = np.random.choice(np.arange(cluster_pts.shape[0]), k, replace=False)
                sampled_pts = cluster_pts[choice]
                frame_idx_col = np.zeros((k, 1), dtype=np.float32)
                query = np.concatenate([frame_idx_col, sampled_pts.astype(np.float32)], axis=1)
                queries_list.append(query)
                cluster_indices.append(cidx)

            if len(queries_list) > 0:
                queries_np = np.concatenate(queries_list, axis=0)
                queries_torch = torch.tensor(queries_np, dtype=torch.float32, device=device)

                device_local = device
                video_arg = images[0].to(device_local).float()
                depth_arg = depth.to(device_local).float() if depth is not None else None
                intrinsics_arg = intrinsics.to(device_local).float()
                extrinsics_arg = extrinsics.to(device_local).float()
                queries_arg = queries_torch.to(device_local).float()

                coords_out = None
                visibs = None
                try:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    coords_out, visibs = inference(
                        model=track_model,
                        video=video_arg,
                        depths=depth_arg,
                        intrinsics=intrinsics_arg,
                        extrinsics=extrinsics_arg,
                        query_point=queries_arg,
                        num_iters=6,
                        grid_size=16,
                        bidrectional=False
                    )
                except Exception as e_float:
                    try:
                        model_device = next(track_model.parameters()).device if any(p.numel() for p in track_model.parameters()) else torch.device("cpu")
                        track_model_cpu = track_model.to("cpu")
                        coords_out, visibs = inference(
                            model=track_model_cpu,
                            video=video_arg.cpu(),
                            depths=(depth_arg.cpu() if depth_arg is not None else None),
                            intrinsics=intrinsics_arg.cpu(),
                            extrinsics=extrinsics_arg.cpu(),
                            query_point=queries_arg.cpu(),
                            num_iters=6,
                            grid_size=16,
                            bidrectional=False
                        )
                        try:
                            track_model.to(model_device)
                        except Exception:
                            pass
                    except Exception as e_cpu:
                        try:
                            if torch.cuda.is_available():
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    coords_out, visibs = inference(
                                        model=track_model,
                                        video=video_arg,
                                        depths=depth_arg,
                                        intrinsics=intrinsics_arg,
                                        extrinsics=extrinsics_arg,
                                        query_point=queries_arg,
                                        num_iters=6,
                                        grid_size=16,
                                        bidrectional=False
                                    )
                        except Exception as e_last:
                            raise
                finally:
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                if coords_out is not None:
                    try:
                        coords_all = coords_out.detach().cpu().numpy()
                    except Exception:
                        coords_all = coords_out
                    visibs_np = visibs.detach().cpu().numpy() if visibs is not None else None
                else:
                    coords_all = None
                    visibs_np = None
                

                if coords_all is not None:
                    acc = 0
                    for cidx in cluster_indices:
                        k = queries_list[cluster_indices.index(cidx)].shape[0]
                        pts_tracked = coords_all[:, acc:acc+k, :]
                        pt_start = pts_tracked[0, :, :]
                        
                        centroids_per_frame = pts_tracked.mean(axis=1)
                        motion_vectors_per_frame = []
                        
                        for t_idx in range(1, T):
                            centroid_t = centroids_per_frame[t_idx - 1]
                            centroid_next = centroids_per_frame[t_idx]
                            motion_vec = centroid_next - centroid_t  # (3,)
                            motion_vectors_per_frame.append(motion_vec)
                        
                        tracked_objects[cidx] = {
                            'pred_centroids': centroids_per_frame,
                            'motion_vectors': motion_vectors_per_frame,
                            'queries_start': pt_start
                        }
                        acc += k
        cluster_assignments = [{} for _ in range(T)]
        obj_to_cluster = {oid: {0: 0} for oid in tracked_objects.keys()} 
        if tracked_objects:
            for t in range(1, T):
                frame_info = frames_clusters[t]
                if not frame_info['clusters']:
                    continue
                centroids_t = np.stack([c['centroid'] for c in frame_info['clusters']])
                obj_pred_centroids = {obj_id: obj_info['pred_centroids'][t] for obj_id, obj_info in tracked_objects.items() if not np.any(np.isnan(obj_info['pred_centroids'][t]))}
                if not obj_pred_centroids:
                    continue
                obj_ids = list(obj_pred_centroids.keys())
                preds = np.stack([obj_pred_centroids[oid] for oid in obj_ids])
                cost = np.linalg.norm(preds[:, None, :] - centroids_t[None, :, :], axis=2)
                assigned_obj = set()
                for cidx in range(centroids_t.shape[0]):
                    obj_dist = cost[:, cidx]
                    min_idx = np.argmin(obj_dist)
                    min_dist = obj_dist[min_idx]
                    if min_dist < match_max_distance:
                        oid = obj_ids[min_idx]
                        if cidx not in cluster_assignments[t]:
                            cluster_assignments[t][cidx] = oid
                            assigned_obj.add(oid)
                            obj_to_cluster[oid][t] = cidx
            B_new = point_map.shape[0]
            T_new = (T - 1) * interval + 1
            H_new = point_map.shape[2]
            W_new = point_map.shape[3]
            point_map_interp = torch.full((B_new, T_new, H_new, W_new, 3), float('nan'), 
                                           dtype=point_map.dtype, device=point_map.device)
            dy_map_interp = torch.full((B_new, T_new, H_new, W_new), float('nan'), 
                                        dtype=dy_map.dtype, device=dy_map.device)
            point_obj_id = [dict() for _ in range(T)]

        for t in range(T):
            frame_info = frames_clusters[t]
            if not frame_info['clusters']:
                continue
            for cidx, cluster in enumerate(frame_info['clusters']):
                if t == 0:
                    oid = cidx
                else:
                    oid = cluster_assignments[t].get(cidx, None)
                if oid is None:
                    continue
                cluster_pixel_idxs = cluster.get('pixel_idxs', None)
                if cluster_pixel_idxs is None or cluster_pixel_idxs.size == 0:
                    continue
                for flat_idx in cluster_pixel_idxs:
                    point_obj_id[t][int(flat_idx)] = oid
        
        for t in range(T):
            frame_info = frames_clusters[t]
            dynamic_mask_t = dy_map_binary[0, t, ...]  # (H, W)
            point_map_interp[0, t*interval, :, :, :] = point_map[0, t, :, :, :]
            dy_map_interp[0, t*interval, :, :] = dy_map[0, t, :, :]
            
            if t < T - 1:
                frame_info_next = frames_clusters[t + 1]
                
                for j in range(1, interval):
                    alpha = float(j) / float(interval)
                    t_interp = t * interval + j
                    
                    point_map_interp[0, t_interp, :, :, :] = float('nan')
                    dy_map_interp[0, t_interp, :, :] = 0
                    src_pts = point_map[0, t, :, :, :]
                    valid_src = torch.isfinite(src_pts).all(dim=-1) 
                    if valid_src.any():
                        point_map_interp[0, t_interp][valid_src] = src_pts[valid_src]
                    if frame_info['clusters']:
                        for cidx, cluster_t in enumerate(frame_info['clusters']):
                            if t == 0:
                                oid = cidx
                            else:
                                oid = cluster_assignments[t].get(cidx, None)
                            
                            if oid is None or oid not in tracked_objects:
                                continue
                            
                            motion_vec_t_to_next = tracked_objects[oid]['motion_vectors'][t]
                            
                            cluster_pixel_idxs = cluster_t.get('pixel_idxs', np.array([], dtype=np.int64))
                            if cluster_pixel_idxs is None or cluster_pixel_idxs.size == 0:
                                continue

                            for flat_idx in cluster_pixel_idxs:
                                flat_idx = int(flat_idx)
                                v_src = flat_idx // W
                                u_src = flat_idx % W
                                
                                pt_world_t = point_map[0, t, v_src, u_src]
                                if not torch.isfinite(pt_world_t).all():
                                    continue

                                mv = torch.tensor(motion_vec_t_to_next, dtype=pt_world_t.dtype, device=device)
                                pt_interp = pt_world_t + alpha * mv  # (3,)

                                point_map_interp[0, t_interp, v_src, u_src, :] = pt_interp
                                dy_map_interp[0, t_interp, v_src, u_src] = 1.0
        point_map = point_map_interp
        dy_map = dy_map_interp
        
        pred_flows = None
        flow_masks = None
    else:
        T_orig = dy_map.shape[1]
        T_new_local = (T_orig - 1) * interval + 1
        dy_map = dy_map.repeat_interleave(interval, dim=1)[:, :T_new_local, ...]
        point_map = point_map.repeat_interleave(interval, dim=1)[:, :T_new_local, ...]
        pred_flows = None
        flow_masks = None
    B_new, T_new, H_new, W_new, _ = point_map.shape
    depth_interp = torch.full((B_new, T_new, H_new, W_new), float('nan'),
                              device=point_map.device, dtype=point_map.dtype)

    for b in range(B_new):
        for t in range(T_new):
            pts_flat = point_map[b, t].reshape(-1, 3)
            extr_t = extrinsics_interp[t].to(device=point_map.device, dtype=point_map.dtype)
            intr_t = intrinsics_interp[t].to(device=point_map.device, dtype=point_map.dtype)

            depth_map_frame, _ = project_point_cloud(pts_flat, extr_t, intr_t, H_new, W_new)
            depth_frame = depth_map_frame[:, :, 2]

            depth_filled = smooth_depth(depth_frame, kernel_size=5, sigma=1.0, iterations=2)
            depth_interp[b, t] = depth_filled

    final_T = point_map.shape[1]
    gs_map = gs_map.repeat_interleave(interval, dim=1)[:, :final_T, ...]

    return (
        extrinsics_interp,
        intrinsics_interp,
        point_map,
        gs_map,
        dy_map,
        gs_conf.repeat_interleave(interval, dim=1)[:, :point_map.shape[1], ...],
        bg_mask.repeat_interleave(interval, dim=1)[:, :point_map.shape[1], ...],
        target_images,
        pred_flows,
        flow_masks,
        depth_interp
    )
