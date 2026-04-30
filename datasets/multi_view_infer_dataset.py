import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.orad_dataset import (
    TEST_RESIZE_HEIGHT,
    load_and_preprocess_binary_masks,
    load_and_preprocess_images,
    load_and_preprocess_normals,
)


@dataclass(frozen=True)
class ClipSpec:
    clip_len: int = 70
    start_idx: int = 0
    base_interval: int = 20
    n_inter_frames: int = 3  # insert 3 frames between two anchors


def _unique_sorted_int(xs):
    return sorted({int(x) for x in xs})


def _build_targets_for_anchors(anchors, n_inter_frames):
    if len(anchors) < 2:
        return anchors
    targets = []
    denom = int(n_inter_frames) + 1
    for a, b in zip(anchors[:-1], anchors[1:]):
        step = (b - a) / float(denom)
        for k in range(denom + 1):
            targets.append(int(round(a + k * step)))
    return _unique_sorted_int(targets)


def _pick_evenly_from_candidates(candidates, k):
    if k <= 0 or not candidates:
        return []
    if len(candidates) <= k:
        return list(candidates)
    ticks = np.linspace(0, len(candidates) - 1, k + 2)[1:-1]
    idxs = np.unique(np.round(ticks).astype(int)).tolist()
    picked = [candidates[i] for i in idxs]
    if len(picked) < k:
        for c in candidates:
            if c not in picked:
                picked.append(c)
            if len(picked) == k:
                break
    return picked[:k]


def _select_context_indices(
    anchors,
    targets,
    num_context_views,
):
    anchors = _unique_sorted_int(anchors)
    targets_set = set(int(x) for x in targets)
    if num_context_views <= len(anchors):
        return anchors[:num_context_views]

    extra_needed = int(num_context_views) - len(anchors)
    n_intervals = max(0, len(anchors) - 1)
    if n_intervals == 0:
        return anchors

    base = extra_needed // n_intervals
    rem = extra_needed % n_intervals

    extras = []
    for i, (a, b) in enumerate(zip(anchors[:-1], anchors[1:])):
        k_i = base + (1 if i < rem else 0)
        if k_i <= 0:
            continue
        candidates = [t for t in range(a + 1, b) if t not in targets_set and t not in anchors]
        extras.extend(_pick_evenly_from_candidates(candidates, k_i))

    context = _unique_sorted_int(list(anchors) + list(extras))
    if len(context) > num_context_views:
        context = context[:num_context_views]
    return context


class MultiViewInferDataset(Dataset):
    """
    Test-only dataset pipeline for interpolation (mode=3).

    - A scene is split into fixed-length clips (default 70 frames), keeping only full clips.
    - For each clip, we build:
      - fixed interpolation targets from 4 anchor frames spaced by base_interval (default 20),
        with n_inter_frames inserted between anchors (default 3) -> step 5.
      - context frames for ablation: 4/7/10 views, by adding extra frames inside anchor gaps
        while keeping targets unchanged.
    """

    def __init__(
        self,
        image_dir,
        scene_names=None,
        dataset_type="orad",
        num_context_views=4,
        clip_spec=ClipSpec(),
        use_normals=True,
        resize_height=TEST_RESIZE_HEIGHT,
    ):
        self.image_dir = image_dir
        self.dataset_type = str(dataset_type).lower()
        if self.dataset_type not in ("orad", "rellis"):
            raise ValueError(f"Unsupported dataset_type: {dataset_type}. Expected 'orad' or 'rellis'.")

        self.num_context_views = int(num_context_views)
        self.clip_spec = clip_spec if isinstance(clip_spec, ClipSpec) else ClipSpec(**dict(clip_spec))
        self.use_normals = bool(use_normals)
        self.resize_height = resize_height

        if scene_names is None:
            self.scene_list = sorted(
                [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
            )
        else:
            self.scene_list = list(scene_names)

        self.all_scene_data = []
        self.valid_scenes = []
        for scene in self.scene_list:
            scene_root = os.path.join(image_dir, scene)
            if self.dataset_type == "rellis":
                img_dir = os.path.join(scene_root, "pylon_camera_node")
                if not os.path.isdir(img_dir):
                    continue
                img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
            else:
                img_dir = os.path.join(scene_root, "image_data")
                if not os.path.isdir(img_dir):
                    continue
                img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

            if not img_files:
                continue

            scene_frames = []
            for fid, f in enumerate(img_files):
                if self.dataset_type == "rellis":
                    stem = os.path.splitext(f)[0]
                    frame_info = {"fid": fid, "img": os.path.join(img_dir, f)}
                    if self.use_normals:
                        frame_info["normal"] = os.path.join(scene_root, "normal_map", f"{stem}.npy")
                        frame_info["normal_valid"] = os.path.join(scene_root, "normal_valid_mask", f"{stem}.png")
                else:
                    stem = f.replace(".png", "")
                    frame_info = {
                        "fid": fid,
                        "img": os.path.join(img_dir, f),
                        "sky": os.path.join(scene_root, "sky_masks", f.replace(".png", "_fillcolor.png")),
                        "dyn": os.path.join(scene_root, "dynamic_masks", f),
                    }
                    if self.use_normals:
                        frame_info["normal"] = os.path.join(scene_root, "normal_map", f"{stem}.npy")
                        frame_info["normal_valid"] = os.path.join(scene_root, "normal_valid_mask", f"{stem}.png")
                scene_frames.append(frame_info)

            self.all_scene_data.append(scene_frames)
            self.valid_scenes.append(scene)

        self.test_samples = []
        seg_len = int(self.clip_spec.clip_len)
        for scene_idx, frames in enumerate(self.all_scene_data):
            n = len(frames)
            for seg_start in range(0, n, seg_len):
                seg_end = min(seg_start + seg_len, n)
                if (seg_end - seg_start) == seg_len:
                    self.test_samples.append((scene_idx, seg_start, seg_end))

    def __len__(self):
        return len(self.test_samples)

    def __getitem__(self, idx):
        scene_idx, seg_start, seg_end = self.test_samples[idx]
        full_scene_frames = self.all_scene_data[scene_idx]
        frames = full_scene_frames[seg_start:seg_end]
        n_frames = len(frames)

        scene_name = f"{self.valid_scenes[scene_idx]}_seg_{seg_start:04d}_{seg_end-1:04d}"

        start = int(self.clip_spec.start_idx)
        base_interval = int(self.clip_spec.base_interval)
        n_inter_frames = int(self.clip_spec.n_inter_frames)

        anchors = [start + k * base_interval for k in range(4)]
        anchors = [min(max(0, a), n_frames - 1) for a in anchors]
        anchors = _unique_sorted_int(anchors)

        targets = _build_targets_for_anchors(anchors, n_inter_frames)
        targets = [t for t in targets if 0 <= t < n_frames]
        targets = _unique_sorted_int(targets)

        context = _select_context_indices(
            anchors=anchors,
            targets=targets,
            num_context_views=self.num_context_views,
        )
        context = [c for c in context if 0 <= c < n_frames]
        context = _unique_sorted_int(context)

        context_frames = [frames[i] for i in context]
        target_frames = [frames[i] for i in targets]

        img_paths = [s["img"] for s in context_frames]
        images = load_and_preprocess_images(img_paths, resize_height=self.resize_height)

        sky_paths = [s.get("sky") for s in context_frames]
        actual_sky_paths = [p for p in sky_paths if p is not None and os.path.exists(p)]
        if len(actual_sky_paths) == len(img_paths):
            masks = load_and_preprocess_images(actual_sky_paths, resize_height=self.resize_height)
        else:
            masks = torch.zeros_like(images)

        target_img_paths = [s["img"] for s in target_frames]
        target_images = load_and_preprocess_images(target_img_paths, resize_height=self.resize_height)

        target_sky_paths = [s.get("sky") for s in target_frames]
        actual_target_sky_paths = [p for p in target_sky_paths if p is not None and os.path.exists(p)]
        if len(actual_target_sky_paths) == len(target_img_paths):
            target_masks = load_and_preprocess_images(actual_target_sky_paths, resize_height=self.resize_height)
        else:
            target_masks = torch.zeros_like(target_images)

        ctx_fids = np.array([context_frames[i]["fid"] for i in range(len(context_frames))], dtype=np.float32)
        tgt_fids = np.array([target_frames[i]["fid"] for i in range(len(target_frames))], dtype=np.float32)

        def _norm_ts(fids):
            if len(fids) == 0:
                return torch.zeros(0, dtype=torch.float32)
            t = fids - fids[0]
            denom = float(t[-1]) if float(t[-1]) > 0 else 1.0
            return torch.from_numpy((t / denom).astype(np.float32))

        timestamps = _norm_ts(ctx_fids)
        target_timestamps = _norm_ts(tgt_fids)

        intervals = []
        if len(context) >= 2:
            intervals = [int(context[i + 1] - context[i]) for i in range(len(context) - 1)]

        out = {
            "images": images,
            "masks": masks,
            "targets": target_images,
            "target_masks": target_masks,
            "image_paths": img_paths,
            "target_image_paths": target_img_paths,
            "timestamps": timestamps,
            "target_timestamps": target_timestamps,
            "interval": intervals,
            "scene_name": scene_name,
            "clip_start": int(seg_start),
            "context_indices": np.array(context, dtype=np.int64),
            "target_indices": np.array(targets, dtype=np.int64),
        }

        print(f"[{scene_name}] context_indices: {context}")
        print(f"[{scene_name}] target_indices:  {targets}")

        dyn_paths = [s.get("dyn") for s in target_frames]
        actual_dyn_paths = [p for p in dyn_paths if p is not None and os.path.exists(p)]
        if len(actual_dyn_paths) == len(target_img_paths):
            out["dynamic_mask"] = load_and_preprocess_images(actual_dyn_paths, resize_height=self.resize_height)

        if self.use_normals:
            normal_paths = [s.get("normal") for s in target_frames]
            actual_normal_paths = [p for p in normal_paths if p is not None and os.path.exists(p)]
            if len(actual_normal_paths) == len(target_img_paths):
                out["normals"] = load_and_preprocess_normals(actual_normal_paths, resize_height=self.resize_height)

            normal_valid_paths = [s.get("normal_valid") for s in target_frames]
            actual_normal_valid_paths = [p for p in normal_valid_paths if p is not None and os.path.exists(p)]
            if len(actual_normal_valid_paths) == len(target_img_paths):
                out["normal_valid_mask"] = load_and_preprocess_binary_masks(
                    actual_normal_valid_paths, resize_height=self.resize_height
                )

        return out

