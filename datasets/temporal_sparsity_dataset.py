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
class TemporalSparsityClipSpec:
    clip_len: int = 30
    query_frame_idx: int = 15

class TemporalSparsityDataset(Dataset):
    """
    Dataset for temporal sparsity analysis experiment.
    
    Extracts 30-frame clips from the dataset.
    For each clip, defines a query frame (index 15) and 2 context frames.
    The temporal distance (d) between the query frame and context frames can be configured.
    
    Levels:
    - d=2: context frames {13, 17}
    - d=5: context frames {10, 20}
    - d=10: context frames {5, 25}
    - d=15: context frames {0, 30}
    """

    def __init__(
        self,
        image_dir,
        temporal_distance,
        scene_names=None,
        dataset_type="orad",
        clip_spec=TemporalSparsityClipSpec(),
        use_normals=True,
        resize_height=TEST_RESIZE_HEIGHT,
    ):
        self.image_dir = image_dir
        self.temporal_distance = int(temporal_distance)
        self.dataset_type = str(dataset_type).lower()
        
        if self.dataset_type not in ("orad", "rellis"):
            raise ValueError(f"Unsupported dataset_type: {dataset_type}. Expected 'orad' or 'rellis'.")

        self.clip_spec = clip_spec if isinstance(clip_spec, TemporalSparsityClipSpec) else TemporalSparsityClipSpec(**dict(clip_spec))
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
            # To match the "0 to 30" index inclusively, we need a 31-frame segment technically.
            # "Extract 30-frame clips" with "index 0 to 30" means 31 frames. Let's make segment size 31 to be safe for d=15.
            # But the prompt says "extract 30-frame clips", so let's stick to seg_len=31 for d=15 to work (0 to 30 = 31 frames).
            required_len = 31
            for seg_start in range(0, n, required_len):
                seg_end = min(seg_start + required_len, n)
                if (seg_end - seg_start) == required_len:
                    self.test_samples.append((scene_idx, seg_start, seg_end))

    def __len__(self):
        return len(self.test_samples)

    def __getitem__(self, idx):
        scene_idx, seg_start, seg_end = self.test_samples[idx]
        full_scene_frames = self.all_scene_data[scene_idx]
        frames = full_scene_frames[seg_start:seg_end]
        n_frames = len(frames)

        scene_name = f"{self.valid_scenes[scene_idx]}_seg_{seg_start:04d}_{seg_end-1:04d}"

        query_idx = self.clip_spec.query_frame_idx
        d = self.temporal_distance
        
        # Calculate context indices based on temporal distance d
        context = [query_idx - d, query_idx + d]
        
        # Ensure indices are within bounds
        context = [max(0, min(c, n_frames - 1)) for c in context]
        targets = [query_idx]

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

        # Normalize timestamps according to the context
        def _norm_ts(fids):
            if len(fids) == 0:
                return torch.zeros(0, dtype=torch.float32)
            t = fids - ctx_fids[0]
            denom = float(ctx_fids[-1] - ctx_fids[0]) if float(ctx_fids[-1] - ctx_fids[0]) > 0 else 1.0
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
