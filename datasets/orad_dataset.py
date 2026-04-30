import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as TF

TARGET_SIZE = 518
TEST_RESIZE_HEIGHT = 256
PATCH_MULTIPLE = 14


def _compute_resize_and_crop(h, w, target_size=TARGET_SIZE):
    nw = target_size
    nh = round(h * (nw / w) / 14) * 14
    sy = (nh - target_size) // 2 if nh > target_size else 0
    return nw, nh, sy


def _resize_and_center_crop_tensor(chw, target_size=TARGET_SIZE, mode="bilinear"):
    _, h, w = chw.shape
    nw, nh, sy = _compute_resize_and_crop(h, w, target_size)
    align_corners = False if mode in ("bilinear", "bicubic") else None
    x = F.interpolate(
        chw.unsqueeze(0),
        size=(nh, nw),
        mode=mode,
        align_corners=align_corners
    ).squeeze(0)
    if nh > target_size:
        x = x[:, sy:sy + target_size, :]
    return x


def _round_to_multiple(value, multiple=PATCH_MULTIPLE):
    return max(multiple, int(round(value / multiple) * multiple))


def _resize_keep_aspect_tensor(chw, target_height, mode="bilinear"):
    _, h, w = chw.shape
    if h <= 0 or w <= 0:
        return chw
    aligned_h = _round_to_multiple(target_height, PATCH_MULTIPLE)
    raw_w = w * (aligned_h / h)
    target_width = _round_to_multiple(raw_w, PATCH_MULTIPLE)
    align_corners = False if mode in ("bilinear", "bicubic") else None
    return F.interpolate(
        chw.unsqueeze(0),
        size=(aligned_h, target_width),
        mode=mode,
        align_corners=align_corners
    ).squeeze(0)


def load_and_preprocess_images(path_list, resize_height=None):
    """
    通用预处理函数：缩放、填充、中心裁剪，确保高度是 14 的倍数。
    """
    if not path_list: 
        return torch.tensor([])
    images = []
    to_tensor = TF.ToTensor()
    for p in path_list:
        if not os.path.exists(p):
            # print(f"Warning: Image not found: {p}")
            continue
            
        try:
            img = Image.open(p)
            if img.mode == "RGBA":
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img)
            img = img.convert("RGB")
            w, h = img.size
            if resize_height is not None:
                aligned_h = _round_to_multiple(resize_height, PATCH_MULTIPLE)
                nw = _round_to_multiple(w * (aligned_h / h), PATCH_MULTIPLE)
                img = img.resize((nw, aligned_h), Image.Resampling.BICUBIC)
                it = to_tensor(img)
            else:
                nw, nh, sy = _compute_resize_and_crop(h, w, TARGET_SIZE)
                img = img.resize((nw, nh), Image.Resampling.BICUBIC)
                it = to_tensor(img)
                if nh > TARGET_SIZE:
                    it = it[:, sy:sy + TARGET_SIZE, :]
            images.append(it)
        except Exception as e:
            print(f"Error loading image {p}: {e}")
            continue
    
    if not images:
        return torch.tensor([])
        
    # 确保序列中形状统一
    # shapes = set(i.shape for i in images)
    # if len(shapes) > 1:
    #     mh = max(i.shape[1] for i in images)
    #     mw = max(i.shape[2] for i in images)
    #     images = [torch.nn.functional.pad(i, (0, mw-i.shape[2], 0, mh-i.shape[1]), value=1.0) for i in images]
    
    return torch.stack(images)


def load_and_preprocess_normals(path_list, resize_height=None):
    """Load normal_map/*.npy and preprocess to [S, 3, H, W]."""
    if not path_list:
        return torch.tensor([])
    normals = []
    for p in path_list:
        if not os.path.exists(p):
            continue
        try:
            arr = np.load(p).astype(np.float32)  # [H, W, 3], expected in [-1, 1]
            if arr.ndim != 3 or arr.shape[-1] != 3:
                continue
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3, H, W]
            if resize_height is not None:
                t = _resize_keep_aspect_tensor(t, target_height=resize_height, mode="bilinear")
            else:
                t = _resize_and_center_crop_tensor(t, target_size=TARGET_SIZE, mode="bilinear")
            t = F.normalize(t, dim=0, eps=1e-6)
            normals.append(t)
        except Exception as e:
            print(f"Error loading normal {p}: {e}")
            continue
    if not normals:
        return torch.tensor([])
    return torch.stack(normals)


def load_and_preprocess_binary_masks(path_list, resize_height=None):
    """Load binary masks to [S, 1, H, W], values in {0,1}."""
    if not path_list:
        return torch.tensor([])
    masks = []
    for p in path_list:
        if not os.path.exists(p):
            continue
        try:
            arr = np.array(Image.open(p))
            if arr.ndim == 3:
                arr = arr[..., 0]
            t = torch.from_numpy((arr > 0).astype(np.float32)).unsqueeze(0)  # [1, H, W]
            if resize_height is not None:
                t = _resize_keep_aspect_tensor(t, target_height=resize_height, mode="nearest")
            else:
                t = _resize_and_center_crop_tensor(t, target_size=TARGET_SIZE, mode="nearest")
            masks.append((t > 0.5).float())
        except Exception as e:
            print(f"Error loading mask {p}: {e}")
            continue
    if not masks:
        return torch.tensor([])
    return torch.stack(masks)

class OradDataset(Dataset):
    """
    ORAD 数据集 (单视角版本)：
    - 原图像: image_data/<unix_time>.png
    - 天空掩码: sky_masks/<unix_time>_fillcolor.png
    - 动态掩码: dynamic_masks/<unix_time>.png
    """
    def __init__(
        self,
        image_dir,
        scene_names=None,
        sequence_length=4,
        mode=1,
        start_idx=0,
        interval=1,
        n_inter_frames=None,
        use_normals=True,
        test_segment_length=70,
        dataset_type="orad",
    ):
        """
        Args:
            image_dir: 数据根目录
            scene_names: 场景列表，None 则扫描全目录
            sequence_length: 序列长度
            mode: 1-训练, 2-重建(测试), 3-插值
            start_idx: 测试时的起始帧索引
            interval: 帧间隔
            n_inter_frames: 指定中间插值帧的数量
        """
        self.image_dir = image_dir
        self.sequence_length = sequence_length
        self.mode = mode
        self.start_idx = start_idx
        self.interval = interval
        self.n_inter_frames = n_inter_frames
        self.use_normals = use_normals
        self.test_segment_length = test_segment_length
        self.dataset_type = str(dataset_type).lower()
        if self.dataset_type not in ("orad", "rellis"):
            raise ValueError(f"Unsupported dataset_type: {dataset_type}. Expected 'orad' or 'rellis'.")
        
        if scene_names is None:
            # 扫描目录下所有的场景文件夹
            self.scene_list = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
        else:
            self.scene_list = scene_names
            
        self.all_scene_data = []
        self.valid_scenes = []
        
        for scene in self.scene_list:
            scene_root = os.path.join(image_dir, scene)
            if self.dataset_type == "rellis":
                img_dir = os.path.join(scene_root, "pylon_camera_node")
                sky_dir = None
                dyn_dir = None
            else:
                img_dir = os.path.join(scene_root, "image_data")
                sky_dir = os.path.join(scene_root, "sky_masks")
                dyn_dir = os.path.join(scene_root, "dynamic_masks")
            
            if not os.path.isdir(img_dir):
                continue
            
            if self.dataset_type == "rellis":
                # Rellis-3D: pylon_camera_node/*.jpg
                img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
            else:
                # ORAD: image_data/*.png
                img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            if len(img_files) < sequence_length:
                continue
            
            scene_frames = []
            for fid, f in enumerate(img_files):
                if self.dataset_type == "rellis":
                    stem = os.path.splitext(f)[0]
                    frame_info = {
                        "fid": fid,
                        "img": os.path.join(img_dir, f),
                    }
                    if self.use_normals:
                        frame_info["normal"] = os.path.join(scene_root, "normal_map", f"{stem}.npy")
                        frame_info["normal_valid"] = os.path.join(scene_root, "normal_valid_mask", f"{stem}.png")
                else:
                    stem = f.replace('.png', '')
                    frame_info = {
                        "fid": fid,
                        "img": os.path.join(img_dir, f),
                        "sky": os.path.join(sky_dir, f.replace('.png', '_fillcolor.png')),
                        "dyn": os.path.join(dyn_dir, f),
                    }
                    if self.use_normals:
                        frame_info["normal"] = os.path.join(scene_root, "normal_map", f"{stem}.npy")
                        frame_info["normal_valid"] = os.path.join(scene_root, "normal_valid_mask", f"{stem}.png")
                scene_frames.append(frame_info)
            self.all_scene_data.append(scene_frames)
            self.valid_scenes.append(scene)

        # For test modes, build sub-segment samples (e.g., 70-frame chunks)
        self.test_samples = []
        if self.mode in (2, 3):
            seg_len = int(self.test_segment_length) if self.test_segment_length is not None else 0
            for scene_idx, frames in enumerate(self.all_scene_data):
                n = len(frames)
                if seg_len <= 0:
                    self.test_samples.append((scene_idx, 0, n))
                    continue
                for seg_start in range(0, n, seg_len):
                    seg_end = min(seg_start + seg_len, n)
                    # Keep only full segments (e.g., exactly 70 frames); drop tail leftovers.
                    if (seg_end - seg_start) == seg_len:
                        self.test_samples.append((scene_idx, seg_start, seg_end))

    def __len__(self):
        if self.mode in (2, 3):
            return len(self.test_samples)
        return len(self.all_scene_data)

    def __getitem__(self, idx):
        resize_height = TEST_RESIZE_HEIGHT if self.mode in (2, 3) else None

        if self.mode in (2, 3):
            scene_idx, seg_start, seg_end = self.test_samples[idx]
            full_scene_frames = self.all_scene_data[scene_idx]
            scene_frames = full_scene_frames[seg_start:seg_end]
            base_scene_name = self.valid_scenes[scene_idx]
            scene_name = f"{base_scene_name}_seg_{seg_start:04d}_{seg_end-1:04d}"
        else:
            scene_frames = self.all_scene_data[idx]
            scene_name = self.valid_scenes[idx]
        n_frames = len(scene_frames)
        
        if self.mode == 3:
            # 插值模式
            start_idx = self.start_idx
            # 检查是否有足够的帧
            max_idx = start_idx + (self.sequence_length - 1) * self.interval
            if max_idx >= n_frames:
                # 尝试调整 start_idx
                start_idx = max(0, n_frames - 1 - (self.sequence_length - 1) * self.interval)
            
            # 输入帧索引
            indices = [start_idx + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            
            # 目标帧索引 (Ground Truth): 包含输入帧和中间插值帧
            target_indices = []
            for i in range(len(indices) - 1):
                start = indices[i]
                end = indices[i+1]
                target_indices.append(start)
                
                inter_list = list(range(start, end+1))
                if self.n_inter_frames is not None and self.n_inter_frames < len(inter_list):
                    sample_ticks = np.linspace(0, len(inter_list)-1, self.n_inter_frames + 2)[1:-1]
                    sample_idxs = np.round(sample_ticks).astype(int)
                    for k in sample_idxs:
                        target_indices.append(inter_list[k])
                else:
                    target_indices.extend(inter_list)
            target_indices.append(indices[-1])
            
            # 边界检查
            indices = [min(i, n_frames - 1) for i in indices]
            target_indices = [min(i, n_frames - 1) for i in target_indices]

            selected_frames = [scene_frames[i] for i in indices]
            target_selected_frames = [scene_frames[i] for i in target_indices]

            # 加载输入
            img_paths = [s["img"] for s in selected_frames]
            sky_paths = [s.get("sky") for s in selected_frames]
            images = load_and_preprocess_images(img_paths, resize_height=resize_height)
            
            actual_sky_paths = [p for p in sky_paths if p is not None and os.path.exists(p)]
            if len(actual_sky_paths) == len(img_paths):
                masks = load_and_preprocess_images(actual_sky_paths, resize_height=resize_height)
            else:
                masks = torch.zeros_like(images)

            # 加载目标
            target_img_paths = [s["img"] for s in target_selected_frames]
            target_sky_paths = [s.get("sky") for s in target_selected_frames]
            target_images = load_and_preprocess_images(target_img_paths, resize_height=resize_height)
            
            actual_target_sky_paths = [p for p in target_sky_paths if p is not None and os.path.exists(p)]
            if len(actual_target_sky_paths) == len(target_img_paths):
                target_masks = load_and_preprocess_images(actual_target_sky_paths, resize_height=resize_height)
            else:
                target_masks = torch.zeros_like(target_images)

            # 时间戳
            global_indices = np.array([selected_frames[i]["fid"] for i in range(len(selected_frames))], dtype=np.float32)
            timestamps = global_indices - global_indices[0]
            if timestamps[-1] > 0:
                # timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
                timestamps = timestamps / timestamps[-1]
            else:
                timestamps = timestamps.astype(np.float32)

            input_dict = {
                "images": images,
                "masks": masks,
                "targets": target_images,
                "target_masks": target_masks,
                "image_paths": img_paths,
                "timestamps": timestamps,
                "interval": intervals,
                "scene_name": scene_name,
                "start_idx": start_idx
            }
            
            # 动态 mask (使用 target 的动态 mask)
            target_dyn_paths = [s.get("dyn") for s in target_selected_frames]
            actual_target_dyn_paths = [p for p in target_dyn_paths if p is not None and os.path.exists(p)]
            if len(actual_target_dyn_paths) == len(target_img_paths):
                 input_dict["dynamic_mask"] = load_and_preprocess_images(actual_target_dyn_paths, resize_height=resize_height)
            if self.use_normals:
                target_normal_paths = [s["normal"] for s in target_selected_frames]
                actual_target_normal_paths = [p for p in target_normal_paths if os.path.exists(p)]
                if len(actual_target_normal_paths) == len(target_img_paths):
                    input_dict["normals"] = load_and_preprocess_normals(actual_target_normal_paths, resize_height=resize_height)
                target_normal_valid_paths = [s["normal_valid"] for s in target_selected_frames]
                actual_target_normal_valid_paths = [p for p in target_normal_valid_paths if os.path.exists(p)]
                if len(actual_target_normal_valid_paths) == len(target_img_paths):
                    input_dict["normal_valid_mask"] = load_and_preprocess_binary_masks(actual_target_normal_valid_paths, resize_height=resize_height)
            
            return input_dict

        elif self.mode == 2:
            # 重建模式
            start_idx = self.start_idx
            indices = [start_idx + i * self.interval for i in range(self.sequence_length)]
            # 防止越界
            indices = [min(i, n_frames - 1) for i in indices]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            
            selected_frames = [scene_frames[i] for i in indices]
            
            img_paths = [s["img"] for s in selected_frames]
            sky_paths = [s.get("sky") for s in selected_frames]
            dyn_paths = [s.get("dyn") for s in selected_frames]
            
            global_indices = np.array([selected_frames[i]["fid"] for i in range(len(selected_frames))], dtype=np.float32)
            timestamps = global_indices - global_indices[0]
            if timestamps[-1] > 0:
                timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            else:
                timestamps = timestamps.astype(np.float32)

            images = load_and_preprocess_images(img_paths, resize_height=resize_height)
            
            actual_sky_paths = [p for p in sky_paths if p is not None and os.path.exists(p)]
            if len(actual_sky_paths) == len(img_paths):
                masks = load_and_preprocess_images(actual_sky_paths, resize_height=resize_height)
            else:
                masks = torch.zeros_like(images)

            input_dict = {
                "images": images,
                "masks": masks,
                "image_paths": img_paths,
                "timestamps": timestamps,
                "interval": intervals,
                "scene_name": scene_name,
                "start_idx": start_idx
            }
            
            actual_dyn_paths = [p for p in dyn_paths if p is not None and os.path.exists(p)]
            if len(actual_dyn_paths) == len(img_paths):
                input_dict["dynamic_mask"] = load_and_preprocess_images(actual_dyn_paths, resize_height=resize_height)
            if self.use_normals:
                normal_paths = [s["normal"] for s in selected_frames]
                actual_normal_paths = [p for p in normal_paths if os.path.exists(p)]
                if len(actual_normal_paths) == len(img_paths):
                    input_dict["normals"] = load_and_preprocess_normals(actual_normal_paths, resize_height=resize_height)
                normal_valid_paths = [s["normal_valid"] for s in selected_frames]
                actual_normal_valid_paths = [p for p in normal_valid_paths if os.path.exists(p)]
                if len(actual_normal_valid_paths) == len(img_paths):
                    input_dict["normal_valid_mask"] = load_and_preprocess_binary_masks(actual_normal_valid_paths, resize_height=resize_height)
                
            return input_dict
            
        else: # mode == 1 训练
            inter_step = random.randint(21, 51)
            start_idx = random.randint(0, max(0, n_frames - inter_step))
            intervals = sorted(random.sample(range(1, inter_step-1), self.sequence_length - 1))
            indices = [start_idx]
            for interval in intervals:
                indices.append(min(start_idx + interval, n_frames - 1))
            
            selected_frames = [scene_frames[i] for i in indices]
            
            img_paths = [s["img"] for s in selected_frames]
            sky_paths = [s.get("sky") for s in selected_frames]
            dyn_paths = [s.get("dyn") for s in selected_frames]
            
            timestamps = np.array(indices) - indices[0]
            if timestamps[-1] > 0:
                timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            else:
                timestamps = timestamps.astype(np.float32)
                
            images = load_and_preprocess_images(img_paths)
            
            actual_sky_paths = [p for p in sky_paths if p is not None and os.path.exists(p)]
            if len(actual_sky_paths) == len(img_paths):
                masks = load_and_preprocess_images(actual_sky_paths)
            else:
                masks = torch.zeros_like(images)

            input_dict = {
                "images": images,
                "masks": masks,
                "image_paths": img_paths,
                "timestamps": timestamps,
                "interval": intervals,
                "scene_name": scene_name,
                "start_idx": start_idx
            }
            
            actual_dyn_paths = [p for p in dyn_paths if p is not None and os.path.exists(p)]
            if len(actual_dyn_paths) == len(img_paths):
                input_dict["dynamic_mask"] = load_and_preprocess_images(actual_dyn_paths)
            if self.use_normals:
                normal_paths = [s["normal"] for s in selected_frames]
                actual_normal_paths = [p for p in normal_paths if os.path.exists(p)]
                if len(actual_normal_paths) == len(img_paths):
                    input_dict["normals"] = load_and_preprocess_normals(actual_normal_paths)
                normal_valid_paths = [s["normal_valid"] for s in selected_frames]
                actual_normal_valid_paths = [p for p in normal_valid_paths if os.path.exists(p)]
                if len(actual_normal_valid_paths) == len(img_paths):
                    input_dict["normal_valid_mask"] = load_and_preprocess_binary_masks(actual_normal_valid_paths)
                
            return input_dict
