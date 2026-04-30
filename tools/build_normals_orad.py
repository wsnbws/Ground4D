import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SNE(nn.Module):
    """Surface normal estimator from depth + camera intrinsics."""

    def __init__(self):
        super().__init__()
        gx = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        gy = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        diff_kernel_array = torch.tensor(
            [
                [-1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, -1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, -1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, -1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, -1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, -1],
            ],
            dtype=torch.float32,
        ).view(8, 1, 3, 3)
        self.register_buffer("gx", gx, persistent=False)
        self.register_buffer("gy", gy, persistent=False)
        self.register_buffer("diff_kernels", diff_kernel_array, persistent=False)
        self._grid_cache = {}

    def _get_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
        key = (h, w, str(device), str(dtype))
        if key not in self._grid_cache:
            v_map, u_map = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing="ij",
            )
            self._grid_cache[key] = (u_map[None, None], v_map[None, None])  # [1,1,H,W]
        return self._grid_cache[key]

    def forward_batch(self, depth: torch.Tensor, cam_param: torch.Tensor) -> torch.Tensor:
        # depth: [B, H, W], cam_param: [B, 3, 3]
        b, h, w = depth.shape
        depth = depth.float()
        cam_param = cam_param.float()
        u_map, v_map = self._get_grid(h, w, depth.device, depth.dtype)
        fx = cam_param[:, 0, 0].view(b, 1, 1, 1)
        fy = cam_param[:, 1, 1].view(b, 1, 1, 1)
        cx = cam_param[:, 0, 2].view(b, 1, 1, 1)
        cy = cam_param[:, 1, 2].view(b, 1, 1, 1)

        z = depth.clone().unsqueeze(1)  # [B,1,H,W]
        z = torch.nan_to_num(z, nan=0.0)
        x = z * (u_map - cx) / (fx + 1e-8)
        y = z * (v_map - cy) / (fy + 1e-8)
        d = 1.0 / (z + 1e-8)

        gx = self.gx.to(device=depth.device, dtype=depth.dtype)
        gy = self.gy.to(device=depth.device, dtype=depth.dtype)
        diff_kernels = self.diff_kernels.to(device=depth.device, dtype=depth.dtype)
        gu = F.conv2d(d, gx, padding=1)
        gv = F.conv2d(d, gy, padding=1)
        nx_t = gu * fx
        ny_t = gv * fy

        phi = torch.atan2(ny_t, nx_t) + np.pi
        a = torch.cos(phi)
        bsin = torch.sin(phi)

        # Vectorized 8-direction differential kernels.
        x_d = F.conv2d(x, diff_kernels, padding=1)  # [B,8,H,W]
        y_d = F.conv2d(y, diff_kernels, padding=1)
        z_d = F.conv2d(z, diff_kernels, padding=1)

        nz_i = (nx_t * x_d + ny_t * y_d) / (z_d + 1e-8)
        norm = torch.sqrt(nx_t * nx_t + ny_t * ny_t + nz_i * nz_i + 1e-8)
        nx_t_i = torch.nan_to_num(nx_t / norm, nan=0.0)
        ny_t_i = torch.nan_to_num(ny_t / norm, nan=0.0)
        nz_t_i = torch.nan_to_num(nz_i / norm, nan=0.0)

        sum_nx = nx_t_i.sum(dim=1, keepdim=True)
        sum_ny = ny_t_i.sum(dim=1, keepdim=True)
        sum_nz = nz_t_i.sum(dim=1, keepdim=True)

        theta = -torch.atan2((sum_nx * a + sum_ny * bsin), (sum_nz + 1e-8))
        nx = torch.sin(theta) * torch.cos(phi)
        ny = torch.sin(theta) * torch.sin(phi)
        nz = torch.cos(theta)

        nan_mask = torch.isnan(nz)
        nx = torch.where(nan_mask, torch.zeros_like(nx), nx)
        ny = torch.where(nan_mask, torch.zeros_like(ny), ny)
        nz = torch.where(nan_mask, torch.full_like(nz, -1.0), nz)

        sign = torch.where(ny > 0, torch.full_like(ny, -1.0), torch.ones_like(ny))
        nx = nx * sign
        ny = ny * sign
        nz = nz * sign

        normals = torch.cat([nx, ny, nz], dim=1)  # [B,3,H,W]
        normals = F.normalize(normals, dim=1, eps=1e-6)
        return normals


def parse_cam_k(calib_path: Path) -> np.ndarray:
    txt = calib_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    k_line = None
    for line in txt:
        if line.startswith("cam_K:"):
            k_line = line
            break
    if k_line is None:
        raise ValueError(f"cam_K not found in {calib_path}")
    vals = [float(x) for x in k_line.replace("cam_K:", "").strip().split()]
    if len(vals) != 9:
        raise ValueError(f"Invalid cam_K in {calib_path}: {len(vals)} values")
    return np.array(vals, dtype=np.float32).reshape(3, 3)


def load_depth(depth_path: Path, depth_scale: float) -> np.ndarray:
    if depth_path.suffix.lower() == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    else:
        depth = np.array(Image.open(depth_path)).astype(np.float32)
    return depth * depth_scale


def load_binary_mask(mask_path: Path) -> np.ndarray:
    if not mask_path.exists():
        return None
    m = np.array(Image.open(mask_path))
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def process_scene(scene_dir: Path, sne: SNE, device: torch.device, args) -> None:
    depth_dir = scene_dir / "dense_depth"
    calib_dir = scene_dir / "calib"
    image_dir = scene_dir / "image_data"
    sky_dir = scene_dir / "sky_masks"
    dyn_dir = scene_dir / "dynamic_masks"
    if not (depth_dir.exists() and calib_dir.exists() and image_dir.exists()):
        return

    out_normal = scene_dir / "normal_map"
    out_vis = scene_dir / "normal_vis"
    out_valid = scene_dir / "normal_valid_mask"
    out_normal.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        out_vis.mkdir(parents=True, exist_ok=True)
    if args.save_valid_mask:
        out_valid.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob("*.png"))
    frame_items = []
    for image_path in image_files:
        stem = image_path.stem
        depth_path = depth_dir / f"{stem}.png"
        calib_path = calib_dir / f"{stem}.txt"
        out_npy = out_normal / f"{stem}.npy"
        if args.skip_existing and out_npy.exists():
            continue
        if depth_path.exists() and calib_path.exists():
            frame_items.append((stem, depth_path, calib_path, out_npy))

    if not frame_items:
        print(f"skip: {scene_dir.name} (all done)")
        return

    # Preload calib to avoid repeated text parsing.
    k_cache = {stem: parse_cam_k(calib_path) for stem, _, calib_path, _ in frame_items}

    with torch.inference_mode():
        for start in tqdm(range(0, len(frame_items), args.batch_size), desc=f"{scene_dir.name}", leave=False):
            chunk = frame_items[start:start + args.batch_size]
            stems = [x[0] for x in chunk]
            depths_np = [load_depth(x[1], args.depth_scale) for x in chunk]
            ks_np = [k_cache[s] for s in stems]

            depth_batch = torch.from_numpy(np.stack(depths_np, axis=0)).to(device=device, dtype=torch.float32)
            k_batch = torch.from_numpy(np.stack(ks_np, axis=0)).to(device=device, dtype=torch.float32)
            normals_batch = sne.forward_batch(depth_batch, k_batch)  # [B,3,H,W]
            normals_batch_np = normals_batch.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

            for i, (stem, _, _, out_npy) in enumerate(chunk):
                depth = depths_np[i]
                valid = np.isfinite(depth) & (depth > args.min_valid_depth)
                if args.max_valid_depth > 0:
                    valid = valid & (depth < args.max_valid_depth)

                if args.use_sky_mask and sky_dir.exists():
                    sky = load_binary_mask(sky_dir / f"{stem}_fillcolor.png")
                    if sky is not None:
                        valid = valid & (sky == 0)

                if args.use_dynamic_mask and dyn_dir.exists():
                    dyn = load_binary_mask(dyn_dir / f"{stem}.png")
                    if dyn is not None:
                        valid = valid & (dyn == 0)

                normals_np = normals_batch_np[i]
                normals_np[~valid] = 0.0
                np.save(out_npy, normals_np)

                if args.save_vis:
                    vis = ((normals_np * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
                    Image.fromarray(vis).save(out_vis / f"{stem}.png")

                if args.save_valid_mask:
                    valid_u8 = (valid.astype(np.uint8) * 255)
                    Image.fromarray(valid_u8).save(out_valid / f"{stem}.png")

    print(f"done: {scene_dir.name}")


def main():
    parser = argparse.ArgumentParser(description="Build normal maps from ORAD dense_depth + calib")
    parser.add_argument("--root", type=str, required=True, help="ORAD split root, e.g. /data20t/.../orad/training")
    parser.add_argument("--scene", type=str, default="", help="Optional single scene name")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1.0 / 256.0,
        help="Depth scale to meters (default follows ORAD uint16/256 convention)",
    )
    parser.add_argument("--min_valid_depth", type=float, default=0.1)
    parser.add_argument("--max_valid_depth", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_sky_mask", action="store_true")
    parser.add_argument("--use_dynamic_mask", action="store_true")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_valid_mask", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for GPU normal estimation")
    parser.add_argument("--skip_existing", action="store_true", help="Skip frames when normal_map/*.npy exists")
    args = parser.parse_args()

    root = Path(args.root)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    sne = SNE().to(device).eval()

    if args.scene:
        scenes = [root / args.scene]
    else:
        scenes = sorted([p for p in root.iterdir() if p.is_dir()])

    for scene_dir in scenes:
        process_scene(scene_dir, sne, device, args)


if __name__ == "__main__":
    main()
