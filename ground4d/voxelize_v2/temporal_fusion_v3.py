import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_softmax


class TemporalVoxelFusionV3(nn.Module):
    """
    V3 fusion strategy (rule-based):
    1) No learned temporal attention.
    2) Build voxel-wise weights directly from gs_conf + alpha_t-style temporal prior.
    3) Sharpen distribution with temperature and fuse each voxel to one Gaussian.
    """

    def __init__(
        self,
        feature_dim=128,
        hidden_dim=64,
        n_time_freqs=4,
        temperature=0.07,
        gamma1=0.1,
    ):
        super().__init__()
        # Keep args for backward compatibility with existing training scripts.
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_time_freqs = n_time_freqs
        self.temperature = float(temperature)
        self.gamma1 = float(gamma1)
        self.eps = 1e-6

    def _alpha_t_prior(self, t, t0, gamma0):
        # Same functional form as train.py::alpha_t, without multiplying opacity.
        gamma0 = gamma0.clamp(min=self.eps)
        gamma1 = torch.as_tensor(self.gamma1, dtype=t.dtype, device=t.device).clamp(min=self.eps, max=1.0)
        sigma = torch.log(gamma1) / (gamma0 ** 2 + self.eps)
        return torch.exp(sigma * (t0 - t) ** 2)

    def forward(self, gaussian_features, voxel_indices, num_voxels, target_time, gs_conf=None):
        """
        Returns:
            fused_gaussians: [M, 15] (one Gaussian per voxel)
            weights: [N] voxel-wise normalized temporal weights
        """
        N = gaussian_features.shape[0]
        M = num_voxels
        device = gaussian_features.device

        if not isinstance(target_time, torch.Tensor):
            target_time = torch.tensor([target_time], device=device, dtype=torch.float32)
        if target_time.dim() == 0:
            target_time = target_time.view(1, 1)
        elif target_time.dim() == 1:
            target_time = target_time.view(-1, 1)

        g_feats = gaussian_features[:, :14]
        g_time = gaussian_features[:, 14:15]

        if gs_conf is None:
            gs_conf = torch.ones(N, dtype=g_feats.dtype, device=device)
        else:
            if gs_conf.dim() == 2 and gs_conf.shape[-1] == 1:
                gs_conf = gs_conf.squeeze(-1)
            if gs_conf.dim() != 1:
                gs_conf = gs_conf.reshape(-1)
            gs_conf = gs_conf.to(dtype=g_feats.dtype, device=device)

        # Pure confidence-temporal prior (no learned attention).
        t0 = target_time.expand_as(g_time)
        temporal_prior = self._alpha_t_prior(g_time, t0, gs_conf.unsqueeze(-1)).squeeze(-1)
        prior_scores = torch.log(temporal_prior.clamp(min=self.eps))
        sharp_scores = prior_scores / max(self.temperature, self.eps)
        weights = scatter_softmax(sharp_scores, voxel_indices, dim=0)

        # Fuse per-voxel Gaussian attributes.
        positions = g_feats[:, 0:3]
        colors = g_feats[:, 3:6]
        opacity = g_feats[:, 6:7]
        scales = g_feats[:, 7:10]
        rotations = g_feats[:, 10:14]

        fused_pos = torch.zeros(M, 3, device=device, dtype=g_feats.dtype)
        scatter_add(positions * weights.unsqueeze(-1), voxel_indices, dim=0, out=fused_pos)

        fused_color = torch.zeros(M, 3, device=device, dtype=g_feats.dtype)
        scatter_add(colors * weights.unsqueeze(-1), voxel_indices, dim=0, out=fused_color)

        max_opacity, _ = scatter_max(opacity.squeeze(-1), voxel_indices, dim=0, dim_size=M)
        weighted_opacity = opacity.squeeze(-1) * weights
        avg_opacity = torch.zeros(M, device=device, dtype=g_feats.dtype)
        scatter_add(weighted_opacity, voxel_indices, dim=0, out=avg_opacity)
        fused_opacity = 0.7 * max_opacity + 0.3 * avg_opacity

        log_scales = torch.log(scales.clamp(min=1e-6))
        weighted_log_scales = log_scales * weights.unsqueeze(-1)
        fused_log_scales = torch.zeros(M, 3, device=device, dtype=g_feats.dtype)
        scatter_add(weighted_log_scales, voxel_indices, dim=0, out=fused_log_scales)
        fused_scales = torch.exp(fused_log_scales)

        weighted_rot = rotations * weights.unsqueeze(-1)
        fused_rot = torch.zeros(M, 4, device=device, dtype=g_feats.dtype)
        scatter_add(weighted_rot, voxel_indices, dim=0, out=fused_rot)
        fused_rot = F.normalize(fused_rot, p=2, dim=-1, eps=1e-6)

        fused_attrs = torch.cat([
            fused_pos,
            fused_color,
            fused_opacity.unsqueeze(-1),
            fused_scales,
            fused_rot,
        ], dim=-1)

        dummy_time = torch.zeros(M, 1, device=device, dtype=g_feats.dtype)
        fused_gaussians = torch.cat([fused_attrs, dummy_time], dim=-1)
        return fused_gaussians, weights