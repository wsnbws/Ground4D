import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter_add, scatter_softmax, scatter_max


class TemporalVoxelFusion(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        hidden_dim=64,
        n_time_freqs=4,
        temperature=0.1,
        use_temporal_attention=True,
        use_voxel_norm=True,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_time_freqs = n_time_freqs
        self.temperature = temperature
        self.use_temporal_attention = use_temporal_attention
        self.use_voxel_norm = use_voxel_norm
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(14, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        time_input_dim = 1 + 2 * n_time_freqs
        self.time_mlp = nn.Sequential(
            nn.Linear(time_input_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.attention_net = None
        if self.use_temporal_attention:
            self.attention_net = nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim),
                nn.SiLU(),
                # nn.Linear(hidden_dim, hidden_dim),
                # nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )
    
    def sinusoidal_encoding(self, t):
        freq_bands = 2.0 ** torch.linspace(
            0, self.n_time_freqs - 1, self.n_time_freqs, device=t.device
        )
        out = [t]
        for freq in freq_bands:
            out.append(torch.sin(t * freq * math.pi))
            out.append(torch.cos(t * freq * math.pi))
        return torch.cat(out, dim=-1)
    
    def forward(self, gaussian_features, voxel_indices, num_voxels, target_time):
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
        
        g_feat_embed = self.feature_encoder(g_feats)
        g_time_encoded = self.sinusoidal_encoding(g_time)
        g_time_embed = self.time_mlp(g_time_encoded)
        g_combined_embed = g_feat_embed + g_time_embed
        
        t_target_encoded = self.sinusoidal_encoding(target_time)
        t_target_embed = self.time_mlp(t_target_encoded)
        t_target_embed_expanded = t_target_embed.expand(N, -1)
        
        if self.use_temporal_attention:
            attn_input = torch.cat([g_combined_embed, t_target_embed_expanded], dim=-1)
            attn_scores = self.attention_net(attn_input).squeeze(-1) / self.temperature
        else:
            attn_scores = torch.zeros(N, device=device, dtype=g_combined_embed.dtype)

        if self.use_voxel_norm:
            # IN on: normalize scores within each voxel.
            weights = scatter_softmax(attn_scores, voxel_indices, dim=0)
        else:
            # IN off: keep unnormalized per-Gaussian contributions.
            if self.use_temporal_attention:
                weights = torch.sigmoid(attn_scores)
            else:
                weights = torch.ones(N, device=device, dtype=g_combined_embed.dtype)
        
        positions = g_feats[:, 0:3]
        colors = g_feats[:, 3:6]
        opacity = g_feats[:, 6:7]
        scales = g_feats[:, 7:10]
        rotations = g_feats[:, 10:14]
        
        fused_pos = torch.zeros(M, 3, device=device)
        scatter_add(positions * weights.unsqueeze(-1), voxel_indices, dim=0, out=fused_pos)
        
        fused_color = torch.zeros(M, 3, device=device)
        scatter_add(colors * weights.unsqueeze(-1), voxel_indices, dim=0, out=fused_color)
        
        max_opacity, _ = scatter_max(opacity.squeeze(-1), voxel_indices, dim=0, dim_size=M)
        weighted_opacity = opacity.squeeze(-1) * weights
        avg_opacity = torch.zeros(M, device=device)
        scatter_add(weighted_opacity, voxel_indices, dim=0, out=avg_opacity)
        fused_opacity = 0.7 * max_opacity + 0.3 * avg_opacity
        
        log_scales = torch.log(scales.clamp(min=1e-6))
        weighted_log_scales = log_scales * weights.unsqueeze(-1)
        fused_log_scales = torch.zeros(M, 3, device=device)
        scatter_add(weighted_log_scales, voxel_indices, dim=0, out=fused_log_scales)
        fused_scales = torch.exp(fused_log_scales)
        
        weighted_rot = rotations * weights.unsqueeze(-1)
        fused_rot = torch.zeros(M, 4, device=device)
        scatter_add(weighted_rot, voxel_indices, dim=0, out=fused_rot)
        fused_rot = F.normalize(fused_rot, p=2, dim=-1, eps=1e-6)
        
        fused_attrs = torch.cat([
            fused_pos,
            fused_color,
            fused_opacity.unsqueeze(-1),
            fused_scales,
            fused_rot
        ], dim=-1)
        
        dummy_time = torch.zeros(M, 1, device=device)
        fused_gaussians = torch.cat([fused_attrs, dummy_time], dim=-1)
        
        return fused_gaussians, weights
