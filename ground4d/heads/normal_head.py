import torch.nn as nn

from .dpt_head import DPTHead
from .head_act import activate_head


class NormalHead(nn.Module):
    """
    Dedicated normal prediction head.

    The feature fusion structure follows the same DPT-style dense head used by GS head.
    Output layout is: normal(3) + confidence(1).
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 256,
        normal_embed_dim: int = 64,
        out_channels=None,
        intermediate_layer_idx=None,
        pos_embed: bool = True,
        down_ratio: int = 1,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = [256, 512, 1024, 1024]
        if intermediate_layer_idx is None:
            intermediate_layer_idx = [4, 11, 17, 23]

        self.backbone = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            out_channels=out_channels,
            intermediate_layer_idx=intermediate_layer_idx,
            pos_embed=pos_embed,
            feature_only=True,
            down_ratio=down_ratio,
        )
        self.normal_predictor = nn.Sequential(
            nn.Conv2d(features, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0),
        )
        self.normal_embed = nn.Sequential(
            nn.Conv2d(features, normal_embed_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(normal_embed_dim, normal_embed_dim, kernel_size=1, stride=1, padding=0),
        )
        self.normal_embed_dim = normal_embed_dim

    def forward(
        self,
        aggregated_tokens_list,
        images,
        patch_start_idx,
        frames_chunk_size=8,
        return_features=False,
    ):
        feat = self.backbone(
            aggregated_tokens_list=aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
            frames_chunk_size=frames_chunk_size,
        )
        bsz, seq_len, channels, height, width = feat.shape
        feat_flat = feat.view(bsz * seq_len, channels, height, width)

        pred = self.normal_predictor(feat_flat)
        normal_map, normal_conf = activate_head(pred, activation="normalize", conf_activation="sigmoid")
        normal_map = normal_map.view(bsz, seq_len, *normal_map.shape[1:])
        normal_conf = normal_conf.view(bsz, seq_len, *normal_conf.shape[1:])

        if not return_features:
            return normal_map, normal_conf

        normal_embed = self.normal_embed(feat_flat).view(
            bsz, seq_len, self.normal_embed_dim, height, width
        )
        return normal_map, normal_conf, normal_embed
