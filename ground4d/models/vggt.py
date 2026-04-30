# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from ground4d.models.aggregator import Aggregator
from ground4d.heads.camera_head import CameraHead
from ground4d.heads.dpt_head import DPTHead, GaussianHead
from ground4d.heads.normal_head import NormalHead
from ground4d.heads.track_head import TrackHead
from ground4d.heads.gs_head import GaussianDecoder
from ground4d.models.sky import SkyGaussian
from ground4d.models.fusion import PointNetGSFusion
#from ground4d.splatformer.feature_predictor import FeaturePredictor


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        semantic_num=10,
        inject_normal_to_gs=False,
        normal_embed_dim=64,
        normal_cond_scale=0.1,
    ):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")# ,down_ratio=2)
        #self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")# ,down_ratio=2)
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="sigmoid")

        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        
        # GS channels: rgb(3)+opacity(1)+scale(3)+rotation(4)+conf(1)
        gs_out_dim = 3 + 1 + 3 + 4 + 1
        self.gs_head = GaussianHead(
            dim_in=3 * embed_dim,
            output_dim=gs_out_dim,
            activation="sigmoid",
            use_normal_condition=inject_normal_to_gs,
            normal_cond_dim=normal_embed_dim,
            normal_cond_scale=normal_cond_scale,
        )
        self.inject_normal_to_gs = bool(inject_normal_to_gs)
        self.normal_head = (
            NormalHead(dim_in=3 * embed_dim, normal_embed_dim=normal_embed_dim) if self.inject_normal_to_gs else None
        )
        self.instance_head = DPTHead(dim_in= embed_dim, output_dim = 1 + 1, activation="linear") # ,down_ratio=2)#RGB
        self.semantic_head = DPTHead(dim_in= embed_dim, output_dim = semantic_num + 1, activation="linear")# ,down_ratio=2)#RGB
        # Color, opacity, scale, rotation
        self.sky_model = SkyGaussian()
        #self.fusion_model = PointNetGSFusion()
        #self.splatformer = FeaturePredictor()
        #self.point_offset_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log_1")


    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
    ):
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, image_tokens_list, dino_token_list, image_feature, patch_start_idx = self.aggregator(images)
        
        predictions = {}

        predictions["image_feature"] = image_feature

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

 
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            normal_map = None
            normal_conf = None
            normal_embed = None
            if self.normal_head is not None:
                if self.inject_normal_to_gs:
                    normal_map, normal_conf, normal_embed = self.normal_head(
                        image_tokens_list,
                        images,
                        patch_start_idx,
                        return_features=True,
                    )
                else:
                    normal_map, normal_conf = self.normal_head(image_tokens_list, images, patch_start_idx)
                predictions["normal_map"] = normal_map
                predictions["normal_conf"] = normal_conf
                # Backward compatibility for existing training scripts.
                predictions["gs_normal_map"] = normal_map

            if self.gs_head is not None:
                gs_map, gs_conf = self.gs_head(
                    image_tokens_list,
                    images,
                    patch_start_idx,
                    normal_cond=normal_embed if self.inject_normal_to_gs else None,
                    normal_conf=normal_conf if self.inject_normal_to_gs else None,
                )
                predictions["gs_map"] = gs_map
                predictions["gs_conf"] = gs_conf

            if self.instance_head is not None:
                dynamic_conf, _ = self.instance_head(dino_token_list, images, patch_start_idx)
                predictions["dynamic_conf"] = dynamic_conf

            if self.semantic_head is not None:
                semantic_logits, _ = self.semantic_head(dino_token_list, images, patch_start_idx)
                predictions["semantic_logits"] = semantic_logits

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        return predictions



