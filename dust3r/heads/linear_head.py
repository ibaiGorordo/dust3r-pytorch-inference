# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# linear head implementation for DUST3R
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from .postprocess import postprocess


class LinearPts3d (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self,
                 width=512,
                 height=512,
                 patch_size=16,
                 dec_embed_dim=768,
                 has_conf=True):
        super().__init__()
        self.patch_size = patch_size
        self.has_conf = has_conf
        self.num_h = height // patch_size
        self.num_w = width // patch_size

        self.proj = nn.Linear(dec_embed_dim, (3 + has_conf)*self.patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, tokens_0, tokens_6, tokens_9, tokens_12):
        B, S, D = tokens_12.shape

        # extract 3D points
        feat = self.proj(tokens_12)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, self.num_h, self.num_w)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return postprocess(feat)
