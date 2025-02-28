# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# post process function for all heads: extract 3D points/confidence from output
# --------------------------------------------------------
import torch


def postprocess(out):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    depth = reg_dense_depth(fmap[:, :, :, 0:3])
    conf = reg_dense_conf(fmap[:, :, :, 3])
    return depth, conf


def reg_dense_depth(xyz):
    """
    extract 3D points from prediction head output
    """

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    return xyz * torch.expm1(d)

def reg_dense_conf(x, vmin=1, vmax=torch.inf):
    """
    extract confidence from prediction head output
    """
    return vmin + x.exp().clip(max=vmax-vmin)
