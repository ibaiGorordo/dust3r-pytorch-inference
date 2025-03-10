from copy import deepcopy
from functools import partial
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12

from .blocks import Block, DecoderBlock, PatchEmbed
from .rope2d import RoPE2D
from .heads import DPTHead, LinearPts3d
from .preprocess import preprocess
from .postprocess import postprocess_with_color, estimate_intrinsics, estimate_camera_pose


@dataclass
class Output:
    input: np.ndarray
    pts3d: np.ndarray
    colors: np.ndarray
    conf_map: np.ndarray
    depth_map: np.ndarray
    intrinsic: np.ndarray
    pose: np.ndarray

class Dust3rEncoder(nn.Module):
    def __init__(self,
                 ckpt_dict,
                 batch=2,
                 width=512,
                 height=512,
                 patch_size=16,
                 enc_embed_dim=1024,
                 enc_num_heads=16,
                 enc_depth=24,
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 device=torch.device('cuda'),
                 ):
        super().__init__()
        self.patch_embed = PatchEmbed((height, width), (patch_size,patch_size), 3, enc_embed_dim)
        self.rope = RoPE2D(batch, width, height, patch_size, base=100.0, device=device)
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, self.rope, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)

        self._load_checkpoint(ckpt_dict)
        self.to(device)

    @torch.inference_mode()
    def forward(self, x):
        x = self.patch_embed(x)

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x)

        return self.enc_norm(x)

    def _load_checkpoint(self, ckpt_dict):
        enc_state_dict = {
            k: v for k, v in ckpt_dict['model'].items()
            if k.startswith("patch_embed") or k.startswith("enc_blocks") or k.startswith("enc_norm")
        }
        self.load_state_dict(enc_state_dict, strict=True)

class Dust3rDecoder(nn.Module):
    def __init__(self,
                 ckpt_dict,
                 batch=1,
                 width=512,
                 height=512,
                 patch_size=16,
                 enc_embed_dim=1024,
                 dec_embed_dim=768,
                 dec_num_heads=12,
                 dec_depth=12,
                 mlp_ratio=4,
                 norm_im2_in_dec=True, # whether to apply normalization of the 'memory' = (second image) in the decoder
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 device=torch.device('cuda'),
                 ):
        super().__init__()


        self.rope = RoPE2D(batch, width, height, patch_size, base=100.0, device=device)

        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                         norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])

        self.dec_blocks2 = deepcopy(self.dec_blocks)

        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

        self._load_checkpoint(ckpt_dict)
        self.to(device)

    @torch.inference_mode()
    def forward(self, f1, f2):
        f1_0 = f1_6 = f1_9 = f1
        f2_0 = f2_6 = f2_9 = f2

        # Project to decoder dimension
        f1_prev, f2_prev = self.decoder_embed(f1), self.decoder_embed(f2)


        for i, (blk1, blk2) in enumerate(zip(self.dec_blocks, self.dec_blocks2), start=1):
            # img1 side
            f1, _ = blk1(f1_prev, f2_prev)

            # img2 side
            f2, _ = blk2(f2_prev, f1_prev)

            # Store the result
            f1_prev, f2_prev = f1, f2

            if i == 6:
                f1_6, f2_6 = f1, f2
            elif i == 9:
                f1_9, f2_9 = f1, f2

        f1_12, f2_12 = self.dec_norm(f1), self.dec_norm(f2)

        return f1_0, f1_6, f1_9, f1_12, f2_0, f2_6, f2_9, f2_12

    def _load_checkpoint(self, ckpt_dict):
        dec_state_dict = {
            k: v for k, v in ckpt_dict['model'].items()
            if k.startswith("decoder_embed") or k.startswith("dec_blocks") or k.startswith("dec_norm")
        }
        self.load_state_dict(dec_state_dict, strict=True)

class Dust3rHead(nn.Module):
    def __init__(self,
                 ckpt_dict,
                 width=512,
                 height=512,
                 device=torch.device('cuda'),
                 ):
        super().__init__()

        self.downstream_head1 = DPTHead(width, height) if self._is_dpt(ckpt_dict) else LinearPts3d(width, height)
        self.downstream_head2 = DPTHead(width, height) if self._is_dpt(ckpt_dict) else LinearPts3d(width, height)

        self._load_checkpoint(ckpt_dict)
        self.to(device)


    @torch.inference_mode()
    def forward(self, d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12):
        pts3d1, conf1 = self.downstream_head1(d1_0, d1_6, d1_9, d1_12)
        pts3d2, conf2 = self.downstream_head2(d2_0, d2_6, d2_9, d2_12)

        return pts3d1, conf1, pts3d2, conf2

    def _load_checkpoint(self, ckpt_dict):
        head_state_dict = {
            k.replace(".dpt", ""): v
            for k, v in ckpt_dict['model'].items()
            if "head" in k
        }
        self.load_state_dict(head_state_dict, strict=True)

    def _is_dpt(self, ckpt_dict):
        return any("dpt" in k for k in ckpt_dict['model'].keys())

class Dust3r(nn.Module):
    def __init__(self,
                 model_path: str,
                 width: int = 512,
                 height: int = 512,
                 conf_threshold: float = 3.0,
                 device: torch.device = torch.device('cuda'),
                 ):
        super().__init__()

        self.width = width
        self.height = height
        self.device = device
        self.conf_threshold = conf_threshold

        ckpt_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.encoder = Dust3rEncoder(ckpt_dict, width=width, height=height, device=device, batch=2)
        self.decoder = Dust3rDecoder(ckpt_dict, width=width, height=height, device=device)
        self.head = Dust3rHead(ckpt_dict, width=width, height=height, device=device)

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> tuple[Output, Output]:
        return self.forward(img1, img2)

    @torch.inference_mode()
    def forward(self, img1: np.ndarray, img2: np.ndarray) -> tuple[Output, Output]:

        input1, frame1 = preprocess(img1, self.width, self.height, self.device)
        input2, frame2 = preprocess(img2, self.width, self.height, self.device)

        input = torch.cat((input1, input2), dim=0)
        feat = self.encoder(input)
        feat1, feat2 = feat.chunk(2, dim=0)

        d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12 = self.decoder(feat1, feat2)
        pt1, cf1, pt2, cf2 = self.head(d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12)

        output1, output2 = self.postprocess(frame1, pt1, cf1, frame2, pt2, cf2)

        return output1, output2

    def postprocess(self,
                    frame1: np.ndarray,
                    pt1: torch.Tensor,
                    cf1: torch.Tensor,
                    frame2: np.ndarray,
                    pt2: torch.Tensor,
                    cf2: torch.Tensor,
                    ) -> tuple[Output, Output]:

        pts1, colors1, conf_map1, depth_map1, mask1 = postprocess_with_color(pt1, cf1, frame1, threshold=self.conf_threshold)
        pts2, colors2, conf_map2, depth_map2, mask2 = postprocess_with_color(pt2, cf2, frame2, threshold=self.conf_threshold)

        # Estimate intrinsics
        intrinsics1 = estimate_intrinsics(pts1, mask1)
        intrinsics2 = estimate_intrinsics(pts2, mask2)

        # Estimate camera pose (the first one is the origin)
        cam_pose1 = np.eye(4)
        cam_pose2 = estimate_camera_pose(pts2, intrinsics2, mask2)

        output1 = Output(frame1, pts1, colors1, conf_map1, depth_map1, intrinsics1, cam_pose1)
        output2 = Output(frame2, pts2, colors2, conf_map2, depth_map2, intrinsics2, cam_pose2)

        return output1, output2


class Dust3rAllToOne(Dust3r):
    def __init__(self,
                 model_path: str,
                 origin_img: torch.Tensor,
                 device: torch.device = torch.device('cuda'),
                 ):
        super().__init__(model_path, origin_img.shape[-1], origin_img.shape[-2], device)
        self.origin_feat = self.encoder(origin_img)

    def update_origin(self, origin_img):
        self.origin_feat = self.encoder(origin_img)

    @torch.inference_mode()
    def forward(self, img: torch.Tensor):

        current_feat = self.encoder(img)

        d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12 = self.decoder(self.origin_feat, current_feat)
        pt1, cf1, pt2, cf2 = self.head(d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12)

        return pt1, cf1, pt2, cf2