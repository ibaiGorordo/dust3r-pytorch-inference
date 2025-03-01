from copy import deepcopy

import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from blocks import Block, DecoderBlock, PatchEmbed
from rope2d import RoPE2D
from heads import DPTHead, LinearPts3d


class Dust3rEncoder(nn.Module):
    def __init__(self,
                 ckpt,
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

        self._load_checkpoint(ckpt)
        self.to(device)

    @torch.inference_mode()
    def forward(self, x):
        x = self.patch_embed(x)

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x)

        return self.enc_norm(x)

    def _load_checkpoint(self, ckpt):
        enc_state_dict = {
            k: v for k, v in ckpt['model'].items()
            if k.startswith("patch_embed") or k.startswith("enc_blocks") or k.startswith("enc_norm")
        }
        self.load_state_dict(enc_state_dict, strict=True)

class Dust3rDecoder(nn.Module):
    def __init__(self,
                 ckpt,
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

        self._load_checkpoint(ckpt)
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

    def _load_checkpoint(self, ckpt):
        dec_state_dict = {
            k: v for k, v in ckpt['model'].items()
            if k.startswith("decoder_embed") or k.startswith("dec_blocks") or k.startswith("dec_norm")
        }
        self.load_state_dict(dec_state_dict, strict=True)

class Dust3rHead(nn.Module):
    def __init__(self,
                 ckpt,
                 width=512,
                 height=512,
                 device=torch.device('cuda'),
                 ):
        super().__init__()

        self.downstream_head1 = DPTHead(width, height) if self._is_dpt(ckpt) else LinearPts3d(width, height)
        self.downstream_head2 = DPTHead(width, height) if self._is_dpt(ckpt) else LinearPts3d(width, height)

        self._load_checkpoint(ckpt)
        self.to(device)


    @torch.inference_mode()
    def forward(self, d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12):
        pts3d1, conf1 = self.downstream_head1(d1_0, d1_6, d1_9, d1_12)
        pts3d2, conf2 = self.downstream_head2(d2_0, d2_6, d2_9, d2_12)

        return pts3d1, conf1, pts3d2, conf2

    def _load_checkpoint(self, ckpt):
        head_state_dict = {
            k.replace(".dpt", ""): v
            for k, v in ckpt['model'].items()
            if "head" in k
        }
        self.load_state_dict(head_state_dict, strict=True)

    def _is_dpt(self, ckpt):
        return any("dpt" in k for k in ckpt['model'].keys())


if __name__ == '__main__':

    import pickle
    model_path = "../models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)


    encoder = Dust3rEncoder(ckpt, width=512, height=288, device=torch.device('cuda'))
    decoder = Dust3rDecoder(ckpt, width=512, height=288, device=torch.device('cuda'))
    head = Dust3rHead(ckpt, width=512, height=288, device=torch.device('cuda'))


    # load the "img1_img2.pkl" file
    with open("../img1_img2.pkl", "rb") as f:
        img1, img2, dec1, dec2, feat1, feat2, pos1, pos2, pts3d1, pts3d2, conf1, conf2 = pickle.load(f)


    feat = encoder(torch.cat((img1, img2)))
    f1, f2 = feat.chunk(2, dim=0)
    d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12 = decoder(f1, f2)
    pt1, cf1, pt2, cf2 = head(d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12)
    print(f1 - feat1)
    print(f2 - feat2)
    print(d1_0 - dec1[0])
    print(d2_0 - dec2[0])
    print(d1_6 - dec1[6])
    print(d2_6 - dec2[6])
    print(d1_9 - dec1[9])
    print(d2_9 - dec2[9])
    print(d1_12 - dec1[12])
    print(d2_12 - dec2[12])
    print(pts3d1 - pt1)
    print(pts3d2 - pt2)
    print(conf1 - cf1)
    print(conf2 - cf2)


    # torch.onnx.export(
    #     encoder,
    #     (torch.cat((img1, img2)),),
    #     "encoder.onnx",
    #     input_names=["img"],
    #     output_names=["feats"],
    #     opset_version=13,  # or whichever opset you need
    # )
    #
    # torch.onnx.export(
    #     decoder,
    #     (f1, f2),
    #     "decoder.onnx",
    #     input_names=["f1", "f2"],
    #     output_names=["d1_0", "d1_6", "d1_9", "d1_12", "d2_0", "d2_6", "d2_9", "d2_12"],
    #     opset_version=13,  # or whichever opset you need
    # )
    #
    # torch.onnx.export(
    #     head,
    #     (d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12),
    #     "head.onnx",
    #     input_names=["d1_0", "d1_6", "d1_9", "d1_12", "d2_0", "d2_6", "d2_9", "d2_12"],
    #     output_names=["pts3d1", "conf1", "pts3d2", "conf2"],
    #     opset_version=13,  # or whichever opset you need
    # )