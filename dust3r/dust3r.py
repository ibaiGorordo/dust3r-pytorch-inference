from copy import deepcopy

import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from blocks import Block, DecoderBlock, PatchEmbed
from rope2d import RoPE2D


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

        self.load_checkpoint(ckpt)
        self.to(device)

    @torch.inference_mode()
    def forward(self, x):
        x = self.patch_embed(x)

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x)

        return self.enc_norm(x)

    def load_checkpoint(self, ckpt):
        enc_state_dict = {
            k: v for k, v in ckpt['model'].items()
            if k.startswith("patch_embed") or k.startswith("enc_blocks") or k.startswith("enc_norm")
        }
        self.load_state_dict(enc_state_dict, strict=False)

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

        self.load_checkpoint(ckpt)
        self.to(device)

    @torch.inference_mode()
    def forward(self, f1, f2):
        # project to decoder dim
        f1_prev = self.decoder_embed(f1)
        f2_prev = self.decoder_embed(f2)
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(f1_prev, f2_prev)

            # img2 side
            f2, _ = blk2(f2_prev, f1_prev)

            # store the result
            f1_prev = f1
            f2_prev = f2

        f1 = self.dec_norm(f1)
        f2 = self.dec_norm(f2)

        return f1, f2

    def load_checkpoint(self, ckpt):
        dec_state_dict = {
            k: v for k, v in ckpt['model'].items()
            if k.startswith("decoder_embed") or k.startswith("dec_blocks") or k.startswith("dec_norm")
        }
        self.load_state_dict(dec_state_dict, strict=False)

if __name__ == '__main__':

    import pickle
    model_path = "../models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    encoder = Dust3rEncoder(ckpt, width=512, height=288, device=torch.device('cuda'))
    decoder = Dust3rDecoder(ckpt, width=512, height=288, device=torch.device('cuda'))

    # load the "img1_img2.pkl" file
    with open("../img1_img2.pkl", "rb") as f:
        img1, img2, dec1, dec2, feat1, feat2, pos1, pos2 = pickle.load(f)


    feat = encoder(torch.cat((img1, img2)))
    f1, f2 = feat.chunk(2, dim=0)
    d1, d2 = decoder(f1, f2)
    print(f1 - feat1)
    print(f2 - feat2)
    print(d1 - dec1[-1])
    print(d2 - dec2[-1])


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
    #     output_names=["d1", "d2"],
    #     opset_version=13,  # or whichever opset you need
    # )