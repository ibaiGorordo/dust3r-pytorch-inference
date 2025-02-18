import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from duster.blocks import Block, DecoderBlock, PatchEmbed
from duster.rope2d import RoPE2D


class ImageEncoder(nn.Module):
    def __init__(self,
                 width=512,
                 height=512,
                 patch_size=16,
                 enc_embed_dim=1024,
                 enc_num_heads=12,
                 enc_depth=12,
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)
                 ):
        super().__init__()
        self.patch_embed = PatchEmbed((height, width), (patch_size,patch_size), 3, enc_embed_dim)
        self.rope = RoPE2D(width, height, patch_size, base=100.0, D=enc_embed_dim//2)

        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, self.rope, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x)

        return self.enc_norm(x)


if __name__ == '__main__':
    model = ImageEncoder()


    model_path = "../models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    enc_state_dict = {
        k: v for k, v in ckpt['model'].items()
        if k.startswith("patch_embed") or k.startswith("enc_blocks")
    }

    model.load_state_dict(enc_state_dict, strict=False)
