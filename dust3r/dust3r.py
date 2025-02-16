import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from dust3r.blocks import Block, DecoderBlock, PatchEmbed
from dust3r.rope2d import RoPE2D

