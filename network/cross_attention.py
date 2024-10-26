import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

from segformer import *


class Cross_Attention(nn.Module):
    def __init__(self, key_dim, value_dim, h, w, head_count=1):
        super().__init__()

        self.h = h
        self.w = w

        self.reprojection = nn.Conv2d(value_dim, 2 * value_dim, 1)
        self.norm = nn.LayerNorm(2 * value_dim)

    def forward(self, x1, x2):
        B, N, C = x1.size()  # (Batch, Tokens, Embedding dim)

        att_maps = []

        query = F.softmax(x2, dim=2)
        key = F.softmax(x2, dim=1)
        value = x1

        ## Choose one of the following attention:
        # -------------- channel cross-attention--------------
        pairwise_similarities = query.transpose(1, 2) @ key
        att_map = pairwise_similarities @ value.transpose(1, 2)

        ##-------------- efficient cross-Attention-------------
        # context = key.transpose(1, 2) @ value
        # att_map = (query @ context).transpose(1, 2)

        ##-------------- efficient channel cross-attention--------------
        # context = key @ value.transpose(1, 2)
        # att_map = query.transpose(1, 2) @ context
        att_maps.append(att_map)
        Att_maps = torch.cat(att_maps, dim=1).reshape(B, C, self.h, self.w)
        reprojected_value = self.reprojection(Att_maps).reshape(B, 2 * C, N).permute(0, 2, 1)
        reprojected_value = self.norm(reprojected_value)

        return reprojected_value

class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(self, in_dim, key_dim, value_dim, height, width, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.H = height
        self.W = width
        self.attn = Cross_Attention(key_dim, value_dim, height, width, head_count=head_count)
        self.norm2 = nn.LayerNorm((in_dim * 2))
        if token_mlp == "mix":
            self.mlp = MixFFN((in_dim * 2), int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip((in_dim * 2), int(in_dim * 4))
        else:
            self.mlp = MLP_FFN((in_dim * 2), int(in_dim * 4))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)

        attn = self.attn(norm_1, norm_2)

        residual = torch.cat([x1, x2], dim=2)
        tx = residual + attn
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W)
        return mx
