import math
from functools import partial
from typing import Type

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from timm.layers import Mlp


class Transformer(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            input_dim: int = 2048,
            embedding_dim: int = 512,
            depth: int = 2,
            num_heads: int = 8,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_prefix_tokens = 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.projection = nn.Sequential(nn.Linear(input_dim, embedding_dim, bias=True), nn.ReLU())
        self.transformer = nn.Sequential(*[
            Block(dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                  proj_drop=proj_drop_rate, attn_drop=attn_drop_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  act_layer=nn.GELU, mlp_layer=Mlp) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(self.embedding_dim, num_classes), nn.ReLU())

    def forward_features(self, x, coords=None):
        batch, n_patches, input_size = x.shape

        x = self.projection(x)
        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
        x = self.transformer(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        x = x[:, 0]
        return self.head(x)

    def forward(self, x, coords):
        x = self.forward_features(x, coords)
        x = self.forward_head(x)
        return x
