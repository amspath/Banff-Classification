import typing
from functools import partial
from typing import Type

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from timm.layers import Mlp


class Transformer(nn.Module):
    def __init__(
            self,
            output_dims: typing.List[int] = None,
            input_dim: int = 2048,
            embedding_dim: int = 512,
            depth: int = 2,
            num_heads: int = 8,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            positional_encoding: Type[nn.Module] = None,
    ):
        super().__init__()

        if output_dims is None:
            output_dims = [1, 1, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 5, 5, 4]

        self.output_dims = output_dims
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
        # Multiple heads, each of which is a linear layer followed by a sigmoid when the output dimension is 1, or a
        # softmax when the output dimension is > 1
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(embedding_dim, output_dim),
                                                  nn.Sigmoid() if output_dim == 1 else nn.Softmax(dim=-1))
                                    for output_dim in output_dims])

        self.positional_encoding = positional_encoding

    def forward_features(self, x: torch.Tensor,
                         coords: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        batch, n_patches, input_size = x.shape

        x = self.projection(x)

        if self.positional_encoding is not None:
            if isinstance(coords, tuple):
                x = x + self.positional_encoding(coords[0], coords[1])
            else:
                x = x + self.positional_encoding(coords)

        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
        x = self.transformer(x)
        x = self.norm(x)

        return x

    def forward_head(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        x = x[:, 0]

        return [head(x) for head in self.heads]

    def forward(self, x: torch.Tensor, coords: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]) \
            -> typing.List[torch.Tensor]:
        x = self.forward_features(x, coords)
        x = self.forward_head(x)

        return x
