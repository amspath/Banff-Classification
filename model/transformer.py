import typing
from functools import partial
from typing import Type, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from timm.models.vision_transformer import LayerScale, DropPath
from timm.layers.mlp import Mlp


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.head_dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        x, _ = self.scaled_dot_product_attention(q, k, v, mask)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, args: typing.Dict[str, typing.Any]):
        x = args["x"]
        mask = args["mask"]
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask=mask)))
        x = x + self.drop_path

        return x


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
        self.projection = nn.Sequential(nn.Linear(input_dim, embedding_dim, bias=False), nn.ReLU())
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
                         coords: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]],
                         mask: torch.Tensor = None) -> torch.Tensor:
        batch, n_patches, input_size = x.shape

        x = self.projection(x)

        if self.positional_encoding is not None:
            if isinstance(coords, tuple):
                x = x + self.positional_encoding(coords[0], coords[1])
            else:
                x = x + self.positional_encoding(coords)

        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
        # Modify the mask to account for the cls token
        if mask is not None:
            print(mask)
            mask = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=mask.device), mask), dim=1)
        x = self.transformer({"x": x, "mask": mask})
        x = self.norm(x)

        return x

    def forward_head(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        x = x[:, 0]

        return [head(x) for head in self.heads]

    def forward(self, x: torch.Tensor, coords: typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]],
                mask: torch.Tensor = None) \
            -> typing.List[torch.Tensor]:
        x = self.forward_features(x, coords, mask)
        x = self.forward_head(x)

        return x
