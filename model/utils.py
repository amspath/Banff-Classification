import typing

from transformer import Transformer
from positional_encoding import *


def load_model_from_config(model_config: typing.Dict[str, typing.Any]) -> Transformer:
    """
    Load a model from a configuration dictionary.
    :param model_config: The configuration dictionary.
    :return: The model.
    """
    positional_encoding = model_config.get("positional_encoding", "none")
    if positional_encoding == "2d":
        positional_encoding = PositionalEncoding2D
    elif positional_encoding == "1d":
        positional_encoding = PositionalEncoding
    elif positional_encoding == "learned":
        positional_encoding = LearnedPositionalEncoding
    else:
        raise ValueError(f"Unknown positional encoding {positional_encoding}")

    model = Transformer(
        output_dims=model_config.get("output_dims", [1, 1, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 5, 5, 4]),
        input_dim=model_config.get("input_dim", 2048),
        embedding_dim=model_config.get("embedding_dim", 768),
        depth=model_config.get("depth", 2),
        num_heads=model_config.get("num_heads", 8),
        mlp_ratio=model_config.get("mlp_ratio", 4.),
        qkv_bias=model_config.get("qkv_bias", True),
        qk_norm=model_config.get("qk_norm", False),
        proj_drop_rate=model_config.get("proj_drop_rate", 0.),
        attn_drop_rate=model_config.get("attn_drop_rate", 0.),
        positional_encoding=positional_encoding,
    )

    return model
