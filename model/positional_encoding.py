import torch
from torch import nn


class PositionalEncoding2D(nn.Module):
    """
    This is the implementation of an idea of my own on how to extend the positional encoding to 2D.
    Essentially, the idea is, at each position (x, y) in the image, to have as positional embedding the average of the
    positional embeddings of the x-th row and the y-th column. I have not found any other implementation of this idea
    online, so I am not sure if it is a good idea or not.
    """

    def __init__(self, encoding_dim: int, max_x_dim: int = 400, max_y_dim: int = 400, device: str = "cuda:0"):
        """
        Constructor for PositionalEncoding2D.

        :param encoding_dim: the dimension of the positional encoding
        :param max_x_dim: the maximum dimension of the x-axis
        :param max_y_dim: the maximum dimension of the y-axis
        :param device: the device on which to store the positional encoding
        """
        super(PositionalEncoding2D, self).__init__()

        self.encoding_dim = encoding_dim
        self.max_x_dim = max_x_dim
        self.max_y_dim = max_y_dim

        self.encoding = torch.zeros(max_x_dim, max_y_dim, encoding_dim, device=device)
        self.encoding.requires_grad = False

        self.projector = nn.Linear(encoding_dim, encoding_dim)  # This way it can be trained
        self.relu = nn.ReLU()

        encoding_x = torch.zeros((max_x_dim, encoding_dim), device=device).float()
        encoding_y = torch.zeros((max_y_dim, encoding_dim), device=device).float()

        pos_x = torch.arange(0, max_x_dim, device=device)
        pos_x = pos_x.float().unsqueeze(dim=1)

        pos_y = torch.arange(0, max_y_dim, device=device)
        pos_y = pos_y.float().unsqueeze(dim=1)

        _2i = torch.arange(0, encoding_dim, step=2, device=device).float()

        encoding_x[:, 0::2] = torch.sin(pos_x / (10000 ** (_2i / encoding_dim)))
        encoding_x[:, 1::2] = torch.cos(pos_x / (10000 ** (_2i / encoding_dim)))

        encoding_y[:, 0::2] = torch.sin(pos_y / (10000 ** (_2i / encoding_dim)))
        encoding_y[:, 1::2] = torch.cos(pos_y / (10000 ** (_2i / encoding_dim)))

        for i in range(max_x_dim):
            for j in range(max_y_dim):
                self.encoding[i, j, :] = (encoding_x[i, :] + torch.flip(encoding_y[j, :], dims=[0]))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Convert the coordinates to int, making sure it's a torch tensor
        x = x * (self.max_x_dim - 1)
        y = y * (self.max_y_dim - 1)

        x = x.long()
        y = y.long()

        return self.relu(self.projector(self.encoding[x, y, :]))


class PositionalEncoding(nn.Module):
    """
    Implementation of the sinusoid encoding proposed in the paper "Attention is all you need". This can be used to encode
    the position of the patches in the image, as if they were words in a sentence.
    """

    def __init__(self, encoding_dim: int, max_n_patches: int, device):
        """
        Constructor for PositionalEncoding.

        :param encoding_dim: the dimension of the positional encoding
        :param max_n_patches: the maximum number of patches in the image
        :param device: the device on which to store the positional encoding
        """
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_n_patches, encoding_dim, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_n_patches, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, encoding_dim, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / encoding_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / encoding_dim)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_n_patches: int, encoding_dim: int, pad_index: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_n_patches, encoding_dim, pad_index)
        self.max_n_patches = max_n_patches
        self.pad_index = pad_index

    def forward(self, input):
        positions = self._make_positions(input, self.pad_index)
        return self.embedding(positions)

    def max_positions(self):
        if self.pad_index is not None:
            return self.num_embeddings - self.pad_index - 1
        else:
            return self.num_embeddings

    def _make_positions(self, tensor, pad_index: int):
        masked = tensor.ne(pad_index).long()
        return torch.cumsum(masked, dim=1) * masked + pad_index
