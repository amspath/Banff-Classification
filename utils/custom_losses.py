import typing

import torch


def ordinal_categorical_crossentropy(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Ordinal categorical cross-entropy loss function.
    :param y_true: The true labels (one-hot encoded).
    :param y_pred: The predicted labels (probabilities, softmax of logits).
    :return: The ordinal categorical cross-entropy loss.
    """
    weights = torch.abs(torch.argmax(y_true, dim=1) - torch.argmax(y_pred, dim=1)) / (y_pred.shape[1] - 1)

    return torch.mean((1.0 + weights) * torch.nn.functional.cross_entropy(y_pred, torch.argmax(y_true, dim=1)))


def custom_banff_loss(y_pred: typing.List[torch.Tensor], y_true: typing.List[torch.Tensor]) -> torch.Tensor:
    """
    Custom loss function for the Banff dataset.
    :param y_pred: The predicted labels (probabilities, softmax of logits).
    :param y_true: The true labels (one-hot encoded).
    :return: The loss.
    """
    # The first 2 attributes are binary, so we use binary cross-entropy
    print(y_pred[0])
    print(y_true[0])
    loss = torch.nn.functional.binary_cross_entropy(y_pred[0], y_true[0])
    print(y_pred[1])
    print(y_true[1])
    loss += torch.nn.functional.binary_cross_entropy(y_pred[1], y_true[1])

    # The next 5 attributes are ordinal, so we use ordinal categorical cross-entropy
    for i in range(2, 7):
        loss += ordinal_categorical_crossentropy(y_true[i], y_pred[i])

    # The next 2 attributes are continuous, therefore we use the mean squared error
    loss += torch.nn.functional.mse_loss(y_pred[7], y_true[7])
    loss += torch.nn.functional.mse_loss(y_pred[8], y_true[8])

    # The next 6 attributes are ordinal, so we use ordinal categorical cross-entropy
    for i in range(9, 15):
        loss += ordinal_categorical_crossentropy(y_true[i], y_pred[i])

    return loss
