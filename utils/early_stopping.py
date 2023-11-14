import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, min_epochs: int = 100, patience: int = 10, verbose: bool = True):
        """
        Args:
            min_epochs (int): Earliest epoch possible for stopping.
            patience (int): How many epochs to wait after last time validation loss improved.
            verbose (bool): If True, prints messages for e.g. each validation loss improvement.
        """
        self.min_epochs = min_epochs
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_epoch = 0

    def __call__(self, epoch: int, val_loss: float):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.min_epochs:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
