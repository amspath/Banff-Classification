import os
import random
import time
import typing

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
)
from torch.utils.tensorboard import SummaryWriter

from banff_dataset import BanffDataset, collate
from model.transformer import Transformer
from utils.early_stopping import EarlyStopping
from utils.custom_losses import (custom_banff_loss)


def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_sampling(training_set: BanffDataset, validation_set: BanffDataset, workers: int) -> (
        typing.Tuple)[DataLoader, DataLoader]:
    """
    Returns the data loaders for training and validation.
    :param training_set: The training set.
    :param validation_set: The validation set.
    :param workers: The number of workers for the data loader.
    """
    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
        dataset=training_set,
        batch_size=32,  # model expects one bag of features at the time.
        shuffle=False,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        dataset=validation_set,
        batch_size=32,  # model expects one bag of features at the time.
        sampler=SequentialSampler(validation_set),
        collate_fn=collate,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()

    avg_loss = 0.0
    with torch.no_grad():
        for batch_idx, (features, masks, coords, scores) in enumerate(loader):
            features, masks, coords = features.to(device), masks.to(device), coords.to(device)
            for i in range(len(scores)):
                scores[i] = scores[i].to(device)

            predictions = model(features, coords=None, mask=masks)
            loss = custom_banff_loss(predictions, scores)
            avg_loss += loss.item()

    avg_loss /= len(loader)

    return avg_loss


def run_train_eval_loop(model: Transformer, train_loader: torch.utils.data.DataLoader,
                        val_loader: torch.utils.data.DataLoader, writer_path: str, save_checkpoints: bool,
                        checkpoints_dir: str, device: torch.device, hparams: typing.Dict[str, typing.Any]):
    writer = SummaryWriter(writer_path)

    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_trainable_params} parameters")

    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=hparams["initial_lr"],
        weight_decay=hparams["adam_weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams["cosine_annealing_n_iters"],
                                                           eta_min=hparams["cosine_annealing_min_lr"])

    early_stop_tracker = EarlyStopping(
        patience=hparams["early_stopping_patience"],
        min_epochs=hparams["early_stopping_min_epochs"],
        verbose=True,
    )

    for epoch in range(hparams["max_epochs"]):
        model.train()
        epoch_start_time = time.time()
        train_loss = 0.0

        batch_start_time = time.time()
        for batch_idx, (features, masks, coords, scores) in enumerate(train_loader):
            data_load_duration = time.time() - batch_start_time

            features, masks, coords = features.to(device), masks.to(device), coords.to(device)
            for i in range(len(scores)):
                scores[i] = scores[i].to(device)

            predictions = model(features, coords=None, mask=masks)

            loss = custom_banff_loss(predictions, scores)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_duration = time.time() - batch_start_time
            batch_start_time = time.time()

            print(
                f"epoch {epoch}, "
                f"batch {batch_idx}, "
                f"batch took: {batch_duration:.2f}s, "
                f"data loading: {data_load_duration:.2f}s, "
                f"loss: {loss.item() :.4f}, "
            )
            writer.add_scalar("data_load_duration", data_load_duration, epoch)
            writer.add_scalar("batch_duration", batch_duration, epoch)

        epoch_duration = time.time() - epoch_start_time
        print(f"Finished training on epoch {epoch} in {epoch_duration:.2f}s")

        train_loss /= len(train_loader)

        writer.add_scalar("epoch_duration", epoch_duration, epoch)
        writer.add_scalar("LR", get_lr(optimizer), epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)

        print("Evaluating model on validation set...")
        val_loss = evaluate_model(model, val_loader, device)

        writer.add_scalar("Loss/val", val_loss, epoch)

        if save_checkpoints:
            torch.save(
                model.state_dict(),
                os.path.join(checkpoints_dir, f"{epoch}_checkpoint.pt"),
            )

        # Update LR decay.
        scheduler.step()

        if early_stop_tracker.early_stop:
            print(
                f"Early stop criterion reached. Broke off training loop after epoch {epoch}."
            )
            print("Best epoch was %d with best validation balanced accuracy equal to %f" %
                  (early_stop_tracker.best_epoch, early_stop_tracker.best_score))
            break

    writer.close()
