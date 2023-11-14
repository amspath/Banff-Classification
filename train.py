import os

import click
import torch
import yaml

from banff_dataset import BanffDataset
from utils.training import data_sampling, run_train_eval_loop
from model.utils import load_model_from_config


@click.command()
@click.option("--features-dir", type=click.Path(exists=True),
              required=True,
              help="The directory that stores features extracted by the backbone.")
@click.option("--banff-scores-csv", type=click.Path(exists=True), required=True,
              help="The CSV file that contains the BANFF scores for each slide.")
@click.option("--save-checkpoints", type=bool, default=True,
              help="Whether to save checkpoints during training.")
@click.option("--checkpoints-dir", type=click.Path(exists=False), required=True,
              help="The directory that stores the checkpoints.")
@click.option("--log-dir", type=click.Path(exists=False), required=True,
              help="The directory that stores the logs.")
@click.option("--hyperparameters-config-filepath", type=click.Path(exists=True), required=True,
              help="The YAML file that contains the hyperparameters for training and model architecture.")
@click.option("--num-workers", type=int, default=4, help="Number of workers for the data loader")
@click.option("--training-set-list", type=click.Path(exists=True),
              help="The list of slides to use as training set.")
@click.option("--validation-set-list", type=click.Path(exists=True),
              help="The list of slides to use as validation set.")
@click.option("--model-checkpoint", type=click.Path(exists=True), default="", required=False,
              help="The checkpoint of the model to load.")
@click.option("--device", type=torch.device, default="cuda:0", help="The device on which to run the training.")
def run(features_dir: str, banff_scores_csv: str, save_checkpoints: bool, checkpoints_dir: str, log_dir: str,
        hyperparameters_config_filepath: str, num_workers: int, training_set_list: str, validation_set_list: str,
        model_checkpoint: str, device: torch.device):
    # Check if the checkpoints directory exists and if not create it
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory {checkpoints_dir} does not exist. Creating it...")
        os.makedirs(checkpoints_dir)

    # Check if the logs directory exists and if not create it
    if not os.path.exists(log_dir):
        print(f"Logs directory {log_dir} does not exist. Creating it...")
        os.makedirs(log_dir)

    # Load the hyperparameters config file
    with open(hyperparameters_config_filepath, "r") as hyperparameters_config_file:
        model_config = yaml.load(stream=hyperparameters_config_file, Loader=yaml.FullLoader)

    # Load the list of slides to use as training set
    with open(training_set_list, "r") as training_set_list_file:
        train_slides = training_set_list_file.read().splitlines()

    # Load the list of slides to use as validation set
    with open(validation_set_list, "r") as validation_set_list_file:
        val_slides = validation_set_list_file.read().splitlines()

    # Load the Banff dataset
    banff_dataset_train = BanffDataset(data_dir=features_dir, banff_scores_csv_filepath=banff_scores_csv,
                                       device=device, slides_to_load=train_slides)
    banff_dataset_val = BanffDataset(data_dir=features_dir, banff_scores_csv_filepath=banff_scores_csv,
                                     device=device, slides_to_load=val_slides)

    # Define the data sampling
    train_loader, val_loader = data_sampling(training_set=banff_dataset_train, validation_set=banff_dataset_val,
                                             workers=num_workers)

    training_parameters = model_config["training_parameters"]
    model_hyperparameters = model_config["model_hyperparameters"]

    # Load the model
    model = load_model_from_config(model_config=model_hyperparameters).to(device)
    if model_checkpoint != "":
        print(f"Loading model checkpoint {model_checkpoint}")
        model.load_state_dict(torch.load(model_checkpoint))

    # Run the training and evaluation loop
    run_train_eval_loop(model=model, train_loader=train_loader, val_loader=val_loader, writer_path=log_dir,
                        save_checkpoints=save_checkpoints, checkpoints_dir=checkpoints_dir, device=device,
                        hparams=training_parameters)


if __name__ == "__main__":
    run()
