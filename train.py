import os
import configargparse
from datetime import datetime

import torch
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from utils.config import *
from dataloaders.get_dataloaders import get_dataloaders
from model.model import BertClassifier
from model.loss import cross_entropy
from model.metrics import accuracy
from trainer.trainer import Trainer


def main(config):

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load data
    trainloader, validloader = get_dataloaders(config, tokenizer)

    # Experiment tracking and saving
    version = datetime.now().strftime("%Y-%m-%d-%H:%M")
    exp_name = "exp-{}".format(version)
    config.out_dir = config.out_dir_root + "{}/".format(exp_name)
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # Load model
    model = BertClassifier(out_dim=2)

    # Loss, metrics, optimizer, scheduler
    loss = cross_entropy
    metrics = [accuracy]
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=len(trainloader) * config.epochs,
    )

    # Load trainer
    trainer = Trainer(
        model,
        loss,
        metrics,
        optimizer,
        config,
        trainloader,
        validloader,
        scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    # Parameters
    parser = configargparse.ArgumentParser(description="Arguments for experiment")
    parser.add_argument(
        "--config_path", is_config_file=True, default="configs/config-train.yaml"
    )
    data_args(parser)  # data configs
    model_args(parser)  # model configs
    train_args(parser)  # training, logging configs
    args = parser.parse_args()

    # Experiment Tracking
    # wandb.init()
    # wandb.config.update(args)

    main(args)
