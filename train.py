import os
import argparse
from datetime import datetime 

import torch
from transformers import AutoTokenizer, AdamW

from utils import config
from dataloaders.get_dataloaders import get_dataloaders
from model.model import BertClassifier
from model.loss import cross_entropy
from model.metrics import accuracy
from trainer.trainer import Trainer


def main(config):

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load data
    trainloader, validloader = get_dataloaders(config, tokenizer)  

    # Experiment tracking and saving
    version = datetime.now().strftime("%Y-%m-%d-%H:%M")
    exp_name = "exp-{}".format(version)
    config.out_dir = config.out_dir + "{}/".format(exp_name)
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # Load model
    model = BertClassifier(out_dim=2)

    # Loss, metrics, optimizer, scheduler
    loss = cross_entropy
    metrics = [accuracy]
    optim = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=300)

    # Load trainer
    trainer = Trainer(model, 
                      loss, 
                      metrics, 
                      optim, 
                      config,
                      trainloader,
                      validloader,
                      scheduler)

    trainer.train()


if __name__=="__main__":
    # Parameters
    parser = argparse.ArgumentParser(description='Arguments for experiment')
    parser.add_argument("--config", help="Experiment config")
    args = parser.parse_args()

    config = config.create_config(args.config)
    print("configs: {}".format(config))

    # Experiment Tracking
    # wandb.init()
    # wandb.config.update(config)

    main(config)