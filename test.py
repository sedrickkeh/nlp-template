import configargparse

import torch
from transformers import AutoTokenizer

from utils.config import *
from dataloaders.get_dataloaders import get_testloader
from model.model import BertClassifier
from model.loss import cross_entropy
from model.metrics import accuracy
from trainer.tester import Tester


def main(config):

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load data
    testloader = get_testloader(config, tokenizer)

    # load model
    model = BertClassifier(out_dim=2)
    loss = cross_entropy
    metrics = accuracy

    # Load tester
    tester = Tester(
        model,
        loss,
        metrics,
        config,
        testloader,
    )

    tester.test()


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="Arguments for testing")
    parser.add_argument(
        "--config_path", is_config_file=True, default="configs/config-test.yaml"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True
    )  # Need to supply path to model

    test_args(parser)
    args = parser.parse_args()

    main(args)
