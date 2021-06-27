import time
import numpy as np
import torch
from tqdm import tqdm

from .trainer_base import BaseTrainer
from utils import AverageMeter


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(
        self,
        model,
        loss,
        metrics,
        optimizer,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
    ):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        losses = AverageMeter()
        start_time = time.time()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        with tqdm(self.data_loader, unit="batch") as pbar:
            for batch_idx, batch in enumerate(pbar):
                pbar.set_description("Epoch {}".format(epoch))
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                segment_ids = batch["token_type_ids"]
                target = batch["target"]

                target = target.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                output = self.model(batch=(input_ids, attention_mask, segment_ids))
                loss = self.loss(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, target)

                losses.update(loss.item(), target.size(0))
                pbar.set_postfix(loss="{:.3f}({:.3f})".format(losses.val, losses.avg))
        
        log = {
            "loss": total_loss / len(self.data_loader),
            "metrics": (total_metrics / len(self.data_loader)).tolist(),
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.logger.info("Epoch {} time taken: {}".format(epoch, time.time() - start_time))
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            with tqdm(self.valid_data_loader, unit="batch") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    pbar.set_description("Epoch {} (Valid)".format(epoch))
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    segment_ids = batch["token_type_ids"]
                    target = batch["target"]

                    target = target.to(self.device)
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    segment_ids = segment_ids.to(self.device)

                    output = self.model(batch=(input_ids, attention_mask, segment_ids))
                    loss = self.loss(output, target)

                    total_val_loss += loss.item()
                    total_val_metrics += self._eval_metrics(output, target)

        return {
            "val_loss": total_val_loss / len(self.valid_data_loader),
            "val_metrics": (total_val_metrics / len(self.valid_data_loader)).tolist(),
        }