from tqdm import tqdm
import numpy as np
import torch
from .logger import get_logger


class Tester:
    def __init__(self, model, loss, metrics, config, test_loader):
        self.config = config
        self.test_loader = test_loader
        self.logger = get_logger(
            config.out_dir, "tester", config.verbosity, is_train=False
        )
        self.logger.info("configs: {}".format(config))

        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(config.n_gpu)
        self.model = model.to(self.device)
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)

        # load model from checkpoint
        model_path = "{}/{}".format(config.out_dir, "model_best.pth")
        self.logger.info("Loading checkpoint: {} ...".format(model_path))
        checkpoint = torch.load(model_path)
        state_dict = checkpoint["state_dict"]
        self.model.load_state_dict(state_dict)

        self.loss = loss
        self.metrics = metrics

    def test(self):
        if self.config.has_targets:
            total_test_loss = 0
            total_test_metrics = np.zeros(len(self.metrics))
            final = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.test_loader)):
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

                    total_test_loss += loss.item()
                    total_test_metrics += self._eval_metrics(output, target)

                    ans = output.argmax(dim=1)
                    final.extend(ans.detach().cpu().numpy())

            self.logger.info("Testing finished. Saving predictions ...")
            final = [int(x) for x in final]
            np.savetxt(
                "{}/preds.csv".format(self.config.out_dir),
                final,
                delimiter=", ",
                header="preds",
            )

            self.logger.info(
                "Test_loss: {}".format(total_test_loss / len(self.test_loader))
            )
            self.logger.info(
                "Test_metrics: {}".format(
                    total_test_metrics / len(self.test_loader)
                ).tolist()
            )

        else:
            final = []
            for batch_idx, batch in enumerate(tqdm(self.test_loader)):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                segment_ids = batch["token_type_ids"]

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                output = self.model(batch=(input_ids, attention_mask, segment_ids))

                ans = output.argmax(dim=1)
                final.extend(ans.detach().cpu().numpy())

            self.logger.info("Testing finished. Saving predictions ...")
            final = [int(x) for x in final]
            np.savetxt(
                "{}/preds.csv".format(self.config.out_dir),
                final,
                delimiter=", ",
                header="preds",
            )

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There's no GPU available on this machine,"
                "training will be performed on CPU."
            )
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU's configured to use is {}, but only {} are available "
                "on this machine.".format(n_gpu_use, n_gpu)
            )
            n_gpu_use = n_gpu
        device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics
