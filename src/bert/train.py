import os.path as osp
import logging
import torch
import numpy as np

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, get_scheduler

from seqlbtoolkit.training.train import BaseTrainer
from seqlbtoolkit.data import label_to_span
from seqlbtoolkit.io import save_json


from .dataset import Dataset
from .collator import DataCollator
from .container import CheckpointContainer
from src.core.metrics import get_ner_metrics

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Bert trainer used for training BERT for token classification (sequence labeling)
    """

    def __init__(self, config, collate_fn=None, training_dataset=None, valid_dataset=None, test_dataset=None):
        if not collate_fn:
            tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name_or_path)
            collate_fn = DataCollator(tokenizer)

        super().__init__(
            config=config,
            training_dataset=training_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            collate_fn=collate_fn,
        )

        if self.valid_dataset:
            self._checkpoint_container = CheckpointContainer("metric-larger")
        else:
            self._checkpoint_container = CheckpointContainer("always")

        self.initialize()

    def initialize_model(self):
        model_config = AutoConfig.from_pretrained(self._config.bert_model_name_or_path)
        model_config.hidden_dropout_prob = self._config.hidden_dropout_prob
        model_config.attention_probs_dropout_prob = self._config.attention_probs_dropout_prob
        model_config.num_labels = self._config.n_lbs

        self._model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self._config.bert_model_name_or_path,
            config=model_config,
        )
        return self

    def initialize_optimizer(self, optimizer=None):
        """
        Initialize training optimizer
        """
        if optimizer is not None:
            self._optimizer = optimizer
        else:
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(), lr=self._config.lr, weight_decay=self.config.weight_decay
            )
        return self

    def initialize_scheduler(self):
        """
        Initialize learning rate scheduler
        """
        num_update_steps_per_epoch = int(np.ceil(len(self._training_dataset) / self._config.batch_size))
        num_warmup_steps = int(
            np.ceil(num_update_steps_per_epoch * self._config.warmup_ratio * self._config.num_epochs)
        )
        num_training_steps = int(np.ceil(num_update_steps_per_epoch * self._config.num_epochs))

        self._scheduler = get_scheduler(
            self._config.lr_scheduler_type,
            self._optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return self

    @property
    def training_dataset(self):
        return self._training_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    def run(self):
        self._model.to(self._device)

        # ----- start training process -----
        logger.info("Start training...")
        for epoch_i in range(self._config.num_epochs):
            logger.info("------")
            logger.info(f"Epoch {epoch_i + 1} of {self._config.num_epochs}")

            training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)

            train_loss = self.training_step(training_dataloader)
            logger.info("Training loss: %.4f" % train_loss)

            self.eval_and_save()

        test_results_full_match, test_report_full_match = self.test()
        logger.info("--------")
        logger.info("Test results (full match):")
        self.log_results(test_results_full_match)

        test_results_partial_match, test_report_partial_match = self.test(allow_partial_match=True)
        logger.info("--------")
        logger.info("Test results (partial_match):")
        self.log_results(test_results_partial_match)

        if self.config.report_dir:
            logger.info(f"Saving reports to {self.config.report_dir}...")
            for item in ("fp", "fn", "tp"):
                save_json(
                    test_report_partial_match[item],
                    osp.join(self.config.report_dir, f"report_partial_match_{item}.json"),
                )
                save_json(
                    test_report_full_match[item], osp.join(self.config.report_dir, f"report_full_match_{item}.json")
                )

            save_json(
                {"partial_match": test_results_partial_match, "full_match": test_results_full_match},
                osp.join(self.config.report_dir, "metrics.json"),
            )

        return None

    def training_step(self, data_loader):
        """
        Implement each training loop
        """
        train_loss = 0

        self._model.train()
        self._optimizer.zero_grad()

        n_tks = 0
        for batch in tqdm(data_loader):
            # get data
            batch.to(self._device)

            # training step
            loss = self._model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
            ).loss
            loss.backward()
            # track loss

            n_tks += torch.sum(batch.labels != -100).cpu()
            train_loss += loss.detach().cpu() * torch.sum(batch.labels != -100).cpu()

            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()

        return train_loss / n_tks

    def evaluate(self, dataset: Dataset, allow_partial_match=False):
        data_loader = self.get_dataloader(dataset)
        self._model.to(self._device)
        self._model.eval()

        pred_lbs = list()
        with torch.no_grad():
            for batch in data_loader:
                batch.to(self._device)

                logits = self._model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                ).logits
                pred_ids = logits.argmax(-1).detach().cpu()

                pred_lb_batch = [
                    [self.config.bio_label_types[i] for i in pred[lbs >= 0]]
                    for lbs, pred in zip(batch.labels.cpu(), pred_ids)
                ]
                pred_lbs += pred_lb_batch

        pred_spans = [label_to_span(lbs) for lbs in pred_lbs]
        gt_spans = [label_to_span(lbs) for lbs in dataset.lbs]

        metric, report = get_ner_metrics(
            pred_list=pred_spans,
            gt_list=gt_spans,
            entity_types=self.config.entity_types,
            tks_list=dataset.text,
            ids_list=dataset.ids,
            allow_partial_match=allow_partial_match,
        )
        return metric, report

    def eval_and_save(self):
        """
        Evaluate the model and save it if its performance exceeds the previous highest
        """

        valid_results = None
        if self.valid_dataset:
            valid_results, _ = self.evaluate(self.valid_dataset)

            logger.info("Validation results:")
            self.log_results(valid_results["micro-average"])

        # ----- check model performance and update buffer -----
        if self._checkpoint_container.check_and_update(
            self._model, valid_results["micro-average"]["f1"] if valid_results else None
        ):
            logger.info("Model buffer is updated!")

        self._model.to(self._device)

        return None

    @staticmethod
    def log_results(metrics):
        if isinstance(metrics, dict):
            for key, val in metrics.items():
                if isinstance(val, dict):
                    logger.info(f"[{key}]")
                    for k, v in val.items():
                        logger.info(f"  {k}: {v:.4f}." if isinstance(v, float) else f"  {k}: {v}")
                else:
                    logger.info(f"{key}: {val:.4f}." if isinstance(val, float) else f"{key}: {val}")

    def test(self, dataset=None, allow_partial_match=False):
        if dataset is None:
            dataset = self._test_dataset
        self._model.load_state_dict(self._checkpoint_container.state_dict)
        metrics, report = self.evaluate(dataset, allow_partial_match=allow_partial_match)
        return metrics, report
