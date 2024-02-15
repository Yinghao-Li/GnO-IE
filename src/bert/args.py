import os.path as osp
import json
import torch
import logging

from typing import Optional
from dataclasses import dataclass, field
from transformers.file_utils import cached_property

from seqlbtoolkit.training.config import NERConfig
from seqlbtoolkit.data import entity_to_bio_labels

from src.core.args import Arguments as CoreArguments

logger = logging.getLogger(__name__)


@dataclass
class Arguments(CoreArguments):
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- manage directories and IO ---
    bert_model_name_or_path: Optional[str] = field(
        default="microsoft/deberta-v3-base",
        metadata={
            "help": "Path to pretrained BERT model or model identifier from huggingface.co/models; "
            "Used to construct BERT embeddings if not exist"
        },
    )

    # --- data arguments ---
    separate_overlength_sequences: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether split the overlength sequences into several smaller pieces"
            "according to their BERT token sequence lengths."
        },
    )
    max_seq_length: Optional[int] = field(
        default=512, metadata={"help": "The maximum length of a BERT token sequence."}
    )

    # --- training arguments ---
    warmup_ratio: Optional[int] = field(
        default=0.1, metadata={"help": "ratio of warmup steps for learning rate scheduler"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={
            "help": "Default as `linear`. See the documentation of "
            "`transformers.SchedulerType` for all possible values"
        },
    )
    hidden_dropout_prob: Optional[float] = field(default=0.1, metadata={"help": "hidden dropout probability"})
    attention_probs_dropout_prob: Optional[float] = field(
        default=0.1, metadata={"help": "attention dropout probability"}
    )
    batch_size: Optional[int] = field(default=16, metadata={"help": "model training batch size"})
    num_epochs: Optional[int] = field(default=100, metadata={"help": "number of denoising model training epochs"})
    lr: Optional[float] = field(default=0.00005, metadata={"help": "learning rate"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "strength of weight decay"})
    no_cuda: Optional[bool] = field(default=False, metadata={"help": "Disable CUDA even when it is available"})
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    warmup_ratio: Optional[int] = field(
        default=0.1, metadata={"help": "ratio of warmup steps for learning rate scheduler"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={
            "help": "Default as `linear`. See the documentation of "
            "`transformers.SchedulerType` for all possible values"
        },
    )
    debug: Optional[bool] = field(default=False, metadata={"help": "Debugging mode with fewer training data"})

    def __post_init__(self):
        self.data_dir = osp.join(self.data_dir, self.prompt_format)
        super().__post_init__()

    # The following three functions are copied from transformers.training_args
    @cached_property
    def _setup_devices(self) -> "torch.device":
        if self.no_cuda or not torch.cuda.is_available():
            device = torch.device("cpu")
            self._n_gpu = 0
        else:
            device = torch.device("cuda")
            self._n_gpu = 1

        return device

    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    def n_gpu(self) -> "int":
        """
        The number of GPUs used by this process.
        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu


@dataclass
class Config(Arguments, NERConfig):
    def get_meta(self):
        # Load meta if exist
        meta_dir = osp.join(self.data_dir, "meta.json")

        if not osp.isfile(meta_dir):
            raise FileNotFoundError("Meta file does not exist!")

        with open(meta_dir, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        self.entity_types = meta_dict["entity_types"]
        self.bio_label_types = entity_to_bio_labels(meta_dict["entity_types"])

        return self

    @property
    def n_ents(self):
        return len(self.entity_types)

    @property
    def n_lbs(self):
        return len(self.bio_label_types)
