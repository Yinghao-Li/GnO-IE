import os.path as osp
import logging

from typing import Optional
from dataclasses import dataclass, field

from src.core.args import Arguments as CoreArguments, Config as CoreConfig

logger = logging.getLogger(__name__)


@dataclass
class Arguments(CoreArguments):
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- manage directories and IO ---
    model_dir: Optional[str] = field(
        default="",
        metadata={"help": "Path to the pretrained GPT model"},
    )

    # --- data arguments ---
    separate_overlength_sequences: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether split the overlength sequences into several smaller pieces"
            "according to their BERT token sequence lengths."
        },
    )
    max_tokens: Optional[int] = field(default=512, metadata={"help": "The maximum length of a BERT token sequence."})

    def __post_init__(self):
        if isinstance(self.entities_to_exclude, str):
            self.entities_to_exclude = [self.entities_to_exclude]
        if isinstance(self.tasks, str):
            self.tasks = [self.tasks]

        self.data_dir = osp.join(self.data_dir, self.dataset)
        self.result_dir = osp.join(self.result_dir, self.dataset)
        self.report_dir = osp.join(self.report_dir, self.dataset)

        return self


@dataclass
class Config(Arguments, CoreConfig):
    pass
