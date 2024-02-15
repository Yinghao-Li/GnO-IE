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
    model_dtype: str = field(
        default="auto",
        metadata={"help": "Data type of the model"},
    )
    max_tokens: Optional[int] = field(default=4096, metadata={"help": "The maximum length of a token sequence."})
    max_model_len: Optional[int] = field(default=None, metadata={"help": "The maximum length of a model."})


@dataclass
class Config(Arguments, CoreConfig):
    pass
