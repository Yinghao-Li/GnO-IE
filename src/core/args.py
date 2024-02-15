"""
# Modified: February 15th, 2024
# ---------------------------------------
# Description: Base classes for arguments and configurations.
"""

import os.path as osp
import json
import logging
from typing import Optional
from dataclasses import dataclass, field, asdict
from seqlbtoolkit.io import prettify_json


logger = logging.getLogger(__name__)

__all__ = ["Arguments", "Config"]


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    data_dir: str = field(default="./data/", metadata={"help": "Path to data directory"})
    dataset: str = field(default="", metadata={"help": "Dataset name"})
    result_dir: str = field(default="./output/", metadata={"help": "Output directory"})
    report_dir: str = field(default="./report/", metadata={"help": "Report directory"})
    conflict_data_dir: str = field(default="./output/conflict_data/", metadata={"help": "Conflict data directory"})
    conflict_resolution_dir: str = field(
        default="./output/conflict_resolve/", metadata={"help": "Conflict resolve directory"}
    )
    gpt_resource_path: str = field(default="./resources/gpt35.json", metadata={"help": "Path to GPT resources"})
    bert_data_dir: str = field(default="./output/data/bert/", metadata={"help": "BERT data directory"})
    log_path: Optional[str] = field(
        default=None, metadata={"help": "the directory of the log file. Set to '' to disable logging"}
    )

    prompt_format: str = field(
        default="ino",
        metadata={
            "help": (
                "Prompt format: ino: Identify and Organize; "
                "one_step: Identify and Organize in one step; "
                "aeio: All-Entity-in-One."
            ),
            "choices": ["one_step", "ino", "aeio"],
        },
    )
    entities_to_exclude: list[str] = field(
        default=None, metadata={"nargs": "*", "help": "Entities to exclude from the training data"}
    )

    tasks: str = field(
        default="generation",
        metadata={
            "help": "Whether to only evaluate the model.",
            "nargs": "*",
            "choices": ["generation", "evaluation", "conflict-detection", "conflict-resolution", "save-for-bert"],
        },
    )

    apply_conflict_resolution_in_eval: bool = field(
        default=False, metadata={"help": "Whether to apply conflict resolution in evaluation."}
    )
    disable_result_removal: bool = field(
        default=False, metadata={"help": "Whether to disable the removal of the generated results."}
    )
    disable_cot: bool = field(default=False, metadata={"help": "Whether to disable zero-shot CoT reasoning."})
    n_test_samples: int = field(
        default=None, metadata={"help": "Number of test samples to evaluate. Set to `None` to evaluate all."}
    )
    save_report: bool = field(default=False, metadata={"help": "Whether to save the evaluation report."})

    def __post_init__(self):
        if isinstance(self.entities_to_exclude, str):
            self.entities_to_exclude = [self.entities_to_exclude]
        if isinstance(self.tasks, str):
            self.tasks = [self.tasks]

        self.data_dir = osp.join(self.data_dir, self.dataset)
        self.result_dir = osp.join(self.result_dir, self.prompt_format, self.dataset)

        report_prompt_format = (
            self.prompt_format if not self.apply_conflict_resolution_in_eval else self.prompt_format + "_cr"
        )
        self.report_dir = osp.join(self.report_dir, report_prompt_format, self.dataset)
        self.bert_data_dir = osp.join(self.bert_data_dir, report_prompt_format, self.dataset)

        self.conflict_data_dir = osp.join(self.conflict_data_dir, self.prompt_format, self.dataset)
        self.conflict_resolution_dir = osp.join(self.conflict_resolution_dir, self.prompt_format, self.dataset)

        return self


@dataclass
class Config(Arguments):
    test_path: str = None

    # --- Properties and Functions ---
    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        else:
            raise ValueError("`Config` can only be subscribed by str!")

    def as_dict(self):
        return asdict(self)

    def from_args(self, args):
        """
        Initialize configuration from arguments

        Parameters
        ----------
        args: arguments (parent class)

        Returns
        -------
        self (type: BertConfig)
        """
        arg_elements = {
            attr: getattr(args, attr)
            for attr in dir(args)
            if not callable(getattr(args, attr)) and not attr.startswith("__") and not attr.startswith("_")
        }
        for attr, value in arg_elements.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass

        self.check()

        return self

    def check(self):
        self.test_path = osp.join(self.data_dir, "test.json")
        if self.n_test_samples == 0:
            self.n_test_samples = None
        return self

    def save(self, file_dir: str, file_name: Optional[str] = "config"):
        """
        Save configuration to file

        Parameters
        ----------
        file_dir: file directory
        file_name: file name (suffix free)

        Returns
        -------
        self
        """
        if osp.isdir(file_dir):
            file_path = osp.join(file_dir, f"{file_name}.json")
        elif osp.isdir(osp.split(file_dir)[0]):
            file_path = file_dir
        else:
            raise FileNotFoundError(f"{file_dir} does not exist!")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"Cannot save config file to {file_path}; " f"encountered Error {e}")
            raise e
        return self

    def load(self, file_dir: str, file_name: Optional[str] = "config"):
        """
        Load configuration from stored file

        Parameters
        ----------
        file_dir: file directory
        file_name: file name (suffix free)

        Returns
        -------
        self
        """
        if osp.isdir(file_dir):
            file_path = osp.join(file_dir, f"{file_name}.json")
            assert osp.isfile(file_path), FileNotFoundError(f"{file_path} does not exist!")
        elif osp.isfile(file_dir):
            file_path = file_dir
        else:
            raise FileNotFoundError(f"{file_dir} does not exist!")

        logger.info(f"Setting {type(self)} parameters from {file_path}.")

        with open(file_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        for attr, value in config.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self

    def log(self):
        """
        Log all configurations
        """
        elements = {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not (attr.startswith("__") or attr.startswith("_"))
        }
        logger.info(f"Configurations:\n{prettify_json(json.dumps(elements, indent=2), collapse_level=2)}")

        return self
