"""
# Modified: February 15th, 2024
# ---------------------------------------
# Description: This module provides functionality to manage and save model checkpoints
               based on certain criteria, such as when a metric value improves.
"""

import copy
import regex
import torch
import logging
import numpy as np
from typing import Optional, Union
from enum import Enum


logger = logging.getLogger(__name__)

__all__ = ["UpdateCriteria", "CheckpointContainer"]


class StrEnum(str, Enum):
    @classmethod
    def options(cls):
        opts = list()
        for k, v in cls.__dict__.items():
            if not k.startswith("_") and k != "options":
                opts.append(v.value)
        return opts


class UpdateCriteria(StrEnum):
    """
    Enumeration for the update criteria.

    - metric_smaller: Update the container dict when metric value becomes smaller.
    - metric_larger: Update the container dict when metric value becomes larger.
    - always: Always update the container dict.
    """

    metric_smaller = "metric-smaller"
    metric_larger = "metric-larger"
    always = "always"


class CheckpointContainer:
    """
    A container to manage model checkpoints.

    This class tracks and updates model checkpoints based on
    a specified update criteria.
    """

    def __init__(self, update_criteria: Optional[Union[str, UpdateCriteria]] = "always"):
        """
        Initialize the checkpoint container.

        Parameters
        ----------
        update_criteria : str or UpdateCriteria
            Criterion to determine whether to update the model.
            Choices:
            - "always": Always update the container state dict.
            - "metric-smaller": Update when the metric is smaller than previously stored.
            - "metric-larger": Update when the metric is larger than previously stored.
        """
        assert update_criteria in UpdateCriteria.options(), ValueError(
            f"Invalid criteria! Options are {UpdateCriteria.options()}"
        )
        self._criteria = update_criteria

        self._state_dict = None
        self._metric = np.inf if self._criteria == UpdateCriteria.metric_smaller else -np.inf

    @property
    def state_dict(self):
        """
        Returns the state dict of the best model
        """
        return self._state_dict

    @property
    def metric(self):
        """
        Returns the metric value of the best model
        """
        return self._metric

    def check_and_update(self, model, metric: Optional[Union[int, float]] = None) -> bool:
        """
        Check if the new model is better than the buffered model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be evaluated.
        metric : int or float, optional
            Performance metric of the model.

        Returns
        -------
        bool
            True if the new model replaces the buffered one, else False.
        """
        update_flag = (
            (self._criteria == UpdateCriteria.always)
            or (self._criteria == UpdateCriteria.metric_smaller and metric <= self.metric)
            or (self._criteria == UpdateCriteria.metric_larger and metric >= self.metric)
        )

        if update_flag:
            self._metric = metric
            model.to("cpu")
            model_cp = copy.deepcopy(model)

            self._state_dict = model_cp.state_dict()
            return True

        return False

    def save(self, model_dir: str):
        """
        Save the buffered model to the specified directory.

        Parameters
        ----------
        model_dir : str
            Directory path to save the model.

        Returns
        -------
        None
        """
        out_dict = dict()
        for attr, value in self.__dict__.items():
            if regex.match(f"^_[a-z]", attr):
                out_dict[attr] = value

        torch.save(out_dict, model_dir)
        return None

    def load(self, model_dir: str):
        """
        Load the model from the specified directory.

        Parameters
        ----------
        model_dir : str
            Directory path from where the model should be loaded.

        Returns
        -------
        CheckpointContainer
            Instance of the loaded model container.
        """
        model_dict = torch.load(model_dir)

        for attr, value in model_dict.items():
            if attr not in self.__dict__:
                logger.warning(f"Attribute {attr} is not natively defined in model buffer!")
            setattr(self, attr, value)

        return self
