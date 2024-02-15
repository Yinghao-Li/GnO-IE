import os
import json
import copy
import regex
import logging
import itertools
import operator
import numpy as np
from transformers import AutoTokenizer

import torch
from seqlbtoolkit.data import span_to_label, span_list_to_dict
from seqlbtoolkit.text import split_overlength_bert_input_sequence, substitute_unknown_tokens
from seqlbtoolkit.training.dataset import pack_instances

from .args import Config

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        text: list[list[str]] = None,
        lbs: list[list[str]] = None,
        ids: list[int] = None,
    ):
        super().__init__()
        self._text = text
        self._lbs = lbs
        self._ids = ids
        # Whether the text and lbs sequences are separated according to maximum BERT input lengths
        self._is_separated = False
        self.data_instances = None

        self._bert_tk_ids = None
        self._bert_attn_masks = None
        self._bert_lbs = None
        self._bert_tk_masks = None

    @property
    def n_insts(self):
        return len(self._text)

    @property
    def text(self):
        return self._text if self._text else list()

    @property
    def lbs(self):
        return self._lbs if self._lbs else list()

    @property
    def ids(self):
        return self._ids if self._ids else list()

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __add__(self, other):
        return Dataset(
            text=copy.deepcopy(self.text + other.text),
            lbs=copy.deepcopy(self.lbs + other.lbs),
        )

    def __iadd__(self, other):
        self.text = copy.deepcopy(self.text + other.text)
        self.lbs = copy.deepcopy(self.lbs + other.lbs)
        return self

    def prepare(self, config: Config, partition: str):
        """
        Load data from disk

        Parameters
        ----------
        config: configurations
        partition: dataset partition; in [train, valid, test]

        Returns
        -------
        self (MultiSrcNERDataset)
        """
        assert partition in ["train", "valid", "test"], ValueError(
            f"Argument `partition` should be one of 'train', 'valid' or 'test'!"
        )

        file_path = os.path.normpath(os.path.join(config.data_dir, f"{partition}.json"))

        logger.info(f"Loading data file: {file_path}")
        sentence_list, label_list, ids_list = load_data_from_json(file_path)

        self._text = sentence_list
        self._lbs = label_list
        self._ids = ids_list

        if config.separate_overlength_sequences:
            self.separate_sequence(config.bert_model_name_or_path, config.max_seq_length)
        self.substitute_unknown_tokens(config.bert_model_name_or_path)

        logger.info("Encoding sequences...")
        self.encode(config.bert_model_name_or_path, {lb: idx for idx, lb in enumerate(config.bio_label_types)})

        logger.info(f"Data loaded.")

        if config.debug:
            self.prepare_debug()

        self.data_instances = pack_instances(
            bert_tk_ids=self._bert_tk_ids,
            bert_attn_masks=self._bert_attn_masks,
            bert_lbs=self._bert_lbs,
        )
        return self

    def substitute_unknown_tokens(self, tokenizer_or_name):
        """
        Substitute the tokens in the sequences that cannot be recognized by the tokenizer
        This will not change sequence lengths

        Parameters
        ----------
        tokenizer_or_name: bert tokenizer

        Returns
        -------
        self
        """

        tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer_or_name, add_prefix_space=True)
            if isinstance(tokenizer_or_name, str)
            else tokenizer_or_name
        )

        self._text = [substitute_unknown_tokens(tk_seq, tokenizer) for tk_seq in self._text]
        return self

    def separate_sequence(self, tokenizer_or_name, max_seq_length):
        """
        Separate the overlength sequences and separate the labels accordingly

        Parameters
        ----------
        tokenizer_or_name: bert tokenizer
        max_seq_length: maximum bert sequence length

        Returns
        -------
        self
        """
        if self._is_separated:
            logger.warning("The sequences are already separated!")
            return self

        tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer_or_name, add_prefix_space=True)
            if isinstance(tokenizer_or_name, str)
            else tokenizer_or_name
        )

        if (
            np.array(
                [
                    len(tk_ids)
                    for tk_ids in tokenizer(self._text, add_special_tokens=True, is_split_into_words=True).input_ids
                ]
            )
            <= max_seq_length
        ).all():
            self._is_separated = True
            return self

        new_text_list = list()
        new_lbs_list = list()

        for text_inst, lbs_inst in zip(self._text, self._lbs):
            split_tk_seqs = split_overlength_bert_input_sequence(text_inst, tokenizer, max_seq_length)
            split_sq_lens = [len(tk_seq) for tk_seq in split_tk_seqs]

            seq_ends = list(itertools.accumulate(split_sq_lens, operator.add))
            seq_starts = [0] + seq_ends[:-1]

            split_lb_seqs = [lbs_inst[s:e] for s, e in zip(seq_starts, seq_ends)]

            new_text_list += split_tk_seqs
            new_lbs_list += split_lb_seqs

        self._text = new_text_list
        self._lbs = new_lbs_list

        self._is_separated = True

        return self

    def encode(self, tokenizer_name, lb2idx):
        """
        Build BERT token masks as model input

        Parameters
        ----------
        tokenizer_name: the name of the assigned Huggingface tokenizer
        lb2idx: a dictionary that maps the str labels to indices

        Returns
        -------
        self
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
        tokenized_text = tokenizer(self._text, add_special_tokens=True, is_split_into_words=True)

        # exclude over-length instances
        encoded_token_lens = np.array([len(tks) for tks in tokenized_text.input_ids])
        assert (encoded_token_lens <= tokenizer.max_model_input_sizes.get(tokenizer_name, 512)).all(), ValueError(
            "One or more sequences are longer than the maximum model input size. "
            "Consider using `self.separate_sequence` to break them into smaller pieces."
        )

        self._bert_tk_ids = tokenized_text.input_ids
        self._bert_attn_masks = tokenized_text.attention_mask

        bert_lbs_list = list()
        bert_tk_masks = list()
        for idx, (tks, bert_tk_idx_list, lbs) in enumerate(zip(self._text, self._bert_tk_ids, self._lbs)):
            word_ids = tokenized_text.word_ids(idx)

            word_ids_shifted_left = np.asarray([-100] + word_ids[:-1])
            word_ids = np.asarray(word_ids)

            is_first_wordpiece = (word_ids_shifted_left != word_ids) & (word_ids != None)
            word_ids[~is_first_wordpiece] = -100  # could be anything less than 0

            # this should not happen
            if np.setdiff1d(np.arange(len(tks)), word_ids).size > 0:
                raise ValueError(
                    "Failed to map all tokens to BERT tokens! "
                    "Consider running `substitute_unknown_tokens` before calling this function"
                )

            bert_lbs = torch.full((len(bert_tk_idx_list),), -100)
            bert_lbs[is_first_wordpiece] = torch.tensor([lb2idx[lb] for lb in lbs])
            bert_lbs_list.append(bert_lbs)

            masks = np.zeros(len(bert_tk_idx_list), dtype=bool)
            masks[is_first_wordpiece] = True
            bert_tk_masks.append(masks)

        self._bert_lbs = bert_lbs_list
        self._bert_tk_masks = bert_tk_masks

        return self

    def prepare_debug(self):
        for attr in self.__dict__.keys():
            if regex.match(f"^_[a-z]", attr):
                try:
                    setattr(self, attr, getattr(self, attr)[:100])
                except TypeError:
                    pass

        return self

    def downsample_training_set(self, ids: list[int]):
        for attr in self.__dict__.keys():
            if regex.match(f"^_[a-z]", attr):
                try:
                    values = getattr(self, attr)
                    sampled_values = [values[idx] for idx in ids]
                    setattr(self, attr, sampled_values)
                except TypeError:
                    pass

        return self

    def save(self, file_path: str):
        """
        Save the entire dataset for future usage

        Parameters
        ----------
        file_path: path to the saved file

        Returns
        -------
        self
        """
        attr_dict = dict()
        for attr, value in self.__dict__.items():
            if regex.match(f"^_[a-z]", attr):
                attr_dict[attr] = value

        os.makedirs(os.path.dirname(os.path.normpath(file_path)), exist_ok=True)
        torch.save(attr_dict, file_path)

        return self

    def load(self, file_path: str):
        """
        Load the entire dataset from disk

        Parameters
        ----------
        file_path: path to the saved file

        Returns
        -------
        self
        """
        attr_dict = torch.load(file_path)

        for attr, value in attr_dict.items():
            if attr not in self.__dict__:
                logger.warning(f"Attribute {attr} is not natively defined in dataset!")

            setattr(self, attr, value)

        return self


def load_data_from_json(file_dir: str):
    """
    Load data stored in the current data format.

    Parameters
    ----------
    file_dir: file directory

    """
    with open(file_dir, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    tk_seqs = list()
    lbs_list = list()
    ids_list = list()

    for idx, data in data_dict.items():
        # get tokens
        sent_tks = data["text"]
        tk_seqs.append(sent_tks)

        # get true labels
        lbs = span_to_label(span_list_to_dict(data["label"]), sent_tks)
        lbs_list.append(lbs)

        ids_list.append(int(idx))

    return tk_seqs, lbs_list, ids_list
