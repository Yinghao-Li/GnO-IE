"""
# Modified: February 15th, 2024
# ---------------------------------------
# Description: Defines the NER pipeline.
"""

import os.path as osp
import json
import spacy
import logging

from tqdm.auto import tqdm
from seqlbtoolkit.data import txt_to_token_span
from seqlbtoolkit.io import init_dir, save_json

from .interaction import Interactions
from src.core.metrics import get_ner_metrics
from src.core.utils import (
    locate_phrase_in_text,
    get_data_downsampling_ids,
    merge_overlapping_spans,
)

logger = logging.getLogger(__name__)


class UniNERPipeline:
    def __init__(
        self,
        model_dir,
        data_path: str = None,
        result_dir: str = None,
        entities_to_exclude: list[str] = None,
        n_test_samples: int = None,
        max_tokens: int = 256,
        tasks: list[str] = None,
    ) -> None:
        """
        Initialize the NER pipeline.
        """
        # only used as a tokenizer
        self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])

        # Support for loading data from a directory.
        if data_path is not None:
            self.sentence_list, self.entity_list, meta_dict = self.load_data(data_path)
        else:
            raise ValueError("Please specify the data path!")
        self.entity_types = [ent_type for ent_type in meta_dict["entities"] if ent_type not in entities_to_exclude]
        self.entity2cols = meta_dict["entity2columns"]
        self.entity_list = [
            {k: v for k, v in entity_dict.items() if v in self.entity_types} for entity_dict in self.entity_list
        ]

        if n_test_samples is not None:
            logger.info(f"Downsampling the data to {n_test_samples} samples...")
            self.sample_ids = self.load_or_create_test_sample_ids(data_path, n_samples=n_test_samples)
            self.downsample_data(self.sample_ids)
        else:
            self.sample_ids = None

        if "generation" in tasks:
            self.interaction = Interactions(
                model_dir=model_dir,
                entity_types=self.entity_types,
                entity2uniner=meta_dict["entity2uniner"],
                context_list=self.sentence_list,
                max_tokens=max_tokens,
            )
        else:
            self.interaction = None

        self.text_tks_list = [[token.text for token in self.nlp(sentence)] for sentence in self.sentence_list]

        self.gt_spans = list()
        for sentence, text_tks, entity_spans in zip(self.sentence_list, self.text_tks_list, self.entity_list):
            self.gt_spans.append(txt_to_token_span(text_tks, sentence, entity_spans))

        self.result_dir = result_dir
        if self.result_dir is not None:
            init_dir(self.result_dir, clear_original_content=False)
        else:
            raise ValueError("Please specify the result directory!")

        self.responses = None
        self.pred_results = None

    @staticmethod
    def load_data(file_path: str) -> None:
        """
        Load data stored in the current data format.
        """
        meta_path = osp.join(osp.dirname(file_path), "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        with open(file_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)

        sentence_list = list()
        entity_list = list()

        for item in data_dict:
            sentence_list.append(item["sentence"])
            entity_list.append(
                {
                    tuple(entity["pos"]): entity["type"]
                    for entity in item["entities"]
                    if entity["type"] in meta_dict["entities"]
                }
            )

        return sentence_list, entity_list, meta_dict

    def load_or_create_test_sample_ids(self, data_path: str, n_samples: int = None):
        """
        Load or create test sample ids.

        Parameters
        ----------
        data_path: str
            Path to the test data.
        n_samples: int
            Number of samples to use.

        Returns
        -------
        sample_ids: list[int]
            List of sample ids.
        """
        if n_samples is None:
            return None

        sample_data_path = osp.join(osp.dirname(data_path), f"test-ids-{n_samples}.json")
        if osp.exists(sample_data_path):
            with open(sample_data_path, "r", encoding="utf-8") as f:
                sample_ids = json.load(f)
        else:
            sample_ids = get_data_downsampling_ids(n_data=len(self.sentence_list), n_samples=n_samples)
            save_json(sample_ids, sample_data_path)

        return sample_ids

    def downsample_data(self, sample_ids: list[int]):
        """
        Downsample the data.

        Parameters
        ----------
        sample_ids: list[int]
            List of sample ids.

        Returns
        -------
        self
        """
        self.sentence_list = [self.sentence_list[idx] for idx in sample_ids]
        self.entity_list = [self.entity_list[idx] for idx in sample_ids]
        return self

    def generate_responses(self) -> None:
        """
        Label the instances.

        This function is designed to be called outside the class.
        """

        logger.info("Labeling instances...")
        self.responses = self.interaction.interact()

        self.pred_results = list()
        for idx in range(len(self.sentence_list)):
            self.pred_results.append({ent_type: self.responses[ent_type][idx] for ent_type in self.entity_types})

        logger.info("Saving responses...")
        self.save_results(osp.join(self.result_dir, "responses.json"))

        logger.info("Done.")
        return self

    def save_results(self, output_path: str) -> None:
        """
        save
        """
        ids = range(len(self.sentence_list)) if self.sample_ids is None else self.sample_ids

        result_dict = dict()
        for idx, (inst_idx, context) in enumerate(zip(ids, self.sentence_list)):
            result_dict[inst_idx] = {
                "text": context,
                "responses": {ent_type: resp[idx] for ent_type, resp in self.responses.items()},
            }
        save_json(result_dict, output_path, collapse_level=4)
        return None

    def load_results(self, input_path: str) -> None:
        """
        load
        """
        if self.responses is not None:
            logger.warning("The responses have already been loaded!")
            return None

        with open(input_path, "r", encoding="utf-8") as f:
            result_dict = json.load(f)

        responses = {ent_type: list() for ent_type in self.entity_types}
        for k, v in result_dict.items():
            for ent_type, resp in v["responses"].items():
                responses[ent_type].append(resp)

        self.responses = responses
        self.pred_results = [v["responses"] for v in result_dict.values()]

        return None

    def parse_response(self, preds: str, text_tks: list[str]) -> dict:
        """
        Parse the GPT response.
        """

        result_dict = dict()
        for entity_type in self.entity_types:
            result_dict[entity_type] = dict()

            ent_spans = list()
            for entity in preds[entity_type]:
                entity_lower = entity.lower()
                ent_spans += locate_phrase_in_text(
                    entity_lower, len(self.nlp(entity_lower)), [tk.lower() for tk in text_tks]
                )

            # handle the case where the entities overlap
            ent_spans = merge_overlapping_spans(ent_spans)

            for ent_span in ent_spans:
                result_dict[entity_type][ent_span[:2]] = " ".join(text_tks[ent_span[0] : ent_span[1]])

        return result_dict

    @staticmethod
    def convert_spans_format(entity_dict: dict):
        span_dict = dict()
        for k, v in entity_dict.items():
            for span in v:
                span_dict[span] = k
        return span_dict

    def load_and_eval(self, report_dir: str = None):
        """
        Load the messagers from a file and evaluate the results.
        """

        # --- load the model responses ---
        self.load_results(osp.join(self.result_dir, "responses.json"))

        pred_spans = list()

        # error tracking
        valid_ids_list = list()

        ids = range(len(self.text_tks_list)) if self.sample_ids is None else self.sample_ids

        logger.info("Parsing GPT results...")
        for idx, text_tks, pred_results in tqdm(zip(ids, self.text_tks_list, self.pred_results), total=len(ids)):
            # --- parse the predicted results ---
            ents = self.parse_response(preds=pred_results, text_tks=text_tks)

            valid_ids_list.append(idx)
            pred_spans.append(ents)

        logger.info("Done.")

        gt_spans = [gt_spans for idx, gt_spans in zip(ids, self.gt_spans) if idx in valid_ids_list]
        text_tks_list = [text_tks for idx, text_tks in zip(ids, self.text_tks_list) if idx in valid_ids_list]
        pred_spans = [self.convert_spans_format(ents) for ents in pred_spans]

        # evaluate the results
        result_dict_partial_match, report_partial_match = get_ner_metrics(
            pred_list=pred_spans,
            gt_list=gt_spans,
            entity_types=self.entity_types,
            tks_list=text_tks_list,
            ids_list=valid_ids_list,
            allow_partial_match=True,
        )
        result_dict_full_match, report_full_match = get_ner_metrics(
            pred_list=pred_spans,
            gt_list=gt_spans,
            entity_types=self.entity_types,
            tks_list=text_tks_list,
            ids_list=valid_ids_list,
            allow_partial_match=False,
        )

        if report_dir:
            logger.info(f"Saving reports to {report_dir}...")
            for item in ("fp", "fn", "tp"):
                save_json(
                    report_partial_match[item],
                    osp.join(report_dir, f"report_partial_match_{item}.json"),
                )
                save_json(report_full_match[item], osp.join(report_dir, f"report_full_match_{item}.json"))

            save_json(
                {"partial_match": result_dict_partial_match, "full_match": result_dict_full_match},
                osp.join(report_dir, "metrics.json"),
            )

        return result_dict_partial_match, result_dict_full_match
