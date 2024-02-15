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

from .interaction import Interaction
from src.core.metrics import get_ner_metrics
from src.core.utils import (
    locate_phrase_in_text,
    find_closest_column_name,
    extract_markdown_tables,
    markdown_table_to_dataframe,
    get_data_downsampling_ids,
    merge_overlapping_spans,
)

logger = logging.getLogger(__name__)


class UniEntNERPipeline:
    def __init__(
        self,
        model_dir,
        data_path: str = None,
        result_dir: str = None,
        entities_to_exclude: list[str] = None,
        n_test_samples: int = None,
        cot: bool = False,
        prompt_format: str = "ino",
        tasks: list[str] = None,
        model_dtype: str = "auto",
        disable_result_removal: bool = False,
    ) -> None:
        """
        Initialize the NER pipeline.

        Parameters
        ----------
        gpt_resource: str
            Path to the GPT resource.
        sentence_list: list[str]
            List of sentences.
        entity_list: list[dict[tuple[int, int], str]]
            List of entities.
        entity_types: list[str]
            List of entity types.
        cot: bool
            Whether to use chain-of-thought promting.
        result_dir: str
            Path to the output directory.
        """
        assert tasks, "Please specify the tasks!"

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

        load_model = "generation" in tasks
        self.interaction = Interaction(
            model_dir=model_dir,
            entity_types=self.entity_types,
            entity2elaboration=meta_dict["entity2elaboration"],
            entity2prompt=meta_dict["entity2prompt"],
            entity2columns=self.entity2cols,
            cot=cot,
            prompt_format=prompt_format,
            context_list=self.sentence_list,
            load_model=load_model,
            model_dtype=model_dtype,
            disable_result_removal=disable_result_removal,
        )
        self.text_tks_list = [[token.text for token in self.nlp(sentence)] for sentence in self.sentence_list]

        self.gt_spans = list()
        for sentence, text_tks, entity_spans in zip(self.sentence_list, self.text_tks_list, self.entity_list):
            self.gt_spans.append(txt_to_token_span(text_tks, sentence, entity_spans))

        self.result_dir = result_dir
        if self.result_dir is not None:
            init_dir(self.result_dir, clear_original_content=False)
        else:
            raise ValueError("Please specify the result directory!")

    @property
    def messagers(self):
        return self.interaction.messagers

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

    def generate_llm_responses(self) -> None:
        """
        Label the instances.

        This function is designed to be called outside the class.
        """

        logger.info("Labeling instances...")

        self.interaction.interact()
        self.save_messagers()

        logger.info("Done.")

        return self

    def parse_llm_response(self, response: str, text_tks: list[str]) -> dict:
        """
        Parse the GPT response.
        """

        result_dict = dict()
        for entity_type in self.entity_types:
            result_dict[entity_type] = dict()

            try:
                markdown_table = extract_markdown_tables(response[entity_type])[-1]
                df = markdown_table_to_dataframe(markdown_table)
            except Exception as e:
                continue

            if len(self.entity2cols[entity_type]) > 1:
                raise NotImplementedError("Multiple columns for a single entity type is not supported yet!")

            column_name = find_closest_column_name(df, self.entity2cols[entity_type][0])
            if not column_name:
                continue

            ent_spans = list()
            for entity in df[column_name]:
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

    def save_messagers(self) -> None:
        """
        Save the messagers to a file.
        """
        messager_ids = range(len(self.text_tks_list)) if self.sample_ids is None else self.sample_ids
        self.interaction.save_messagers(self.result_dir, messager_ids)
        return self

    def load_messagers(self, messagers_dir: str, ids: list[int]) -> None:
        """
        Load the messagers from a file.
        """
        self.interaction.load_messagers(messagers_dir, ids)
        return self

    def load_and_eval(self, report_dir: str = None):
        """
        Load the messagers from a file and evaluate the results.
        """

        # error tracking
        ids = range(len(self.text_tks_list)) if self.sample_ids is None else self.sample_ids

        logger.info("Parsing GPT results...")
        # --- load the messagers ---
        self.load_messagers(self.result_dir, ids)

        pred_spans = list()
        for idx, text_tks in enumerate(tqdm(self.text_tks_list)):
            # --- load the GPT response ---
            responses = {
                ent_type: self.messagers[ent_type][idx].content[-1]["content"] for ent_type in self.entity_types
            }

            # --- parse the GPT response ---
            ents = self.parse_llm_response(response=responses, text_tks=text_tks)
            pred_spans.append(ents)

        logger.info("Done.")

        pred_spans = [self.convert_spans_format(ents) for ents in pred_spans]

        # evaluate the results
        result_dict_partial_match, report_partial_match = get_ner_metrics(
            pred_list=pred_spans,
            gt_list=self.gt_spans,
            entity_types=self.entity_types,
            tks_list=self.text_tks_list,
            ids_list=ids,
            allow_partial_match=True,
        )
        result_dict_full_match, report_full_match = get_ner_metrics(
            pred_list=pred_spans,
            gt_list=self.gt_spans,
            entity_types=self.entity_types,
            tks_list=self.text_tks_list,
            ids_list=ids,
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
