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

from ..interaction import AEiOInteraction
from src.core.metrics import get_ner_metrics
from src.core.utils import (
    locate_phrase_in_text,
    find_closest_column_name,
    extract_markdown_tables,
    markdown_table_to_dataframe,
    get_data_downsampling_ids,
)

logger = logging.getLogger(__name__)


class MultiEntNERPipeline:
    def __init__(
        self,
        gpt_resource,
        data_path: str = None,
        result_dir: str = None,
        entities_to_exclude: list[str] = None,
        n_test_samples: int = None,
        cot: bool = False,
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
        # only used as a tokenizer
        self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])

        # Support for loading data from a directory.
        if data_path is not None:
            self.sentence_list, self.entity_list, meta_dict = self.load_data(data_path)
        else:
            raise ValueError("Please specify the data path!")
        self.entity_types = [ent_type for ent_type in meta_dict["entities"] if ent_type not in entities_to_exclude]
        self.entity_list = [
            {k: v for k, v in entity_dict.items() if v in self.entity_types} for entity_dict in self.entity_list
        ]
        self.entity2elaboration = meta_dict["entity2elaboration"]

        if n_test_samples is not None:
            logger.info(f"Downsampling the data to {n_test_samples} samples...")
            self.sample_ids = self.load_or_create_test_sample_ids(data_path, n_samples=n_test_samples)
            self.downsample_data(self.sample_ids)
        else:
            self.sample_ids = None

        self.interaction = AEiOInteraction(
            gpt_resource,
            entity_types=self.entity_types,
            entity2elaboration=self.entity2elaboration,
            cot=cot,
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
    def curr_messager(self):
        return self.interaction.messager

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

    def generate_gpt_responses(self) -> None:
        """
        Label the instances.

        This function is designed to be called outside the class.
        """
        ids = range(len(self.text_tks_list)) if self.sample_ids is None else self.sample_ids

        logger.info("Labeling instances...")
        for idx, text_tks in tqdm(zip(ids, self.text_tks_list), total=len(ids)):
            # check whether the output file already exists
            result_path = osp.join(self.result_dir, f"messagers-{idx:05d}.json")

            if osp.exists(result_path):
                logger.info(f"File {result_path} already exists.")
                continue

            try:
                self.interaction.interact(" ".join(text_tks))
            except Exception as e:
                logger.error(f"Error occurred when locating entities in text: {e}")
                continue

            self.save_messager(result_path)

        logger.info("Done.")
        return self

    def parse_gpt_response(self, response: str, text_tks: list[str]) -> dict:
        try:
            markdown_table = extract_markdown_tables(response)[-1]
        except IndexError:
            return dict()

        df = markdown_table_to_dataframe(markdown_table)
        ent_col = find_closest_column_name(df, "Entity")
        ent_type_col = find_closest_column_name(df, "Entity Type")

        candidate_ent_types = [self.entity2elaboration[ent] for ent in self.entity_types]
        elaboration2ent = {v: k for k, v in self.entity2elaboration.items()}
        ent_dict = dict()

        for ent, et in zip(df[ent_col], df[ent_type_col]):
            # filter out entities whose types are not in the candidate list
            if not isinstance(et, str) or et not in candidate_ent_types:
                continue

            ent_dict[elaboration2ent[et]] = dict()

            ent_lower = ent.lower()
            spans = locate_phrase_in_text(ent_lower, len(self.nlp(ent_lower)), [tk.lower() for tk in text_tks])

            for span in spans:
                ent_dict[elaboration2ent[et]][span[:2]] = " ".join(text_tks[span[0] : span[1]])

        return ent_dict

    @staticmethod
    def convert_spans_format(entity_dict: dict):
        span_dict = dict()
        for k, v in entity_dict.items():
            for span in v:
                span_dict[span] = k
        return span_dict

    def save_messager(self, path: str) -> None:
        """
        Save the messagers to a file.
        """
        self.interaction.save_messager(path)
        return self

    def load_messagers(self, path: str) -> None:
        """
        Load the messagers from a file.
        """
        self.interaction.load_messager(path)
        return self

    def integrity_check(self) -> None:
        """
        Check whether the result is correct.
        """
        return True if len(self.curr_messager) >= 3 else False

    def load_and_eval(self, report_dir: str = None):
        """
        Load the messagers from a file and evaluate the results.
        """
        pred_spans = list()

        # error tracking
        file_not_found_ids_list = list()
        gpt_failure_ids_list = list()

        ids = range(len(self.text_tks_list)) if self.sample_ids is None else self.sample_ids

        valid_ids_list = list()
        logger.info("Parsing GPT results...")
        for idx, text_tks in tqdm(zip(ids, self.text_tks_list), total=len(ids)):
            # check whether the output file already exists
            output_path = osp.join(self.result_dir, f"messagers-{idx:05d}.json")

            if not osp.exists(output_path):
                logger.warning(f"File {output_path} does not exist. Skip.")
                file_not_found_ids_list.append(idx)
                continue

            self.load_messagers(output_path)

            if not self.integrity_check():
                logger.warning(f"File {output_path} is incomplete. Skip.")
                gpt_failure_ids_list.append(idx)
                continue

            gpt_response = self.curr_messager.content[-1]["content"]

            ents = self.parse_gpt_response(response=gpt_response, text_tks=text_tks)
            pred_spans.append(ents)

            valid_ids_list.append(idx)

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
            save_json(
                {
                    "file_not_found_ids_list": file_not_found_ids_list,
                    "gpt_failure_ids_list": gpt_failure_ids_list,
                },
                osp.join(report_dir, "error_ids.json"),
            )
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
