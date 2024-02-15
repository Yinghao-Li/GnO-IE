"""
# Modified: February 15th, 2024
# ---------------------------------------
# Description: Defines the NER pipeline.
"""

import os.path as osp
import json
import glob
import spacy
import logging
import natsort

from tqdm.auto import tqdm
from seqlbtoolkit.data import txt_to_token_span
from seqlbtoolkit.io import init_dir, save_json

from ..interaction import Interaction, ConflictResolutionInteraction
from src.core.metrics import get_ner_metrics
from src.core.utils import (
    locate_phrase_in_text,
    find_closest_column_name,
    extract_markdown_tables,
    markdown_table_to_dataframe,
    get_data_downsampling_ids,
    merge_overlapping_spans,
    overlaps,
)

logger = logging.getLogger(__name__)


class UniEntNERPipeline:
    def __init__(
        self,
        gpt_resource,
        data_path: str = None,
        result_dir: str = None,
        conflict_data_dir: str = None,
        conflict_resolution_dir: str = None,
        entities_to_exclude: list[str] = None,
        n_test_samples: int = None,
        cot: bool = False,
        prompt_format: str = "ino",
        apply_conflict_resolution_in_eval: bool = True,
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

        self.interaction = Interaction(
            gpt_resource_path=gpt_resource,
            entity_types=self.entity_types,
            entity2elaboration=meta_dict["entity2elaboration"],
            entity2prompt=meta_dict["entity2prompt"],
            entity2columns=self.entity2cols,
            cot=cot,
            prompt_format=prompt_format,
            disable_result_removal=disable_result_removal,
        )
        self.conflict_resolution_interaction = ConflictResolutionInteraction(
            gpt_resource_path=gpt_resource, entity_types=self.entity_types, cot=cot
        )
        self.text_tks_list = [[token.text for token in self.nlp(sentence)] for sentence in self.sentence_list]

        self.gt_spans = list()
        for sentence, text_tks, entity_spans in zip(self.sentence_list, self.text_tks_list, self.entity_list):
            self.gt_spans.append(txt_to_token_span(text_tks, sentence, entity_spans))

        self.result_dir = result_dir
        self.conflict_data_dir = conflict_data_dir
        self.conflict_resolution_dir = conflict_resolution_dir
        if self.result_dir is not None:
            init_dir(self.result_dir, clear_original_content=False)
        else:
            raise ValueError("Please specify the result directory!")

        self.apply_conflict_resolution_in_eval = apply_conflict_resolution_in_eval

    @property
    def curr_messagers(self):
        return self.interaction.messagers

    @property
    def conflict_resolution_messager(self):
        return self.conflict_resolution_interaction.messager

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

            self.save_messagers(result_path)

        logger.info("Done.")
        return self

    def parse_gpt_response(self, response: str, text_tks: list[str]) -> dict:
        """
        Parse the GPT response.
        """

        result_dict = dict()
        for entity_type in self.entity_types:
            result_dict[entity_type] = dict()

            try:
                markdown_table = extract_markdown_tables(response[entity_type])[-1]
            except IndexError:
                continue

            df = markdown_table_to_dataframe(markdown_table)

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

    def parse_gpt_conflict_resolution_response(self, response: str, text_tks: list[str]) -> dict:
        try:
            markdown_table = extract_markdown_tables(response)[-1]
        except IndexError:
            return list()

        df = markdown_table_to_dataframe(markdown_table)
        ent_column_name = find_closest_column_name(df, "Entity")
        ent_type_column_name = find_closest_column_name(df, "Entity Type")

        ent2type = dict()
        for ent, ent_type in zip(df[ent_column_name], df[ent_type_column_name]):
            if ent not in ent2type:
                ent2type[ent] = ent_type

        span2type = dict()
        candidate_ent_types = [ent.casefold() for ent in self.entity_types]
        for ent, ent_type in ent2type.items():
            if ent_type.casefold() not in candidate_ent_types:
                continue

            ent_lower = ent.lower()
            spans = locate_phrase_in_text(ent_lower, len(self.nlp(ent_lower)), [tk.lower() for tk in text_tks])

            for span in spans:
                span2type[span[:2]] = ent_type

        return span2type

    @staticmethod
    def convert_spans_format(entity_dict: dict):
        span_dict = dict()
        for k, v in entity_dict.items():
            for span in v:
                span_dict[span] = k
        return span_dict

    def save_messagers(self, path: str) -> None:
        """
        Save the messagers to a file.
        """
        self.interaction.save_messagers(path)
        return self

    def save_conflict_resolution_messager(self, path: str) -> None:
        self.conflict_resolution_interaction.save_messager(path)
        return self

    def load_messagers(self, path: str) -> None:
        """
        Load the messagers from a file.
        """
        self.interaction.load_messagers(path)
        return self

    def load_conflict_resolution_messager(self, path: str) -> None:
        self.conflict_resolution_interaction.load_messager(path)
        return self

    def integrity_check(self) -> None:
        """
        Check whether the result is correct.
        """
        for k, v in self.curr_messagers.items():
            if len(v) < 3:
                return False
        return True

    def load_and_eval(self, report_dir: str = None):
        """
        Load the messagers from a file and evaluate the results.
        """
        pred_spans = list()

        # error tracking
        valid_ids_list = list()
        file_not_found_ids_list = list()
        gpt_failure_ids_list = list()

        ids = range(len(self.text_tks_list)) if self.sample_ids is None else self.sample_ids

        conflict_resolution_applied = False
        logger.info("Parsing GPT results...")
        for idx, text_tks in tqdm(zip(ids, self.text_tks_list), total=len(ids)):
            # --- messager path integrity check ---
            output_path = osp.join(self.result_dir, f"messagers-{idx:05d}.json")

            if not osp.exists(output_path):
                logger.warning(f"File {output_path} does not exist. Skip.")
                file_not_found_ids_list.append(idx)
                continue

            # --- load the messagers ---
            self.load_messagers(output_path)

            if not self.integrity_check():
                logger.warning(f"File {output_path} is incomplete. Skip.")
                gpt_failure_ids_list.append(idx)
                continue

            # --- load the GPT response ---
            gpt_response = {ent_type: msg.content[-1]["content"] for ent_type, msg in self.curr_messagers.items()}

            # --- parse the GPT response ---
            ents = self.parse_gpt_response(response=gpt_response, text_tks=text_tks)

            # --- conflict resolution ---
            resolution_path = osp.join(self.conflict_resolution_dir, f"resolution-{idx:05d}.json")
            if self.apply_conflict_resolution_in_eval and osp.exists(resolution_path):
                conflict_resolution_applied = True
                logger.info(f"Applying conflict resolution for instance {idx}...")

                self.load_conflict_resolution_messager(resolution_path)
                resolution_resp = self.conflict_resolution_messager.content[-1]["content"]
                resolution_spans2type = self.parse_gpt_conflict_resolution_response(resolution_resp, text_tks)
                ents = self.apply_conflict_resolutions(ents, resolution_spans2type)

            valid_ids_list.append(idx)
            pred_spans.append(ents)

        if self.apply_conflict_resolution_in_eval and not conflict_resolution_applied:
            logger.warning("Conflict resolution is enabled but not applied in evaluation due to missing files!")

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

    @staticmethod
    def detect_entity_type_conflict(entity_dict: dict[str, dict[tuple[int, int], str]]):
        """
        Detect entity type conflicts.
        """
        span_dict = dict()
        for entity_type, entity_spans in entity_dict.items():
            for span in entity_spans:
                overlaps_flag = False
                for k in span_dict:
                    if overlaps(span, k):
                        new_span = (min(span[0], k[0]), max(span[1], k[1]))
                        original_entity_types = span_dict.pop(k)
                        span_dict[new_span] = original_entity_types + [entity_type]
                        overlaps_flag = True
                        break

                if not overlaps_flag:
                    span_dict[span] = [entity_type]

        span_list = [[k[0], k[1]] + v for k, v in span_dict.items() if len(v) > 1]

        return span_list

    def detect_and_save_conflict_results(self) -> None:
        assert self.conflict_data_dir is not None, "Please specify the confliction result directory!"

        ids = range(len(self.text_tks_list)) if self.sample_ids is None else self.sample_ids

        logger.info("Parsing GPT results and detecting conflicts...")

        for idx, text_tks in tqdm(zip(ids, self.text_tks_list), total=len(ids)):
            # check whether the output file already exists
            output_path = osp.join(self.result_dir, f"messagers-{idx:05d}.json")
            conflict_path = osp.join(self.conflict_data_dir, f"conflict-{idx:05d}.json")

            if osp.exists(conflict_path):
                logger.info(f"File {conflict_path} already exists. Skip.")
                continue

            if not osp.exists(output_path):
                logger.warning(f"File {output_path} does not exist. Skip.")
                continue

            self.load_messagers(output_path)

            if not self.integrity_check():
                logger.warning(f"File {output_path} is incomplete. Skip.")
                continue

            gpt_response = {ent_type: msg.content[-1]["content"] for ent_type, msg in self.curr_messagers.items()}

            ents = self.parse_gpt_response(response=gpt_response, text_tks=text_tks)

            conflict_dict = self.detect_entity_type_conflict(ents)
            if not conflict_dict:
                continue

            conflict_msg_dict = {
                "idx": idx,
                "text_tks": text_tks,
                "gpt_response": gpt_response,
                "conflict": conflict_dict,
            }
            save_json(conflict_msg_dict, conflict_path, collapse_level=2)

        logger.info("Done.")

        return None

    def load_and_resolve_conflicts(self):
        assert self.conflict_data_dir is not None, "Please specify the confliction result directory!"
        conflict_files = glob.glob(osp.join(self.conflict_data_dir, "conflict-*.json"))
        conflict_files = natsort.natsorted(conflict_files)
        conflict_ids = list()

        logger.info("Resolving NER conflicts...")
        for conflict_file in tqdm(conflict_files):
            conflict_resolution_file = osp.join(
                self.conflict_resolution_dir, osp.basename(conflict_file).replace("conflict", "resolution")
            )
            if osp.exists(conflict_resolution_file):
                logger.info(f"File {conflict_resolution_file} already exists. Skip.")
                continue

            with open(conflict_file, "r", encoding="utf-8") as f:
                conflict_dict = json.load(f)

            idx = conflict_dict["idx"]
            conflict_ids.append(idx)

            text_tks = conflict_dict["text_tks"]
            ner_gpt_response = conflict_dict["gpt_response"]

            self.conflict_resolution_interaction.interact(text_tks, ner_gpt_response)
            self.save_conflict_resolution_messager(conflict_resolution_file)

        logger.info("Done.")

        return self

    @staticmethod
    def apply_conflict_resolutions(detected_type2spans, resolution_spans2type):
        new_ents = dict()
        for ent_type, ent_spans in detected_type2spans.items():
            new_ents[ent_type] = dict()
            for span, ent in ent_spans.items():
                if span in resolution_spans2type and resolution_spans2type[span] != ent_type:
                    continue
                new_ents[ent_type][span] = ent
        return new_ents

    def save_for_bert(self, bert_data_dir: str):
        pred_spans = list()

        # error tracking
        valid_ids_list = list()
        file_not_found_ids_list = list()
        gpt_failure_ids_list = list()

        ids = range(len(self.text_tks_list)) if self.sample_ids is None else self.sample_ids

        conflict_resolution_applied = False
        logger.info("Parsing GPT results...")
        for idx, text_tks in tqdm(zip(ids, self.text_tks_list), total=len(ids)):
            # --- messager path integrity check ---
            output_path = osp.join(self.result_dir, f"messagers-{idx:05d}.json")

            if not osp.exists(output_path):
                logger.warning(f"File {output_path} does not exist. Skip.")
                file_not_found_ids_list.append(idx)
                continue

            # --- load the messagers ---
            self.load_messagers(output_path)

            if not self.integrity_check():
                logger.warning(f"File {output_path} is incomplete. Skip.")
                gpt_failure_ids_list.append(idx)
                continue

            # --- load the GPT response ---
            gpt_response = {ent_type: msg.content[-1]["content"] for ent_type, msg in self.curr_messagers.items()}

            # --- parse the GPT response ---
            ents = self.parse_gpt_response(response=gpt_response, text_tks=text_tks)

            # --- conflict resolution ---
            resolution_path = osp.join(self.conflict_resolution_dir, f"resolution-{idx:05d}.json")
            if self.apply_conflict_resolution_in_eval and osp.exists(resolution_path):
                conflict_resolution_applied = True
                logger.info(f"Applying conflict resolution for instance {idx}...")

                self.load_conflict_resolution_messager(resolution_path)
                resolution_resp = self.conflict_resolution_messager.content[-1]["content"]
                resolution_spans2type = self.parse_gpt_conflict_resolution_response(resolution_resp, text_tks)
                ents = self.apply_conflict_resolutions(ents, resolution_spans2type)

            valid_ids_list.append(idx)
            pred_spans.append(ents)

        if self.apply_conflict_resolution_in_eval and not conflict_resolution_applied:
            logger.warning("Conflict resolution is enabled but not applied in evaluation due to missing files!")

        logger.info("Done.")

        text_tks_list = [text_tks for idx, text_tks in zip(ids, self.text_tks_list) if idx in valid_ids_list]
        pred_spans = [self.convert_spans_format(ents) for ents in pred_spans]
        gt_spans = [gt_spans for idx, gt_spans in zip(ids, self.gt_spans) if idx in valid_ids_list]

        bert_training_data = dict()
        bert_test_data = dict()
        for idx, pred_span, gt_span, text_tks in zip(valid_ids_list, pred_spans, gt_spans, text_tks_list):
            bert_training_data[idx] = {
                "text": text_tks,
                "label": [[k[0], k[1], v] for k, v in pred_span.items()],
            }
            bert_test_data[idx] = {
                "text": text_tks,
                "label": [[k[0], k[1], v] for k, v in gt_span.items()],
            }

        save_json(
            bert_training_data,
            osp.join(bert_data_dir, "train.json"),
            collapse_level=3,
            disable_content_checking=True,
        )
        save_json(
            bert_test_data,
            osp.join(bert_data_dir, "test.json"),
            collapse_level=3,
            disable_content_checking=True,
        )

        meta = {
            "entity_types": self.entity_types,
        }
        save_json(meta, osp.join(bert_data_dir, "meta.json"))

        return None
