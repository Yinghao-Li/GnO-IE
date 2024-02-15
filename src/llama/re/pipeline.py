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
from src.core.metrics import get_re_metrics, get_re_metrics_relaxed
from src.core.utils import (
    locate_phrase_in_text,
    find_closest_column_name,
    extract_markdown_tables,
    markdown_table_to_dataframe,
    get_data_downsampling_ids,
)
from src.core.args import Config

logger = logging.getLogger(__name__)


class REPipeline:
    def __init__(
        self,
        config: Config,
        data_path: str = None,
        cot: bool = False,
    ) -> None:
        """
        Initialize the NER pipeline.

        Parameters
        ----------
        config: Config
            Configurations.
        data_path: str
            Path to the test data.
        cot: bool
            Whether to use chain-of-thought promting.
        """
        self.config = config
        # only used as a tokenizer
        self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])

        # Support for loading data from a directory.
        if data_path is not None:
            self.sentence_list, self.relation_list, meta_dict = self.load_data(data_path)
        else:
            raise ValueError("Please specify the data path!")
        self.relation_types = meta_dict["relations"]
        self.relation2cols = meta_dict["relation2columns"]

        if config.n_test_samples is not None:
            logger.info(f"Downsampling the data to {config.n_test_samples} samples...")
            self.sample_ids = self.load_or_create_test_sample_ids(data_path, n_samples=config.n_test_samples)
            self.downsample_data(self.sample_ids)
        else:
            self.sample_ids = None

        load_model = "generation" in config.tasks
        self.interaction = Interaction(
            model_dir=config.model_dir,
            relation_types=self.relation_types,
            relation2prompt=meta_dict["relation2prompt"],
            relation2cols=self.relation2cols,
            cot=cot,
            prompt_format=config.prompt_format,
            context_list=self.sentence_list,
            max_tokens=config.max_tokens,
            max_model_len=config.max_model_len,
            load_model=load_model,
            model_dtype=config.model_dtype,
        )
        self.text_tks_list = [[token.text for token in self.nlp(sentence)] for sentence in self.sentence_list]

        self.respan_relation_lbs()

        self.result_dir = config.result_dir
        if self.result_dir is not None:
            init_dir(self.result_dir, clear_original_content=False)
        else:
            raise ValueError("Please specify the result directory!")

    def respan_relation_lbs(self):
        """
        Respan the relation labels.
        """
        for text_tks, sentence, relations in zip(self.text_tks_list, self.sentence_list, self.relation_list):
            for relation_type in relations:
                relations[relation_type] = [
                    tuple(txt_to_token_span(text_tks, sentence, list(rls))) for rls in relations[relation_type]
                ]

        return self

    @property
    def messagers(self):
        return self.interaction.messagers

    @staticmethod
    def load_data(file_path: str):
        """
        Load data from a specified file and extract sentences, relations, and entities.

        Parameters
        ----------
        file_path: str
            The path to the file containing the data.

        Returns
        -------
        A tuple containing lists of sentences, relations, entities, and metadata.
        """

        # Load metadata from the associated meta.json file
        meta_path = osp.join(osp.dirname(file_path), "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        # Load data from the given file path
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        # Initialize lists for sentences, relations, and entities
        sentence_list = []
        relation_list = []

        # Process each instance in the data list
        for instance in data_list:
            # Append the sentence to the sentence list
            sentence_list.append(instance["sentence"])

            # Process relations
            relation_inst = {rl_type: list() for rl_type in meta_dict["relations"]}
            for rl in instance["relations"]:
                if rl["type"] in meta_dict["relations"]:
                    if "head" in rl and "tail" in rl:
                        relation_inst[rl["type"]].append((tuple(rl["head"]["pos"]), tuple(rl["tail"]["pos"])))
                    else:
                        if len(rl["entities"]) == 3 or len(rl["entities"]) == 4:
                            relation_inst[rl["type"]].append(
                                (tuple(rl["entities"][0]), tuple(rl["entities"][1]), tuple(rl["entities"][2]))
                            )

            relation_list.append(relation_inst)

        return sentence_list, relation_list, meta_dict

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
        self.relation_list = [self.relation_list[idx] for idx in sample_ids]
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
        for relation_type in self.relation_types:
            result_dict[relation_type] = list()

            try:
                markdown_table = extract_markdown_tables(response[relation_type])[-1]
                df = markdown_table_to_dataframe(markdown_table)
            except Exception as e:
                continue

            sbj_col = find_closest_column_name(df, self.relation2cols[relation_type][0])
            obj_col = find_closest_column_name(df, self.relation2cols[relation_type][1])
            check_col = find_closest_column_name(df, self.relation2cols[relation_type][2])

            if not (sbj_col and obj_col and check_col):
                continue

            for sbj_ent, obj_ent, check_txt in zip(df[sbj_col], df[obj_col], df[check_col]):
                if not (sbj_ent and obj_ent and check_txt):
                    continue
                if not "yes" in [token.text.lower() for token in self.nlp(check_txt)]:
                    continue

                lower_tks = [tk.lower() for tk in text_tks]
                sbj_spans = locate_phrase_in_text(sbj_ent.lower(), len(self.nlp(sbj_ent.lower())), lower_tks)
                obj_spans = locate_phrase_in_text(obj_ent.lower(), len(self.nlp(sbj_ent.lower())), lower_tks)

                if sbj_spans and obj_spans:
                    for sbj_ent_span in sbj_spans:
                        for obj_ent_span in obj_spans:
                            if (
                                sbj_ent_span[:2] != obj_ent_span[:2]
                                and sbj_ent_span[2].lower() != obj_ent_span[2].lower()
                            ):
                                result_dict[relation_type].append((sbj_ent_span[:2], obj_ent_span[:2]))

        return result_dict

    def parse_llm_response_polyie(self, response: str, text_tks: list[str]) -> dict:
        result_dict = dict()
        for relation_type in self.relation_types:
            result_dict[relation_type] = list()

            try:
                markdown_table = extract_markdown_tables(response[relation_type])[-1]
            except IndexError:
                continue

            try:
                df = markdown_table_to_dataframe(markdown_table)
            except Exception as e:
                logger.error(f"Error occurred when parsing the markdown table: {e}")
                continue

            exp_column_names = self.relation2cols[relation_type]
            col_names = list()
            for exp_column_name in exp_column_names:
                col_names.append(find_closest_column_name(df, exp_column_name))

            # skip if any of the column names is not found
            if not all([col_name for col_name in col_names]):
                continue

            for cn, pn, pn_abbr, pv, condition in zip(*[df[col_name] for col_name in col_names]):
                if not cn or not pn or not pv:
                    continue

                lower_tks = [tk.lower() for tk in text_tks]

                cn_spans = locate_phrase_in_text(cn.lower(), len(self.nlp(cn.lower())), lower_tks)
                pn_spans = locate_phrase_in_text(pn.lower(), len(self.nlp(pn.lower())), lower_tks)
                pv_spans = locate_phrase_in_text(pv.lower(), len(self.nlp(pv.lower())), lower_tks)

                if not pn_spans:
                    pn_spans = locate_phrase_in_text(pn_abbr, len(self.nlp(pn_abbr)), text_tks)
                if not pn_spans:
                    pn_spans = locate_phrase_in_text(pn_abbr.lower(), len(self.nlp(pn_abbr.lower())), lower_tks)

                if not (cn_spans and pn_spans and pv_spans):
                    continue

                result_dict[relation_type].append(
                    (
                        [cn_span[:2] for cn_span in cn_spans],
                        [pn_span[:2] for pn_span in pn_spans],
                        [pv_span[:2] for pv_span in pv_spans],
                    )
                )

        return result_dict

    def save_messagers(self) -> None:
        """
        Save the messagers to a file.
        """
        messager_ids = range(len(self.text_tks_list)) if self.sample_ids is None else self.sample_ids
        self.interaction.save_messagers(self.result_dir, messager_ids)
        return self

    def load_messagers(self, messager_dir: str, ids: list[int]) -> None:
        """
        Load the messagers from a file.
        """
        self.interaction.load_messagers(messagers_dir=messager_dir, ids=ids)
        return self

    def load_and_eval(self, report_dir: str = None):
        """
        Load the messagers from a file and evaluate the results.
        """
        pred_relations = list()

        ids = range(len(self.text_tks_list)) if self.sample_ids is None else self.sample_ids

        logger.info("Parsing GPT results...")
        # --- load the messagers ---
        self.load_messagers(self.result_dir, ids)

        for idx, text_tks in enumerate(tqdm(self.text_tks_list)):
            # check whether the output file already exists
            responses = {rt: self.messagers[rt][idx].content[-1]["content"] for rt in self.relation_types}

            if self.config.dataset == "PolyIE":
                relations = self.parse_llm_response_polyie(response=responses, text_tks=text_tks)
            else:
                relations = self.parse_llm_response(response=responses, text_tks=text_tks)

            pred_relations.append(relations)

        logger.info("Done.")

        # evaluate the results

        if self.config.dataset == "PolyIE":
            result_dict_partial_match, report_partial_match = get_re_metrics_relaxed(
                pred_list=pred_relations,
                gt_list=self.relation_list,
                relation_types=self.relation_types,
                tks_list=self.text_tks_list,
                ids_list=ids,
                allow_partial_match=True,
            )
            result_dict_full_match, report_full_match = get_re_metrics_relaxed(
                pred_list=pred_relations,
                gt_list=self.relation_list,
                relation_types=self.relation_types,
                tks_list=self.text_tks_list,
                ids_list=ids,
                allow_partial_match=False,
            )

        else:
            result_dict_partial_match, report_partial_match = get_re_metrics(
                pred_list=pred_relations,
                gt_list=self.relation_list,
                relation_types=self.relation_types,
                tks_list=self.text_tks_list,
                ids_list=ids,
                allow_partial_match=True,
            )
            result_dict_full_match, report_full_match = get_re_metrics(
                pred_list=pred_relations,
                gt_list=self.relation_list,
                relation_types=self.relation_types,
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
                    collapse_level=5,
                )
                save_json(
                    report_full_match[item], osp.join(report_dir, f"report_full_match_{item}.json"), collapse_level=5
                )

            save_json(
                {"partial_match": result_dict_partial_match, "full_match": result_dict_full_match},
                osp.join(report_dir, "metrics.json"),
            )

        return result_dict_partial_match, result_dict_full_match
