"""
# Modified: February 15th, 2024
# ---------------------------------------
# Description: NER Entry Script.
"""

import sys
import logging
import os.path as osp
from datetime import datetime

from src.core.argparser import ArgumentParser
from src.core.args import Arguments, Config

from src.gpt.ner.pipeline import UniEntNERPipeline, MultiEntNERPipeline
from seqlbtoolkit.io import set_logging

logger = logging.getLogger(__name__)


def main(args):
    config = Config().from_args(args).check().log()
    if config.entities_to_exclude is not None:
        logger.warning(f"Entities to exclude: {config.entities_to_exclude}")

    if not config.prompt_format == "aeio":
        ner_pipeline = UniEntNERPipeline(
            gpt_resource=config.gpt_resource_path,
            data_path=config.test_path,
            result_dir=config.result_dir,
            conflict_data_dir=config.conflict_data_dir,
            conflict_resolution_dir=config.conflict_resolution_dir,
            entities_to_exclude=config.entities_to_exclude,
            n_test_samples=config.n_test_samples,
            cot=not config.disable_cot,
            prompt_format=config.prompt_format,
            apply_conflict_resolution_in_eval=config.apply_conflict_resolution_in_eval,
            disable_result_removal=config.disable_result_removal,
        )

        if "generation" in config.tasks:
            logger.info("Generating GPT responses...")

            ner_pipeline.generate_gpt_responses()

        if "evaluation" in config.tasks:
            logger.info("Evaluating GPT responses...")

            metrics_partial_match, metrics_full_match = ner_pipeline.load_and_eval(report_dir=config.report_dir)
            logger.info(f"Metrics with partial match: {metrics_partial_match}")
            logger.info(f"Metrics full match: {metrics_full_match}")

        if "save-for-bert" in config.tasks:
            logger.info("Saving GPT responses for BERT fine-tuning...")

            ner_pipeline.save_for_bert(bert_data_dir=config.bert_data_dir)

        if "conflict-detection" in config.tasks:
            logger.info("Detecting conflicts...")

            ner_pipeline.detect_and_save_conflict_results()

        if "conflict-resolution" in config.tasks:
            logger.info("Resolving conflicts...")

            ner_pipeline.load_and_resolve_conflicts()

    else:
        ner_pipeline = MultiEntNERPipeline(
            gpt_resource=config.gpt_resource_path,
            data_path=config.test_path,
            result_dir=config.result_dir,
            entities_to_exclude=config.entities_to_exclude,
            n_test_samples=config.n_test_samples,
            cot=True,
        )

        if "generation" in config.tasks:
            logger.info("Generating GPT responses...")

            ner_pipeline.generate_gpt_responses()

        if "evaluation" in config.tasks:
            logger.info("Evaluating GPT responses...")

            metrics_partial_match, metrics_full_match = ner_pipeline.load_and_eval(report_dir=config.report_dir)
            logger.info(f"Metrics with partial match: {metrics_partial_match}")
            logger.info(f"Metrics full match: {metrics_full_match}")

    return None


if __name__ == "__main__":
    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = osp.basename(__file__)
    if _current_file_name.endswith(".py"):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = ArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        (arguments,) = parser.parse_json_file(json_file=osp.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    if not getattr(arguments, "log_path", None):
        arguments.log_path = osp.join("./logs", f"{_current_file_name}", f"{_time}.log")

    set_logging(log_path=arguments.log_path)
    main(arguments)
