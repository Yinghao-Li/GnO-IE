"""
# Modified: February 15th, 2024
# ---------------------------------------
# Description: NER Entry Script.
"""

import sys
import logging
import os.path as osp
from datetime import datetime
from seqlbtoolkit.io import set_logging

from src.core.argparser import ArgumentParser
from src.llama.args import Arguments, Config
from src.llama.ner.pipeline import UniEntNERPipeline


logger = logging.getLogger(__name__)


def main(args):
    config = Config().from_args(args).check().log()
    if config.entities_to_exclude is not None:
        logger.warning(f"Entities to exclude: {config.entities_to_exclude}")

    ner_pipeline = UniEntNERPipeline(
        model_dir=config.model_dir,
        data_path=config.test_path,
        result_dir=config.result_dir,
        entities_to_exclude=config.entities_to_exclude,
        n_test_samples=config.n_test_samples,
        cot=True,
        prompt_format=config.prompt_format,
        tasks=config.tasks,
        model_dtype=config.model_dtype,
        disable_result_removal=config.disable_result_removal,
    )

    if "generation" in config.tasks:
        logger.info("Generating GPT responses...")

        ner_pipeline.generate_llm_responses()

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
