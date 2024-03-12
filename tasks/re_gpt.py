"""
# Modified: February 15th, 2024
# ---------------------------------------
# Description: RE Entry Script.
"""

import sys
import json
import logging
import os.path as osp
from datetime import datetime

from src.core.argparser import ArgumentParser
from src.core.args import Arguments, Config
from src.gpt.re.pipeline import REPipeline
from seqlbtoolkit.io import set_logging, prettify_json

logger = logging.getLogger(__name__)


def main(args):
    config = Config().from_args(args).check().log()
    if config.entities_to_exclude is not None:
        logger.warning(f"Entities to exclude: {config.entities_to_exclude}")

    assert config.prompt_format != "aeio", "AEiO prompting is not supported in Relation Extraction."

    re_pipeline = REPipeline(config=config, data_path=config.test_path, cot=True)

    if "generation" in config.tasks:
        re_pipeline.generate_gpt_responses()

    if "evaluation" in config.tasks:
        metrics_pm, metrics_fm = re_pipeline.load_and_eval(report_dir=config.report_dir if config.save_report else None)
        logger.info(f"Metrics partial match:\n{prettify_json(json.dumps(metrics_pm, indent=2), collapse_level=3)}")
        logger.info(f"Metrics full match:\n{prettify_json(json.dumps(metrics_fm, indent=2), collapse_level=3)}")

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
