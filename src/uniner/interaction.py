"""
# Modified: February 15th, 2024
# ---------------------------------------
# Description: Interaction functions
"""

import logging

from vllm import LLM, SamplingParams
from tqdm import tqdm

logger = logging.getLogger(__name__)


class UniNERPrompt:
    def __init__(
        self,
        entity_types: str,
    ) -> None:
        self.entity_types = entity_types
        self.prompt = None

    def initialize_prompt(self, context):
        self.prompt = "A virtual assistant answers questions from a user based on the provided text.\n"
        self.prompt += f"USER: Text: {context}\n"
        self.prompt += "ASSISTANT: I’ve read this text.\n"
        return self.prompt

    def append_next_entity_prompt(self):
        next_entity_type = self.entity_types.pop(0)
        self.prompt += f"USER: What describes {next_entity_type} in the text?\n"
        self.prompt += "ASSISTANT: "
        return self.prompt

    def __repr__(self) -> str:
        return self.prompt


class Interactions:
    def __init__(
        self,
        model_dir: str,
        entity_types: list[str] = None,
        entity2uniner: dict[str, str] = None,
        context_list: list[str] = None,
        max_tokens: int = 256,
    ) -> None:
        assert entity_types is not None, "Please specify the entity types."
        self.model = LLM(model=model_dir)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        self.entity_types = entity_types

        self.prompters = [
            UniNERPrompt(entity_types=[entity2uniner[et] for et in entity_types]) for _ in range(len(context_list))
        ]
        self.context_list = context_list

        self.init_prompts()

    def init_prompts(self):
        logger.info("Initializing prompts...")
        assert len(self.prompters) == len(self.context_list), "The number of prompts and contexts must be the same."
        for prompter, context in tqdm(zip(self.prompters, self.context_list), total=len(self.context_list)):
            prompter.initialize_prompt(context)
        return None

    def interact(self) -> None:
        responses = {}
        for entity_type in self.entity_types:
            cand_ent_type = self.prompters[0].entity_types[0]
            logger.info(f"Appending {cand_ent_type} entity type to prompts...")
            for prompter in tqdm(self.prompters):
                prompter.append_next_entity_prompt()

            prompts = [prompter.prompt for prompter in self.prompters]
            logger.info("Generating response...")
            llm_outputs = self.model.generate(prompts, self.sampling_params)

            llm_responses = [output.outputs[0].text.strip() for output in llm_outputs]

            assert len(llm_responses) == len(self.prompters), "The number of responses and prompts do not match."
            logger.info(f"Appending response to prompts...")
            for prompter, llm_response in tqdm(zip(self.prompters, llm_responses), total=len(llm_responses)):
                prompter.prompt += f"{llm_response}\n"

            resp_list = list()
            for resp in llm_responses:
                try:
                    resp_list.append(eval(resp))
                except:
                    resp_list.append(list())
            responses[entity_type] = resp_list

        return responses
