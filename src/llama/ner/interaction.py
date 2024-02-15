"""
# Modified: February 15th, 2024
# ---------------------------------------
# Description: Interaction functions
"""

import json
import logging
import torch
import os.path as osp
from vllm import LLM, SamplingParams
from seqlbtoolkit.io import save_json

from src.core.llm import LlamaMessageCache

logger = logging.getLogger(__name__)


class InOPrompt:
    def __init__(
        self,
        entity_type: str,
        entity_prompt: str = "",
        entity_columns: list[str] = None,
        cot: bool = False,
    ) -> None:
        self.entity_type = entity_type
        self.entity_prompt = entity_prompt
        self.entity_columns = entity_columns
        self.cot = cot

    def get_ner_prompt(self, sentence: str) -> str:
        if self.entity_prompt:
            prompt = f"{self.entity_prompt.strip()}\n\n"
        else:
            prompt = f'Please identify the "{self.entity_type.capitalize()}" entities in the following paragraph.\n\n'

        prompt += f"Paragraph: {sentence}\n\n"

        if self.cot:
            prompt += f"Let's think step by step.\n\n"
        return prompt.strip()

    def remove_uncertain_entities_prompt(self) -> str:
        # prompt = f'Please remove entities that do not clearly refer to "{self.entity_type.capitalize()}" \n\n'
        prompt = f"Please remove irrelevant entities and only keep the entities that clearly refer to {self.entity_type}.\n\n"
        return prompt.strip()

    def get_formatting_prompt(self) -> str:
        if len(self.entity_columns) > 1:
            prompt = f'Please present the valid entities as a Markdown table with columns "{self.entity_columns}".\n\n'
        else:
            prompt = (
                f'Please present the valid entities as a Markdown table with one column "{self.entity_columns[0]}".\n\n'
            )
        prompt += f"Make sure to present the entities precisely in the same words as in the original paragraph.\n\n"
        return prompt.strip()


class OneStepPrompt:
    def __init__(
        self,
        entity_type: str,
        entity_prompt: str = "",
        entity_columns: list[str] = None,
        cot: bool = False,
    ) -> None:
        self.entity_type = entity_type
        self.entity_prompt = entity_prompt
        self.entity_columns = entity_columns
        self.cot = cot

    def get_ner_prompt(self, sentence: str) -> str:
        if self.entity_prompt:
            prompt = f"{self.entity_prompt.strip()}"
        else:
            prompt = f'Please identify the "{self.entity_type.capitalize()}" entities in the following paragraph and '

        if len(self.entity_columns) > 1:
            prompt += f' Please present the valid entities as a Markdown table with columns "{self.entity_columns}". '
        else:
            prompt += (
                f' Please present the valid entities as a Markdown table with one column "{self.entity_columns[0]}". '
            )

        prompt += f"Make sure to present the entities precisely in the same words as in the original paragraph.\n\n"

        prompt += f"Paragraph: {sentence}\n\n"

        if self.cot:
            prompt += f"Let's think step by step.\n\n"

        return prompt.strip()

    def remove_uncertain_entities_prompt(self) -> str:
        prompt = f'Please remove entities that do not clearly refer to "{self.entity_type.capitalize()}" and '

        if len(self.entity_columns) > 1:
            prompt += f'present the valid entities as a Markdown table with columns "{self.entity_columns}".\n\n'
        else:
            prompt += f'present the valid entities as a Markdown table with one column "{self.entity_columns[0]}".\n\n'
        prompt += f"Make sure to present the entities precisely in the same words as in the original paragraph.\n\n"

        return prompt.strip()


class Interaction:
    def __init__(
        self,
        model_dir: str,
        entity_types: list[str] = None,
        entity2elaboration: dict[str, str] = None,
        entity2prompt: dict[str, str] = None,
        entity2columns: dict[str, str] = None,
        cot: bool = True,
        prompt_format: str = "ino",
        context_list: list[str] = None,
        max_tokens: int = 4096,
        load_model: bool = True,
        model_dtype: str = "auto",
        disable_result_removal: bool = False,
    ) -> None:
        assert entity_types is not None, "Please specify the entity types."
        self.cot = cot
        if load_model:
            self.model = LLM(model=model_dir, tensor_parallel_size=torch.cuda.device_count(), dtype=model_dtype)
        else:
            self.model = None
        self.sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        self.entity_types = entity_types
        self.messagers = {
            entity_type: [LlamaMessageCache() for _ in range(len(context_list))] for entity_type in self.entity_types
        }

        self.prompt_format = prompt_format
        self.context_list = context_list
        self.disable_result_removal = disable_result_removal

        if prompt_format == "ino":
            self.prompters = [
                InOPrompt(
                    entity_type=entity2elaboration[ent_type],
                    entity_prompt=entity2prompt[ent_type],
                    entity_columns=entity2columns[ent_type],
                    cot=self.cot,
                )
                for ent_type in entity_types
            ]
        elif prompt_format == "one_step":
            self.prompters = [
                OneStepPrompt(
                    entity_type=entity2elaboration[ent_type],
                    entity_prompt=entity2prompt[ent_type],
                    entity_columns=entity2columns[ent_type],
                    cot=self.cot,
                )
                for ent_type in entity_types
            ]

    def interact(self) -> None:
        response_dict = dict()

        for entity_type, prompter in zip(self.entity_types, self.prompters):
            logger.info(f"Generating response for {entity_type}...")

            et_messagers = self.messagers[entity_type]

            response = self.ner_prompter_interaction(prompter, et_messagers)
            response_dict[entity_type] = response

        return response_dict

    def ner_prompter_interaction(self, prompter, messagers) -> None:
        for context, messager in zip(self.context_list, messagers):
            messager.add_user_message(prompter.get_ner_prompt(context))

        logger.info("Generating responses...")
        responses = self.get_response(messagers)

        if not self.disable_result_removal:
            for messager, response in zip(messagers, responses):
                messager.add_assistant_message(response)
                messager.add_user_message(prompter.remove_uncertain_entities_prompt())

            logger.info("Removing irrelevant entities...")
            responses = self.get_response(messagers)

        if self.prompt_format == "ino":
            for messager, response in zip(messagers, responses):
                messager.add_assistant_message(response)
                messager.add_user_message(prompter.get_formatting_prompt())

            logger.info("Formatting responses...")
            responses = self.get_response(messagers)

        for messager, response in zip(messagers, responses):
            messager.add_assistant_message(response)

        return responses

    def get_response(self, messagers) -> str:
        prompts = [messager.text for messager in messagers]

        llm_outputs = self.model.generate(prompts, self.sampling_params)
        llm_responses = [output.outputs[0].text.strip() for output in llm_outputs]

        return llm_responses

    def save_messagers(self, result_dir: str, messager_ids: list[int]) -> None:
        assert len(messager_ids) == len(
            self.messagers[self.entity_types[0]]
        ), "The number of messager ids should be the same as the number of messagers."

        for idx, msg_idx in enumerate(messager_ids):
            msgs = {et: self.messagers[et][idx].content for et in self.entity_types}
            save_json(msgs, osp.join(result_dir, f"messagers-{msg_idx:05d}.json"))

        return None

    def load_messagers(self, messagers_dir: str, ids: list[int]) -> None:
        messagers = {ent: list() for ent in self.entity_types}
        for idx in ids:
            with open(osp.join(messagers_dir, f"messagers-{idx:05d}.json"), "r", encoding="utf-8") as f:
                msgs = json.load(f, strict=False)

            for ent in self.entity_types:
                messagers[ent].append(LlamaMessageCache(content=msgs[ent]))

        self.messagers = messagers

        return self
