"""
# Modified: February 15th, 2024
# ---------------------------------------
# Description: Interaction functions
"""

import json
import torch
import logging
import os.path as osp
from vllm import LLM, SamplingParams
from seqlbtoolkit.io import save_json

from src.core.llm import LlamaMessageCache

logger = logging.getLogger(__name__)

SYSTEM_ROLE = (
    "You are a knowledgeable assistant specialized in recognizing and "
    "understanding named entities and their interrelations. "
    "When requested to organize information in tabular format, you are adept at "
    "filtering and presenting only the relevant and valid results. "
    "You will exclude any results that are not pertinent or are inaccurate "
    "from the table according to the discussion history."
)


class InOPrompt:
    def __init__(
        self,
        relation_prompt: str,
        relation_columns: list[str],
        cot: bool = False,
    ) -> None:
        self.relation_prompt = relation_prompt
        self.relation_columns = relation_columns
        self.cot = cot

    def get_re_prompt(self, sentence: str) -> str:
        prompt = self.relation_prompt.strip() + "\n\n"
        prompt += f"Paragraph: {sentence}\n\n"
        if self.cot:
            prompt += f"Let's think step by step.\n\n"
        return prompt.strip()

    def get_formatting_prompt(self) -> str:
        prompt = (
            f'If exists, please present the valid relationships as a Markdown table with columns "{self.relation_columns}". '
            "Make sure the table items are from the original paragraph.\n\n"
        )
        return prompt.strip()


class OneStepPrompt:
    def __init__(
        self,
        relation_prompt: str,
        relation_columns: list[str],
        cot: bool = False,
    ) -> None:
        self.relation_prompt = relation_prompt
        self.relation_columns = relation_columns
        self.cot = cot

    def get_re_prompt(self, sentence: str) -> str:
        prompt = self.relation_prompt.strip() + "\n\n"
        prompt += (
            f'If exists, please present the valid relationships as a Markdown table with columns "{self.relation_columns}". '
            "Make sure the table items are from the original paragraph.\n\n"
        )
        prompt += f"Paragraph: {sentence}\n\n"
        if self.cot:
            prompt += f"Let's think step by step.\n\n"
        return prompt.strip()


class Interaction:
    def __init__(
        self,
        model_dir: str,
        relation_types: list[str] = None,
        relation2prompt: dict[str, str] = None,
        relation2cols: dict[str, list[str]] = None,
        cot: bool = False,
        prompt_format: str = "ino",
        context_list: list[str] = None,
        max_tokens: int = 4096,
        max_model_len: int = None,
        load_model: bool = True,
        model_dtype: str = "auto",
    ) -> None:
        assert relation_types is not None, "Please specify the relation types."

        if load_model:
            self.model = LLM(
                model=model_dir,
                tensor_parallel_size=torch.cuda.device_count(),
                dtype=model_dtype,
                max_model_len=max_model_len,
            )
        else:
            self.model = None
        self.sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)

        self.cot = cot
        self.relation_types = relation_types

        self.messagers = {rt: [LlamaMessageCache() for _ in range(len(context_list))] for rt in self.relation_types}

        self.prompt_format = prompt_format
        self.context_list = context_list

        if prompt_format == "ino":
            self.prompters = [
                InOPrompt(
                    relation_prompt=relation2prompt[rt],
                    relation_columns=relation2cols[rt],
                    cot=self.cot,
                )
                for rt in relation_types
            ]
        elif prompt_format == "one_step":
            self.prompters = [
                OneStepPrompt(
                    relation_prompt=relation2prompt[rt],
                    relation_columns=relation2cols[rt],
                    cot=self.cot,
                )
                for rt in relation_types
            ]

    def interact(self) -> dict:
        """
        Interact with GPT to extract relations with give types from the input sentence.

        Parameters
        ----------
        sentence: str
            The sentence to be processed.

        Returns
        -------
        response_dict: dict
            A dictionary of relation_type: response pairs.
        """

        response_dict = dict()

        for relation_type, prompter in zip(self.relation_types, self.prompters):
            logger.info(f"Generating response for {relation_type}...")

            rt_messagers = self.messagers[relation_type]

            response = self.interact_each_relation_type(prompter, rt_messagers)
            response_dict[relation_type] = response

        return response_dict

    def interact_each_relation_type(self, prompter, messagers) -> None:
        for context, messager in zip(self.context_list, messagers):
            messager.add_user_message(prompter.get_re_prompt(context))

        logger.info("Generating responses...")
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
            self.messagers[self.relation_types[0]]
        ), "The number of messager ids should be the same as the number of messagers."

        for idx, msg_idx in enumerate(messager_ids):
            msgs = {rt: self.messagers[rt][idx].content for rt in self.relation_types}
            save_json(msgs, osp.join(result_dir, f"messagers-{msg_idx:05d}.json"))

        return None

    def load_messagers(self, messagers_dir: str, ids: list[int]) -> None:
        messagers = {rt: list() for rt in self.relation_types}
        for idx in ids:
            with open(osp.join(messagers_dir, f"messagers-{idx:05d}.json"), "r", encoding="utf-8") as f:
                msgs = json.load(f, strict=False)
                "--prompt_format", "one_step"

            for rt in self.relation_types:
                messagers[rt].append(LlamaMessageCache(content=msgs[rt]))

        self.messagers = messagers

        return self
