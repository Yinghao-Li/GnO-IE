"""
# Modified: February 15th, 2024
# ---------------------------------------
# Description: Interaction functions
"""

import json
import logging
from seqlbtoolkit.io import save_json

from src.core.llm import GPT, GPTMessageCache

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


class REInteraction:
    def __init__(
        self,
        gpt_resource_path: str,
        relation_types: list[str] = None,
        relation2prompt: dict[str, str] = None,
        relation2cols: dict[str, list[str]] = None,
        cot: bool = False,
        prompt_format: str = "ino",
    ) -> None:
        assert relation_types is not None, "Please specify the relation types."

        self.cot = cot
        self.gpt = GPT(resource_path=gpt_resource_path)
        self.relation_types = relation_types

        self.messagers = None
        self.prompt_format = prompt_format

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

    def interact(self, sentence: str) -> dict:
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

        self.init_messagers()
        response_dict = dict()

        for relation_type, prompter in zip(self.relation_types, self.prompters):
            response = self.interact_each_relation_type(prompter, self.messagers[relation_type], sentence)
            response_dict[relation_type] = response

        return response_dict

    def interact_each_relation_type(self, prompter, messager, sentence) -> None:
        messager.add_user_message(prompter.get_re_prompt(sentence))
        gpt_response = self.get_gpt_response(messager)

        if self.prompt_format == "ino":
            messager.add_user_message(prompter.get_formatting_prompt())
            gpt_response = self.get_gpt_response(messager)

        return gpt_response

    def get_gpt_response(self, messager) -> str:
        response = self.gpt(messager)
        messager.add_assistant_message(response)
        return response

    def save_messages(self, path: str) -> None:
        for idx, messager in enumerate(self.messagers):
            suffix = path.split["."][-1]
            messager.save_plain(f"{path.rstrip(suffix)}-{idx}.txt")

    def init_messagers(self):
        self.messagers = {rt: GPTMessageCache(system_role=SYSTEM_ROLE) for rt in self.relation_types}
        return self

    def save_messagers(self, path: str):
        if not self.messagers:
            logger.warning("Messagers are not initialized.")
            return self

        serializable_messagers = {rt: messager.content for rt, messager in self.messagers.items()}
        save_json(serializable_messagers, path, collapse_level=3)

        return self

    def load_messagers(self, path: str):
        self.init_messagers()

        with open(path, "r", encoding="utf-8") as f:
            serializable_messagers = json.load(f)

        for rt, content in serializable_messagers.items():
            self.messagers[rt] = GPTMessageCache(content)

        return self
