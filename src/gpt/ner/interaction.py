"""
# Modified: February 15th, 2024
# ---------------------------------------
# Description: Interaction functions
"""

import json
import logging
from typing import Union
from seqlbtoolkit.io import save_json

from src.core.llm import GPT, GPTMessageCache

logger = logging.getLogger(__name__)

SYSTEM_ROLE = (
    "You are a knowledgeable assistant specialized in recognizing and "
    "understanding named entities and their interrelations. "
    "If requested to organize information in tabular format, you are adept at "
    "filtering and presenting only the relevant and valid results. "
    "You will exclude any results that are not pertinent or are inaccurate "
    "from the table according to the discussion history."
)


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
        gpt_resource_path: str,
        entity_types: list[str] = None,
        entity2elaboration: dict[str, str] = None,
        entity2prompt: dict[str, str] = None,
        entity2columns: dict[str, str] = None,
        cot: bool = False,
        prompt_format: str = "ino",
        disable_result_removal: bool = False,
    ) -> None:
        assert entity_types is not None, "Please specify the entity types."
        self.cot = cot
        self.gpt = GPT(resource_path=gpt_resource_path)
        self.entity_types = entity_types
        self.messagers = None
        self.prompt_format = prompt_format
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

    def interact(self, sentence: str) -> None:
        self.init_messagers()
        response_dict = dict()
        for entity_type, prompter in zip(self.entity_types, self.prompters):
            messager = self.messagers[entity_type]
            response = self.ner_prompter_interaction(prompter, messager, sentence)
            response_dict[entity_type] = response
        return response_dict

    def ner_prompter_interaction(self, prompter, messager, sentence) -> None:
        messager.add_user_message(prompter.get_ner_prompt(sentence))
        gpt_response = self.get_gpt_response(messager)

        if not self.disable_result_removal:
            messager.add_user_message(prompter.remove_uncertain_entities_prompt())
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

    def init_messagers(self) -> None:
        self.messagers = {entity_type: GPTMessageCache(system_role=SYSTEM_ROLE) for entity_type in self.entity_types}
        return self

    def save_messagers(self, path: str) -> None:
        if not self.messagers:
            logger.warning("Messagers are not initialized.")
            return None
        serializable_messagers = {entity_type: messager.content for entity_type, messager in self.messagers.items()}
        save_json(serializable_messagers, path, collapse_level=3)

    def load_messagers(self, path: str) -> None:
        self.init_messagers()
        with open(path, "r", encoding="utf-8") as f:
            serializable_messagers = json.load(f)
        for entity_type, content in serializable_messagers.items():
            self.messagers[entity_type] = GPTMessageCache(content)
        return self


class ConflictResolutionPrompt:
    def __init__(self, entity_types: str, cot: bool = False) -> None:
        self.entity_types = entity_types
        self.cot = cot

    def conflict_resolution_prompt(self, tokens: list[str], gpt_responses: dict) -> str:
        sentence = " ".join(tokens)
        prompt = (
            f"According to the following paragraph, please identify and resolve the conflicts in the Named Entity Recognition (NER) results:\n\n"
            f"Paragraph: {sentence}\n\n"
        )
        for idx, response in enumerate(gpt_responses.values()):
            prompt += f"NER Response {idx + 1}: {response}\n\n"
        if self.cot:
            prompt += f"Let's think step by step.\n\n"
        return prompt.strip()

    def get_formatting_prompt(self) -> str:
        prompt = (
            f'Please present the entities with corrected entity types as a Markdown table with columns ["Entity", "Entity Type"]. '
            f"Make sure the Entities are expressed in the same words as the original answers and the Entity Types in {self.entity_types}.\n\n"
        )
        return prompt.strip()


class ConflictResolutionInteraction:
    def __init__(
        self,
        gpt_resource_path: str,
        entity_types: Union[str, list[str]] = None,
        cot: bool = False,
    ) -> None:
        assert entity_types is not None, "Please specify the entity types."
        if isinstance(entity_types, str):
            entity_types = [entity_types]
        self.cot = cot
        self.gpt = GPT(resource_path=gpt_resource_path)
        self.entity_types = entity_types
        self.messager = None

        self.prompter = ConflictResolutionPrompt(entity_types=entity_types, cot=self.cot)

    def interact(self, tokens, gpt_responses) -> None:
        self.init_messager()
        self.messager.add_user_message(self.prompter.conflict_resolution_prompt(tokens, gpt_responses=gpt_responses))
        gpt_response = self.get_gpt_response(self.messager)

        self.messager.add_user_message(self.prompter.get_formatting_prompt())
        gpt_response = self.get_gpt_response(self.messager)

        return gpt_response

    def get_gpt_response(self, messager) -> str:
        response = self.gpt(messager)
        messager.add_assistant_message(response)
        return response

    def init_messager(self) -> None:
        self.messager = GPTMessageCache(system_role=SYSTEM_ROLE)
        return self

    def save_messager(self, path: str) -> None:
        if not self.messager:
            logger.warning("Messager is not initialized.")
            return None
        save_json(self.messager.content, path, collapse_level=2)

    def load_messager(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
        self.messager = GPTMessageCache(content)
        return self


class AEiOPrompt:
    def __init__(self, entity_types: list[str], cot: bool = False) -> None:
        self.entity_types = entity_types
        self.cot = cot

    def get_ner_prompt(self, sentence: str) -> str:
        prompt = f'Please identify the "{self.entity_types}" entities in the following paragraph.\n\n'
        prompt += f"Paragraph: {sentence}\n\n"
        if self.cot:
            prompt += f"Let's think step by step.\n\n"
        return prompt.strip()

    def remove_uncertain_entities_prompt(self) -> str:
        prompt = f'Please remove entities that do not clearly refer to any of "{self.entity_types}" \n\n'
        return prompt.strip()

    def get_formatting_prompt(self) -> str:
        prompt = (
            f'Please present the valid entities as a Markdown table with columns ["Entity", "Entity Type"]. '
            f"Make sure to present the entities precisely in the same words as in the original paragraph "
            f"and the Entity Types are in {self.entity_types}.\n\n"
        )
        return prompt.strip()


class AEiOInteraction:
    def __init__(
        self,
        gpt_resource_path: str,
        entity_types: Union[str, list[str]] = None,
        entity2elaboration: dict[str, str] = None,
        cot: bool = False,
    ) -> None:
        assert entity_types is not None, "Please specify the entity types."
        if isinstance(entity_types, str) or len(entity_types) == 1:
            raise ValueError("Please specify at least two entity types.")
        self.entity_types = entity_types
        self.cot = cot
        self.gpt = GPT(resource_path=gpt_resource_path)
        self.messager = None

        self.prompter = AEiOPrompt(
            entity_types=[entity2elaboration[et] for et in entity_types],
            cot=self.cot,
        )

    def interact(self, sentence) -> None:
        self.init_messager()
        self.messager.add_user_message(self.prompter.get_ner_prompt(sentence))
        gpt_response = self.get_gpt_response(self.messager)

        self.messager.add_user_message(self.prompter.remove_uncertain_entities_prompt())
        gpt_response = self.get_gpt_response(self.messager)

        self.messager.add_user_message(self.prompter.get_formatting_prompt())
        gpt_response = self.get_gpt_response(self.messager)

        return gpt_response

    def get_gpt_response(self, messager) -> str:
        response = self.gpt(messager)
        messager.add_assistant_message(response)
        return response

    def init_messager(self) -> None:
        self.messager = GPTMessageCache(system_role=SYSTEM_ROLE)
        return self

    def save_messager(self, path: str) -> None:
        if not self.messager:
            logger.warning("Messager is not initialized.")
            return None
        save_json(self.messager.content, path, collapse_level=2, disable_content_checking=True)

    def load_messager(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
        self.messager = GPTMessageCache(content)
        return self


# NCBI_GUIDELINES = """
# What to Annotate:
# - Annotate all Specific Disease mentions
# - Annotate contiguous text strings.
# - Annotate disease mentions that are used as Modifiers for other concepts
# - Annotate duplicate mentions
# - Annotate minimum necessary span of text
# - Annotate all synonymous mentions

# What NOT to Annotate:
# - Do not annotate organism names
# - Do not annotate gender
# - Do not annotate overlapping mentions
# - Do not annotate general terms
# - Do not annotate references to biological processes
# - Do not annotate disease mentions interrupted by nested mentions

# """
