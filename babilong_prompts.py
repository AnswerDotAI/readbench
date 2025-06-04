"""
PROMPTS ADAPTED FROM THE ORIGIANL BABILONG REPOSITORY UNDER APACHE 2.0 LICENSE
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Union
COT_PROMPTS: Dict[str, Dict[str, str]] = {
    'qa1': {
        'instruction':
            'I will give you context with the facts about positions of different persons hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location to answer the question.',
        'post_prompt':
            'Think step by step and then end your response with Answer: followed by "The most recent location of $person$ is $location$".'
    },
    'qa2': {
        'instruction':
            'I give you context with the facts about locations and actions of different persons '
            'hidden in some random text and a question.'
            'You need to answer the question based only on the information from the facts.\n'
            'If a person got an item in the first location and travelled to the second location '
            'the item is also in the second location. '
            'If a person dropped an item in the first location and moved to the second location '
            'the item remains in the first location.',
        'post_prompt':
            'Think step by step and then end your response with Answer: followed by "The $item$ is in $location$".'
    },
    'qa3': {
        'instruction':
            'I give you context with the facts about locations and actions of different persons '
            'hidden in some random text and a question. '
            'You need to answer the question based only on the information from the facts.\n'
            'If a person got an item in the first location and travelled to the second location '
            'the item is also in the second location. '
            'If a person dropped an item in the first location and moved to the second location '
            'the item remains in the first location.',
        'post_prompt':
            'Think step by step and then end your response with Answer: followed by "Before the $location_1$ the $item$ was in the $location_2$".'
    },
    'qa4': {
        'instruction':
            'I will give you context with the facts about different people, their location and actions, hidden in '
            'some random text and a question. '
            'You need to answer the question based only on the information from the facts.',
        'post_prompt':
            'Think step by step and then end your response with Answer: followed by the location you have identified.'
    },
    'qa5': {
        'instruction':
            'I will give you context with the facts about locations and their relations hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts.',
        'post_prompt':
            'Think step by step and end your response with Answer: followed by a single word corresponding to the answer.'
    },
    'qa6': {
        'instruction':
            'I will give you context with the facts about people and their locations hidden in some random text and a '
            'question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location the person was in to answer the question.',
        'post_prompt':
            'Think step by step and then end your response with Answer: followed by "yes" or "no".'
    },
    'qa7': {
        'instruction':
            'I will give you context with the facts about people and objects they carry, hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts.',
        'post_prompt':
            'Think step by step and then end your response with Answer: followed by "none" or the number of objects (e.g. "one", "two", etc...).'
    },
    'qa8': {
        'instruction':
            'I will give you context with the facts about people and objects they carry, hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts.',
        'post_prompt':
            'Think step by step and then end your response with Answer: followed by "nothing" or "$object$" or "$object_1$, $object_2$".'
    },
    'qa9': {
        'instruction':
            'I will give you context with the facts about people and their locations hidden in some random text and '
            'a question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location the person was in to answer the question.',
        'post_prompt':
            'Think step by step and then end your response with Answer: followed by "yes" or "no".'
    },
    'qa10': {
        'instruction':
            'I will give you context with the facts about people and their locations hidden in some random text and a '
            'question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location the person was in to answer the question.',
        'post_prompt':
            'Think step by step, then end your response with Answer: followed by "yes" or "no" or "maybe".'
    },
}

DEFAULT_PROMPTS: Dict[str, Dict[str, str]] = {
    'qa1': {
        'instruction':
            'I will give you context with the facts about positions of different persons hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location to answer the question.',
        'post_prompt':
            'Always return your answer in the following format: '
            'The most recent location of ’person’ is ’location’. Do not write anything else after that.'
    },
    'qa2': {
        'instruction':
            'I give you context with the facts about locations and actions of different persons '
            'hidden in some random text and a question.'
            'You need to answer the question based only on the information from the facts.\n'
            'If a person got an item in the first location and travelled to the second location '
            'the item is also in the second location. '
            'If a person dropped an item in the first location and moved to the second location '
            'the item remains in the first location.',
        'post_prompt':
            'Always return your answer in the following format: The ’item’ is in ’location’. '
            'Do not write anything else after that.'
    },
    'qa3': {
        'instruction':
            'I give you context with the facts about locations and actions of different persons '
            'hidden in some random text and a question. '
            'You need to answer the question based only on the information from the facts.\n'
            'If a person got an item in the first location and travelled to the second location '
            'the item is also in the second location. '
            'If a person dropped an item in the first location and moved to the second location '
            'the item remains in the first location.',
        'post_prompt':
            'Always return your answer in the following format: '
            'Before the $location_1$ the $item$ was in the $location_2$. Do not write anything else after that.'
    },
    'qa4': {
        'instruction':
            'I will give you context with the facts about different people, their location and actions, hidden in '
            'some random text and a question. '
            'You need to answer the question based only on the information from the facts.',
        'post_prompt':
            'Your answer should contain only one word - location. Do not write anything else after that.'
    },
    'qa5': {
        'instruction':
            'I will give you context with the facts about locations and their relations hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts.',
        'post_prompt':
            'Your answer should contain only one word. Do not write anything else after that. '
            'Do not explain your answer.'
    },
    'qa6': {
        'instruction':
            'I will give you context with the facts about people and their locations hidden in some random text and a '
            'question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location the person was in to answer the question.',
        'post_prompt':
            'Your answer should contain only one word - $yes$ or $no$. Do not write anything else after that. '
            'Do not explain your answer.'
    },
    'qa7': {
        'instruction':
            'I will give you context with the facts about people and objects they carry, hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts.',
        'post_prompt':
            'Your answer should contain only one word - none or $number_of_objects$. '
            'Do not write anything else after that. Do not explain your answer.',
    },
    'qa8': {
        'instruction':
            'I will give you context with the facts about people and objects they carry, hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts.',
        'post_prompt':
            'Your answer should contain only one or two words: $nothing$ or $object$ or $object_1$, $object_2$. '
            'Do not write anything else. Do not explain your answer.'
    },
    'qa9': {
        'instruction':
            'I will give you context with the facts about people and their locations hidden in some random text and '
            'a question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location the person was in to answer the question.',
        'post_prompt':
            'Your answer should contain only one word - $yes$ or $no$. Do not write anything else. '
            'Do not explain your answer.'
    },
    'qa10': {
        'instruction':
            'I will give you context with the facts about people and their locations hidden in some random text and a '
            'question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location the person was in to answer the question.',
        'post_prompt':
            'Your answer should contain only one word - $yes$ or $no$ or $maybe$. Do not write anything else. '
            'Do not explain your answer.'
    },
}



# --------------------------------------------------------------------------- #
#                                                   PROMPT-BUILDING HELPERS   #
# --------------------------------------------------------------------------- #
TextOrImg = Union[str, Path]


def _mmify_instruction(instr: str) -> str:
    """
    Convert “context” wording to “images with text” so the instruction makes
    sense when the haystack is delivered as PNGs instead of inline text.
    """
    instr = re.sub(r"\b[Ii] (give|will give) you context",
                   "I will give you images with text", instr)
    return instr.replace(" context ", " images with text ").replace(" context.", " images with text.")


def build_prompt(rec: dict, mode: str = "text", cot: bool = False) -> List[TextOrImg]:
    qa_tag = rec.get("subset", "qa1")
    meta   = DEFAULT_PROMPTS[qa_tag]         # raises KeyError if bad tag
    if cot:
        meta = COT_PROMPTS[qa_tag]

    instruction  = meta["instruction"]
    post_prompt  = meta["post_prompt"]
    question     = rec["prompt_text"]

    if mode == "text":
        context_text = Path(rec["text_file"]).read_text()
        prompt = (
            f"{instruction}\n"
            f"<context>\n{context_text}\n</context>\n"
            f"<question>\n{question}\n</question>\n"
            f"{post_prompt}"
        )
        return [prompt]

    elif mode == "multimodal":
        instruction_mm = _mmify_instruction(instruction)
        tail = (
            f"<question>\n{question}\n</question>\n"
            f"{post_prompt}"
        )
        return [instruction_mm] + rec["images"] + [tail]

    else:
        raise ValueError(f"Unknown mode {mode!r}")
