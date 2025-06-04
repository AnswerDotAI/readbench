"""
Evaluation code and prompts for the various datasets are adapted from the original evaluation code, released under permissive licenses.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hashlib, json, os, random, re, time, shutil, statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import dotenv; dotenv.load_dotenv()
import typer
from google import genai
from PIL import Image
from tqdm import tqdm

from pathlib import Path
from claudette import Chat as ClaudetteChat
from claudette import Client as ClaudetteClient
from cosette import Chat as CosetteChat
from cosette import Client as CosetteClient
from vertexauth import get_claudette_client
from openai import AzureOpenAI
azure_endpoint = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-12-01-preview",
    )

import babilong_prompts as bp
import longbench_prompts as lbp
import longbench_metrics as lbm

try: CLAUDETTE_CLIENT = get_claudette_client()
except Exception as e: pass

# ────────────────────────────────────────────────────────────── CONFIG
MAX_RETRIES   = 5
RETRY_BACKOFF = 5.0
TextOrImg     = Union[str, Path]

BABILONG_BUCKETS = ["0k", "1k", "2k", "4k", "8k"]
NOLIMA_BUCKETS   = ["100", "500", "1000", "2000", "4000", "8000"]


def length_bucket_tag(ds: str, rec: dict) -> str | None:
    hay = f"{rec.get('id', '')}_{rec.get('text_file', '')}".lower()
    if ds == "babilong":
        for tag in BABILONG_BUCKETS:
            if tag in hay:
                return tag
    elif ds == "nolima":
        for tag in NOLIMA_BUCKETS:
            if re.search(rf"\b{tag}\b", hay):
                return tag
    elif ds == "longbench":
        if "length_bin" in rec and rec["length_bin"]:
            return rec["length_bin"]
        length = rec.get("length") or rec.get("token_length") or 0
        if length < 512:
            return "0-512"
        elif length < 1024:
            return "512-1024"
        elif length < 2048:
            return "1024-2048"
        elif length < 4096:
            return "2048-4096"
        elif length < 8192:
            return "4096-8192"
        elif length < 16384:
            return "8192-16384"
        else:
            return "16384+"
    return None


def question_tag(ds: str, rec: dict) -> str | None:
    """Return the question-set tag (e.g. subset / subject / category)."""
    if ds in {"babilong", "longbench"}:
        return rec.get("subset")
    if ds in {"mmlu_pro", "mmlu_redux", "gpqa"}:
        return rec.get("category")
    return None


# ────────────────────────────────────────────────────────────── PRICING
PRICING_RATES = {
    # GPT-4.1
    "gpt-4.1-2025-04-14": {"in": 2.00 / 1_000_000, "out": 8.00 / 1_000_000},
    "gpt-4.1-mini-2025-04-14": {"in": 0.40 / 1_000_000, "out": 1.60 / 1_000_000},
    "gpt-4.1-nano-2025-04-14": {"in": 0.10 / 1_000_000, "out": 0.40 / 1_000_000},
    # GPT-4o
    "gpt-4o-2024-08-06": {"in": 2.50 / 1_000_000, "out": 10.00 / 1_000_000},
    # GPT-4o-mini
    "gpt-4o-mini-2024-07-18": {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
    # o1
    "o1-2024-12-17": {"in": 15.00 / 1_000_000, "out": 60.00 / 1_000_000},
    # o1-pro
    "o1-pro-2025-03-19": {"in": 150.00 / 1_000_000, "out": 600.00 / 1_000_000},
    # o3
    "o3-2025-04-16": {"in": 10.00 / 1_000_000, "out": 40.00 / 1_000_000},
    # o4-mini
    "o4-mini-2025-04-16": {"in": 1.10 / 1_000_000, "out": 4.40 / 1_000_000},
    # o3-mini
    "o3-mini-2025-01-31": {"in": 1.10 / 1_000_000, "out": 4.40 / 1_000_000},
    # o1-mini
    "o1-mini-2024-09-12": {"in": 1.10 / 1_000_000, "out": 4.40 / 1_000_000},
    "gemini-2.0-flash":      {"in": 0.10 / 1_000_000,   "out": 0.40 / 1_000_000},
    "gemini-2.5-flash-preview-04-17": {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
    "gemini-2.0-flash-lite": {"in": 0.0075 / 1_000_000, "out": 0.30 / 1_000_000},
    "gemini-1.5-pro":        {"in": 1.25 / 1_000_000,   "out": 5.00 / 1_000_000},
    "claude-3-5-sonnet-20241022":     {"in": 3.0 / 1_000_000,   "out": 15.00 / 1_000_000},
    "claude-3-5-haiku-20241022":     {"in": 0.8 / 1_000_000,   "out": 4.00 / 1_000_000},
    "pixtral-large-latest": {"in": 2.00 / 1_000_000, "out": 6.00 / 1_000_000},
    "pixtral-12b-2409": {"in": 0.15 / 1_000_000, "out": 0.15 / 1_000_000},
}

def _model_pricing(model_name: str):
    for k, v in PRICING_RATES.items():
        if k in model_name:
            return v
    return None



import base64
from io import BytesIO
from openai import OpenAI

_QWEN_VLLM_CLIENT: OpenAI | None = None            

def _get_qwen_vllm_client() -> OpenAI:
    global _QWEN_VLLM_CLIENT
    if _QWEN_VLLM_CLIENT is None:
        _QWEN_VLLM_CLIENT = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),           
            base_url=os.getenv("QWEN_VLLM_BASE_URL",
                               "http://0.0.0.0:8004/v1"),
            timeout=60,                                            
        )
    return _QWEN_VLLM_CLIENT


def _encode_image_b64(path: str | Path) -> str:
    path = Path(path)
    mime = "png" if path.suffix.lower() == ".png" else "jpeg"
    return f"data:image/{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _segments_to_qwen_content(segments: Sequence[TextOrImg], max_img: int = 24) -> list[dict]:
    content: list[dict] = []
    img_count = 0
    for seg in segments:
        s = str(seg)
        if s.lower().endswith((".png", ".jpg", ".jpeg")) and Path(s).exists():
            if img_count >= max_img:
                print(f'Skipping image {s} because max_img={max_img} reached')
                continue
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_b64(s)},
                }
            )
            img_count += 1
        else:                              
            content.append({"type": "text", "text": s})
    return content


def qwen_vllm_model_call(
    segments: Sequence[TextOrImg],
    model_name: str,
    *,
    debug: bool = False,
) -> Tuple[str, int | None, int | None]:        
    client = _get_qwen_vllm_client()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": _segments_to_qwen_content(segments)},
    ]

    rsp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.01,
    )

    txt = rsp.choices[0].message.content.strip()
    in_tok = getattr(rsp.usage, "prompt_tokens", None)
    out_tok = getattr(rsp.usage, "completion_tokens", None)

    if debug:
        print("[qwen-vllm]", txt[:300].replace("\n", " ") + "…", flush=True)
        print(f"[qwen-vllm tokens] in={in_tok}  out={out_tok}", flush=True)

    return txt, in_tok, out_tok

def mistral_model_call(segments: Sequence[TextOrImg], model_name: str) -> str:
    cli = OpenAI(api_key=os.getenv("MISTRAL_API_KEY"), base_url='https://api.mistral.ai/v1')

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": _segments_to_qwen_content(segments, max_img=8)},
    ]

    r = cli.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.01,
    )
    in_tok = getattr(r.usage, "prompt_tokens", None)
    out_tok = getattr(r.usage, "completion_tokens", None)
    return r.choices[0].message.content.strip(), in_tok, out_tok

def gemini_model_call(
        segments: Sequence[TextOrImg],
        model_name: str = "gemini-2.0-pro",
        *,
        debug: bool = False,
) -> Tuple[str, int | None, int | None]:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    parts = []
    for seg in segments:
        seg = str(seg)
        if seg.lower().endswith((".png", ".jpg", ".jpeg")):
            parts.append(Image.open(seg))
        else:
            parts.append(seg)

    rsp = client.models.generate_content(
        model=model_name,
        contents=parts,
        config=genai.types.GenerateContentConfig(
            system_instruction="You are a helpful assistant",
            temperature=0.01,
            seed=42,
        ),
    )

    text = getattr(rsp, "text", None)
    if text is None and hasattr(rsp, "candidates"):
        text = rsp.candidates[0].content.parts[0].text

    usage_meta = getattr(rsp, "usage_metadata", None)
    in_tok  = getattr(usage_meta, "prompt_token_count", None) if usage_meta else None
    out_tok = getattr(usage_meta, "candidates_token_count", None) if usage_meta else None

    if debug:
        print("[gemini]", text, flush=True)
        print(f"[gemini tokens] in={in_tok}  out={out_tok}", flush=True)

    return text.strip(), in_tok, out_tok

def claudette_model_call(segments: Sequence[TextOrImg], model_name: str) -> str:
    if '@' in model_name: cli = get_claudette_client()
    else: cli = ClaudetteClient(model_name)
    chat = ClaudetteChat(cli=cli, sp="You are a helpful assistant")
    inp = []
    for seg in segments:
        seg = str(seg)
        if seg.lower().endswith((".png", ".jpg", ".jpeg")):
            inp.append(Path(seg).read_bytes())
        else:
            inp.append(seg)
    r = chat(inp, temp=0.0)
    in_tok = r.usage.input_tokens
    out_tok = r.usage.output_tokens
    
    return r.content[0].text.strip(), in_tok, out_tok

def cosette_model_call(segments: Sequence[TextOrImg], model_name: str) -> str:
    chat = CosetteChat(cli=CosetteClient(model_name, azure_endpoint), sp="You are a helpful assistant")
    inp = []
    for seg in segments:
        seg = str(seg)
        if seg.lower().endswith((".png", ".jpg", ".jpeg")):
            inp.append(Path(seg).read_bytes())
        else:
            inp.append(seg)
    r = chat(inp, temperature=0.0)
    in_tok = r.usage.prompt_tokens
    out_tok = r.usage.completion_tokens
    
    return r.choices[0].message.content.strip(), in_tok, out_tok

def placeholder_model_call(segments: Sequence[TextOrImg], model_name: str) -> str:
    raise NotImplementedError("Plug in your own model here")


def dispatch_model(
        model_name: str,
        segments: Sequence[TextOrImg],
        *,
        debug: bool = False,
) -> Tuple[str, int | None, int | None]:
    if "qwen" in model_name.lower():
        return qwen_vllm_model_call(segments, model_name, debug=debug)
    if "gemini" in model_name.lower():
        return gemini_model_call(segments, model_name, debug=debug)
    if "claude" in model_name.lower():
        return claudette_model_call(segments, model_name)
    if 'pixtral' in model_name.lower():
        return mistral_model_call(segments, model_name)
    if any(x in model_name.lower() for x in ["gpt", "o4", "o3"]):
        return cosette_model_call(segments, model_name)
    txt = placeholder_model_call(segments, model_name)
    return txt, None, None



COT_DEFAULT = {'mmlu_pro': True, 'mmlu_redux': True, 'gpqa': True, 'babilong': False, 'longbench': False, 'boolq': False}

def mmlu_text(rec, cot  = True):
    opts = Path(rec["text_file"]).read_text()
    if cot:
        return [(
            "The following are multiple-choice questions (with answers) about "
            f"{rec['category']}. Think step by step and then finish your answer "
            "with \"The answer is (X)\" where X is the correct letter choice.\n\n"
            f"Question: {rec['prompt_text']}\n\nOptions:\n{opts}"
        )]
    else:
        return [(
            "The following are multiple-choice questions (with answers) about "
            f"{rec['category']}. Do not output reasoning, answer with just the letter of the correct answer in the format \"The answer is (X)\" where X is the correct letter choice.\n\n"
            f"Question: {rec['prompt_text']}\n\nOptions:\n{opts}"
        )]


def mmlu_mm(rec, cot = True):
    head = (
        "The following are multiple-choice questions about "
        f"{rec['category']}. The possible answers are listed in the attached "
        "image.\n\nQuestion: {q}\nOptions:\n"
    ).format(q=rec["prompt_text"])
    if cot:
        tail = ('Think step by step and then output the answer in the format '
                '"The answer is (X)" at the end, where X is the letter associated '
                'with the correct answer.')
    else:
        tail = ('Do not output reasoning, answer with just the letter of the correct answer in the format "The answer is (X)" where X is the correct letter choice.')
    return [head] + rec["images"] + [tail]


def nolima_text(rec):
    hay = Path(rec["text_file"]).read_text()
    return [(
        "You will answer a question based on the following book snippet:\n\n"
        f"{hay}\n\nUse the information provided in the book snippet to answer "
        "the question. Your answer should be short and based on either "
        "explicitly stated facts or strong, logical inferences.\n\n"
        f"Question: {rec['prompt_text']}\n\nReturn only the final answer with "
        "no additional explanation or reasoning."
    )]


def nolima_mm(rec):
    tail = (
        "Use the information provided in the book snippet to answer the "
        "question. Your answer should be short and based on either explicitly "
        "stated facts or strong, logical inferences.\n\n"
        f"Question: {rec['prompt_text']}\n\nReturn only the final answer with "
        "no additional explanation or reasoning."
    )
    return (["You will answer a question based on the following book snippet:"]
            + rec["images"] + [tail])

def boolq_text(rec, cot = False):
    passage = Path(rec["text_file"]).read_text()
    q = rec["prompt_text"]
    if cot:
        return [(
            "Read the passage and answer the question based on the information provided.\n\n"
            f"Passage:\n{passage}\n\nQuestion: {q}\n\n"
            "Think step by step and then answer with yes or no."
        )]
    else:
        return [(
            "Read the passage and answer the question based on the information provided.\n\n"
            f"Passage:\n{passage}\n\nQuestion: {q}\n\n"
            "Do not output reasoning. Answer with yes or no and nothing else."
        )]


def boolq_mm(rec, cot = False):
    head = f"Read the passage below and answer the question based on the information provided."
    if cot:
        tail = f"Question: {rec['prompt_text']}\n\nThink step by step and then answer with yes or no."
    else:
        tail = f"Question: {rec['prompt_text']}\n\nDo not ouput reasoning. Answer with yes or no and nothing else."
    return ([head] + rec["images"] + [tail])


def babi_text(rec, cot = False): 
    return bp.build_prompt(rec, mode="text", cot=cot)

def babi_mm(rec, cot = False):
    return bp.build_prompt(rec, mode="multimodal", cot=cot)

def gpqa_text(rec, cot = True):
    """Build text prompt for GPQA (similar to MMLU)."""
    opts = Path(rec["text_file"]).read_text()
    head = (
        "What is the correct answer to this question: " f"{rec['prompt_text']}\n\nOptions:"
    )
    if cot:
        tail = (
            "Your answer must end with Answer: (LETTER). Let's think step by step: "
        )
    else:
        tail = (
            "Do not output reasoning. Simply respond with Answer: (LETTER)."
        )
    return [head, opts, tail]


def gpqa_mm(rec, cot = True):
    """Build multimodal prompt for GPQA using rendered choices image."""
    head = (
        "What is the correct answer to this question: " f"{rec['prompt_text']}\n\nOptions:"
    )
    if cot:
        tail = (
            "Your answer must end with Answer: (LETTER). Let's think step by step: "
        )
    else:
        tail = (
            "Do not output reasoning. Simply respond with Answer: (LETTER)."
        )
    # Use all pages images (should contain all choices)
    return [head] + rec["images"] + [tail]

def longbench_text(rec, cot = False):
    return lbp.build_prompt(rec, mode="text", cot=cot)

def longbench_mm(rec, cot = False):
    return lbp.build_prompt(rec, mode="multimodal", cot=cot)

PROMPT_BUILDERS = {
    "mmlu_pro": {"text": mmlu_text,   "multimodal": mmlu_mm},
    "mmlu_redux": {"text": mmlu_text, "multimodal": mmlu_mm},
    "gpqa": {"text": gpqa_text, "multimodal": gpqa_mm},
    "nolima":   {"text": nolima_text, "multimodal": nolima_mm},
    "babilong": {"text": babi_text,   "multimodal": babi_mm},
    "longbench": {"text": longbench_text, "multimodal": longbench_mm},
    "boolq": {"text": boolq_text, "multimodal": boolq_mm},
}



ANSWER_RE = re.compile(
    r'[A-K]',  # Simply match any capital letter A through J
    re.DOTALL,  # Allow matching across multiple lines
)

LAST_LETTER_RE = re.compile(
    r"\b([A-J])\b(?!.*\b[A-J]\b)",  # Match A-J as whole words, with negative lookahead to ensure it's the last one
    re.IGNORECASE | re.DOTALL  # Case insensitive, allow matching across lines
)

ANSWER_IS_RE = re.compile(r"answer is[:\s]*\(?([A-J])\)?", re.S)
ANSWER_COLON_RE = re.compile(r'.*[Aa]nswer:\s*([A-J])',  re.S)

def _extract_answer_mmlu(text: str) -> str | None:
    if not text:
        return None
    
    m = ANSWER_COLON_RE.search(text)
    if m:
        return m.group(1).upper()
    
    m = ANSWER_IS_RE.search(text)
    if m:
        return m.group(1).upper()
    
    m = LAST_LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    
    text_tail = text[-10:]
    tail_matches = list(ANSWER_RE.finditer(text_tail))
    if tail_matches:
        return tail_matches[-1].group(0).upper()
    
    pred_tail = text[-300:]
    m = LAST_LETTER_RE.search(pred_tail)
    if m:
        return m.group(1).upper()
    
    return None

def eval_mmlu(pred: str, gold: str) -> bool:
    letter = _extract_answer_mmlu(pred) or random.choice(list("ABCDEFGHIJ"))
    return letter == gold.upper()

_BOOL_YES = {"yes", "true"}
_BOOL_NO  = {"no", "false"}

def _extract_boolq(pred: str, cot: bool = False) -> bool:
    if cot:
        t = pred.lower().split('answer:')[1].strip()
    else:
        t = pred.lower()
    if any(w in t for w in _BOOL_YES): return True
    if any(w in t for w in _BOOL_NO):  return False
    raise ValueError(f"Unknown BoolQ prediction: {pred!r}")

def eval_boolq(pred: str, gold: bool, cot: bool = False) -> bool:
    return _extract_boolq(pred, cot) == bool(gold)

def nolima_official_evaluate_response(resp: str,
                                      gold: List[str],
                                      metric: str) -> bool:
    resp = resp.strip()
    if metric == "EM":
        return resp in gold
    if metric == "contains":
        return any(g in resp for g in gold)
    if metric == "lastline_EM":
        return resp.splitlines()[-1] in gold
    if metric == "lastline_contains":
        last = resp.splitlines()[-1]
        return any(g in last for g in gold)
    raise ValueError(f"Unknown metric {metric!r}")


TASK_LABELS = {
    'qa1':  ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
    'qa2':  ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
    'qa3':  ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
    'qa4':  ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
    'qa5':  ['Bill', 'Fred', 'Jeff', 'Mary', 'apple', 'football', 'milk'],
    'qa6':  ['no', 'yes'],
    'qa7':  ['none', 'one', 'three', 'two'],
    'qa8':  ['apple', 'football', 'milk', 'nothing'],
    'qa9':  ['no', 'yes'],
    'qa10': ['maybe', 'no', 'yes'],
}


def babi_preprocess(out: str) -> str:
    # print('[BABI] out: ', out, flush=True)
    out = out.lower().split('.')[0]
    out = out.split('<context>')[0].split('<example>')[0]
    out = out.split('question')[0]
    out = out.strip('$').strip().lower()
    digit_map = {
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine'
    }
    def _replace_digits(match):
        return digit_map[match.group(0)]
    out = re.sub(r'\b[1-9]\b', _replace_digits, out)
    return out.strip().lower()

def babi_preprocess_cot(out: str) -> str:
    # print('[BABI] out: ', out, flush=True)
    out = out.lower().split('answer:')[1].strip()
    out = out.strip('$').strip().lower()
    digit_map = {
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine'
    }
    def _replace_digits(match):
        return digit_map[match.group(0)]
    out = re.sub(r'\b[1-9]\b', _replace_digits, out)
    return out.strip().lower()

def eval_babi(pred: str, gold: str, question: str, qa_tag: str, cot: bool = False) -> bool:
    if cot:
        pred   = babi_preprocess_cot(pred)
    else:
        pred   = babi_preprocess(pred)
    gold   = gold.lower()
    labels = set(map(str.lower, TASK_LABELS[qa_tag]))

    in_q   = {l for l in labels if l in question.lower()}
    in_q -= set(['none'])
    import uuid
    _id = str(uuid.uuid4())
    # print('in_out pre-filtering: ', {l for l in labels if l in pred}, '(', _id, ')')
    in_out = {l for l in labels if l in pred} - in_q
    # print('in_out post-filtering: ', in_out, '(', _id, ')')

    if "none" in in_out and "one" in in_out:
        in_out.discard("one")

    if ',' in gold and len(gold) > 3:
        sub = gold.split(',')
        return all(t in in_out for t in sub) and len(in_out) == len(sub)
    return gold in in_out and len(in_out) == 1

def eval_longbench(pred: str, rec: dict, cot: bool = False) -> tuple[bool, float]:
    subset = rec.get("subset") or rec.get("lb_subset") \
             or rec.get("dataset_name")
    if not subset:
        raise KeyError("LongBench record missing 'subset' field")

    if subset.endswith('_e'):
        metric_fn = lbm.dataset2metric.get(subset[:-2])
    else:
        metric_fn = lbm.dataset2metric.get(subset)
    if metric_fn is None:
        raise ValueError(f"Unknown LongBench subset '{subset}'")

    pred_proc = pred
    if subset in {"trec", "triviaqa", "samsum", "lsht"}:
        pred_proc = pred.lstrip("\n")
        if not cot: pred_proc = pred_proc.split("\n")[0]
    if cot:
        # print('[LONGBENCH] cot: ', cot, 'pred: ', pred, flush=True)
        if 'Answer:' in pred:
            pred_proc = pred.split('Answer:')[1].strip()
        else:
            pred_proc = pred.split('answer:')[1].strip()

    best = 0.0
    golds = rec.get("answers")
    if not golds:
        g = rec.get("ground_truth")
        golds = g if isinstance(g, list) else [g]
    for g in golds:
        best = max(best,
                   metric_fn(pred_proc, g,
                             all_classes=rec.get("all_classes")))
    return best > 0, best

# ───────────────────────────── WORKER
def _grade(rec: Dict[str, Any], pred: str, cot: bool = False) -> bool:
    ds = rec["dataset"]
    if ds == "mmlu_pro":
        return eval_mmlu(pred, rec["ground_truth"])
    if ds == "mmlu_redux":
        return eval_mmlu(pred, rec["ground_truth"])
    if ds == "gpqa":
        return eval_mmlu(pred, rec["ground_truth"])
    if ds == "nolima":
        return nolima_official_evaluate_response(
            pred,
            rec.get("gold_answers", [rec["ground_truth"]]),
            rec.get("metric", "EM"),
        )
    if ds == "babilong":
        qa_tag = rec.get("subset", "qa1")
        return eval_babi(pred, rec["ground_truth"],
                         rec["prompt_text"], qa_tag, cot)
    if ds == "longbench":
        ok, score = eval_longbench(pred, rec, cot)
        rec["lb_score"] = score
        return ok
    if ds == "boolq":
        return eval_boolq(pred, rec["ground_truth"], cot)
    raise ValueError(f"Unknown dataset {ds!r}")


def worker(args) -> Tuple[str, bool, Dict[str, Any]]:
    rec, mode, model, debug, cot = args
    ds   = rec["dataset"]
    if cot == 'default':
        cot = COT_DEFAULT[ds]
    elif cot == 'revert':
        cot = not COT_DEFAULT[ds]
    elif cot == 'true':
        cot = True
    elif cot == 'false':
        cot = False
    segs = PROMPT_BUILDERS[ds][mode](rec, cot=cot)

    prompt_sent = []
    for seg in segs:
        if isinstance(seg, (str, Path)):
            seg_str = str(seg)
            prompt_sent.append(seg_str)
        else:
            prompt_sent.append(str(seg))

    last_err = ""
    tokens_in = tokens_out = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            out, tokens_in, tokens_out = dispatch_model(model, segs, debug=debug)
            break
        except Exception as e:
            print(e)
            last_err = str(e)
            if attempt == MAX_RETRIES:
                return ds, False, {
                    "id":          rec.get("id") or rec.get("_id"),
                    "dataset":     ds,
                    "length_bin":  length_bucket_tag(ds, rec),
                    "qset":        question_tag(ds, rec),
                    "correct":     False,
                    "mode":        mode,
                    "error":       f"model:{last_err}",
                    "ground_truth":rec.get("ground_truth"),
                }
            time.sleep(RETRY_BACKOFF * attempt * attempt)

    evaluation_error: str | None = None
    try:
        correct = _grade(rec, out, cot)
    except Exception as e:
        correct = False
        evaluation_error = f"eval:{e}"

    len_tag = length_bucket_tag(ds, rec)            # e.g. '2k'
    q_tag   = question_tag(ds, rec)                 # e.g. 'qa4'

    if debug:
        print("─" * 60, flush=True)
        print(f"[{ds}] len={len_tag}  q={q_tag}", flush=True)
        print("[prompt]", str(segs[0])[:300].replace("\n", " ") + "...",
              flush=True)
        print("[output]", out.replace("\n", " "), flush=True)
        print("[actual answer]", rec.get("ground_truth"), flush=True)
        print("[grade]", "✔" if correct else "✘", flush=True)
        if evaluation_error:
            print("[error]", evaluation_error, flush=True)

    detail = {
        "id":           rec.get("id") or rec.get("_id"),
        "dataset":      ds,
        "length_bin":   len_tag,
        "qset":         q_tag,
        "correct":      correct,
        "mode":         mode,
        "ground_truth": rec.get("ground_truth"),
        "prompt_sent":  prompt_sent,
        "model_output": out,
        "lb_score":     rec.get("lb_score", None),
        "input_tokens":  tokens_in,
        "output_tokens": tokens_out,
    }
    if evaluation_error:
        detail["error"] = evaluation_error

    return ds, correct, detail

def run_mode(entries, mode, model, workers, debug, cot = False):
    args = [(r, mode, model, debug, cot) for r in entries
            if r["dataset"] in PROMPT_BUILDERS]

    ds_stats    = defaultdict(lambda: Counter(ok=0, tot=0))        # dataset
    len_stats   = defaultdict(lambda: Counter(ok=0, tot=0))        # (ds,len)
    len_q_stats = defaultdict(lambda: Counter(ok=0, tot=0))        # (ds,len,q)

    details: List[Dict[str, Any]] = []
    errors_counter = Counter()
    bar = tqdm(total=len(args), ncols=90, desc=f"{mode} prompts", leave=False)

    def _tally(ds: str, ok: bool, len_tag: str | None, q_tag: str | None):
        ds_stats[ds]["tot"]  += 1
        ds_stats[ds]["ok"]   += ok
        if len_tag:
            len_stats[(ds, len_tag)]["tot"] += 1
            len_stats[(ds, len_tag)]["ok"]  += ok
        if len_tag and q_tag:
            len_q_stats[(ds, len_tag, q_tag)]["tot"] += 1
            len_q_stats[(ds, len_tag, q_tag)]["ok"]  += ok

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(worker, a): a for a in args}
            for fut in as_completed(futs):
                ds, ok, det = fut.result()
                ok_int = int(ok)
                if "error" in det:
                    errors_counter[ds] += 1
                details.append(det)
                _tally(ds, ok_int, det.get("length_bin"), det.get("qset"))
                bar.update()
    else:
        for a in args:
            ds, ok, det = worker(a)
            ok_int = int(ok)
            if "error" in det:
                errors_counter[ds] += 1
            details.append(det)
            _tally(ds, ok_int, det.get("length_bin"), det.get("qset"))
            bar.update()

    bar.close()

    acc = {ds: ds_stats[ds]["ok"] / ds_stats[ds]["tot"] for ds in ds_stats}

    bucket_acc: dict[str, dict[str, float]] = defaultdict(dict)
    for (ds, tag), c in len_stats.items():
        bucket_acc[ds][tag] = c["ok"] / c["tot"]

    q_acc: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (ds, ltag, qtag), c in len_q_stats.items():
        q_acc[ds][ltag][qtag] = c["ok"] / c["tot"]

    ds_tok   = defaultdict(lambda: {"inp": 0, "out": 0})        # per dataset
    len_tok  = defaultdict(lambda: {"inp": 0, "out": 0})        # per (ds,len)
    len_q_tok= defaultdict(lambda: {"inp": 0, "out": 0})        # per (ds,len,q)

    ds_missing   = defaultdict(bool)
    len_missing  = defaultdict(bool)
    len_q_missing= defaultdict(bool)

    for det in details:
        ds = det.get("dataset")
        ltag = det.get("length_bin")
        qtag = det.get("qset")
        ti, to = det.get("input_tokens"), det.get("output_tokens")

        if ti is None or to is None:
            ds_missing[ds] = True
            if ltag:
                len_missing[(ds, ltag)] = True
            if ltag and qtag:
                len_q_missing[(ds, ltag, qtag)] = True
            continue

        ds_tok[ds]["inp"]  += ti
        ds_tok[ds]["out"] += to

        if ltag:
            len_tok[(ds, ltag)]["inp"]  += ti
            len_tok[(ds, ltag)]["out"] += to
        if ltag and qtag:
            len_q_tok[(ds, ltag, qtag)]["inp"]  += ti
            len_q_tok[(ds, ltag, qtag)]["out"] += to

    pricing = _model_pricing(model)
    usage: Dict[str, Any] = {}

    for ds in ds_stats:
        if ds_missing[ds]:
            usage[ds] = {
                "input_tokens": None,
                "output_tokens": None,
                "price_usd": None,
                "length_bins": {},
                "question_bins": {},
            }
            continue

        inp_cnt = ds_tok[ds]["inp"]
        out_cnt = ds_tok[ds]["out"]
        price = None
        if pricing is not None:
            price = inp_cnt * pricing["in"] + out_cnt * pricing["out"]

        usage[ds] = {
            "input_tokens": inp_cnt,
            "output_tokens": out_cnt,
            "price_usd": price,
            "length_bins": {},
            "question_bins": {},
        }

    # Fill length_bins and question_bins
    for (ds, ltag), c in len_tok.items():
        if len_missing[(ds, ltag)]:
            tok_dict = {"input_tokens": None, "output_tokens": None, "price_usd": None}
        else:
            inp_cnt = c["inp"]
            out_cnt = c["out"]
            price = None if pricing is None else inp_cnt * pricing["in"] + out_cnt * pricing["out"]
            tok_dict = {"input_tokens": inp_cnt, "output_tokens": out_cnt, "price_usd": price}

        usage.setdefault(ds, {"length_bins": {}, "question_bins": {}})
        usage[ds].setdefault("length_bins", {})[ltag] = tok_dict

    for (ds, ltag, qtag), c in len_q_tok.items():
        if len_q_missing[(ds, ltag, qtag)]:
            tok_dict = {"input_tokens": None, "output_tokens": None, "price_usd": None}
        else:
            inp_cnt = c["inp"]
            out_cnt = c["out"]
            price = None if pricing is None else inp_cnt * pricing["in"] + out_cnt * pricing["out"]
            tok_dict = {"input_tokens": inp_cnt, "output_tokens": out_cnt, "price_usd": price}

        usage.setdefault(ds, {"question_bins": {}})
        usage[ds].setdefault("question_bins", {}).setdefault(ltag, {})[qtag] = tok_dict

    # Collect error detail list (for separate file)
    error_details = [d for d in details if "error" in d]

    return acc, bucket_acc, q_acc, usage, details, errors_counter, error_details


def sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    """Return hex SHA‑256 of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dict_delta(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """Return *b - a* for keys in either dict (missing → None)."""
    out: Dict[str, float] = {}
    for k in set(a) | set(b):
        if k in a and k in b:
            out[k] = b[k] - a[k]
    return out



def usage_delta(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Compute delta of total token counts & price between *a* and *b* usage dicts.

    Returns dict {{'input_tokens': Δin, 'output_tokens': Δout, 'price_usd': Δprice}}
    if both sides have numeric values.  Otherwise returns None.
    """
    if not a or not b:
        return None
    if any(a.get(k) is None or b.get(k) is None for k in ("input_tokens", "output_tokens")):
        return None
    delta_in  = b["input_tokens"]  - a["input_tokens"]
    delta_out = b["output_tokens"] - a["output_tokens"]
    price_a = a.get("price_usd")
    price_b = b.get("price_usd")
    delta_price = None if price_a is None or price_b is None else price_b - price_a
    return {
        "input_tokens": delta_in,
        "output_tokens": delta_out,
        "price_usd": delta_price,
    }






ALL_DATASETS: List[str] = sorted(list(PROMPT_BUILDERS.keys()))


def parse_csv_option(opt: str | None) -> set[str]:
    """Return a clean set from a comma- and/or space-separated option string."""
    if not opt:
        return set()
    # split on comma or whitespace then strip empties
    items = re.split(r"[,\s]+", opt.strip())
    return {x for x in (i.strip() for i in items) if x}

def archive_existing_results(outdir: Path) -> Path | None:
    """Move existing JSONs into a new run_{x} folder. Return that folder or None."""
    json_files = list(outdir.glob("*.json"))
    if not json_files:
        return None
    run_num = 0
    while (outdir / f"run_{run_num}").exists():
        run_num += 1
    archive_dir = outdir / f"run_{run_num}"
    archive_dir.mkdir()
    for f in json_files:
        shutil.move(str(f), archive_dir / f.name)
    print(f"[archive] moved {len(json_files)} files → {archive_dir}")
    return archive_dir



def build_overview_from_datasets(outdir: Path,
                                 model: str,
                                 meta_hash: str,
                                 datasets_ran: list[str]) -> None:
    """Recompute an overview_*.json purely from dataset-level JSONs."""
    datasets: Dict[str, Any] = {}
    text_accs, mm_accs = [], []

    for f in outdir.glob("dataset-*.json"):
        d = json.load(f.open())
        ds = d["dataset"]
        datasets[ds] = d["stats"]
        txt = d["stats"]["text"]["overall_accuracy"] if d["stats"]["text"] else None
        mm  = d["stats"]["multimodal"]["overall_accuracy"] if d["stats"]["multimodal"] else None
        if txt is not None:
            text_accs.append(txt)
        if mm is not None:
            mm_accs.append(mm)

    def _mean(xs: list[float]) -> float | None:
        return statistics.mean(xs) if xs else None

    overview = {
        "model": model,
        "meta_sha256": meta_hash,
        "datasets": datasets,
        "overall_text_accuracy": _mean(text_accs),
        "overall_mm_accuracy":   _mean(mm_accs),
        "overall_delta": None if not text_accs or not mm_accs
                         else _mean(mm_accs) - _mean(text_accs),
        "datasets_ran": sorted(datasets_ran),
    }

    out_f = outdir / f"overview_{model.replace('/', '_')}.json"
    json.dump(overview, out_f.open("w"), indent=2)
    print(f"[saved] {out_f}")



def save_mode_file(outdir: Path, mode: str, model: str, meta_hash: str,
                   acc, bucket_acc, q_acc, usage, details, errors, cot: str = "default"):
    out = dict(model=model,
               mode=mode,
               meta_sha256=meta_hash,
               accuracy=acc,
               length_accuracy=bucket_acc,
               question_accuracy=q_acc,
               errors=errors,
               usage=usage,
               results=details)
    f = outdir / f"{mode}_{model.replace('/', '_')}_cot{cot}.json"
    json.dump(out, f.open("w"), indent=2)
    print(f"[saved] {f}")


    lite_out = dict(model=model,
                    mode=mode,
                    meta_sha256=meta_hash,
                    accuracy=acc,
                    length_accuracy=bucket_acc,
                    question_accuracy=q_acc,
                    errors=errors,
                    usage=usage)
    f_lite = outdir / f"{mode}_{model.replace('/', '_')}_metrics_cot{cot}.json"
    json.dump(lite_out, f_lite.open("w"), indent=2)
    print(f"[saved] {f_lite}")


def save_dataset_file(outdir: Path, ds: str, model: str, meta_hash: str,
                      text_stats: dict | None, mm_stats: dict | None,
                      delta: dict | None,
                      text_usage: dict | None, mm_usage: dict | None,
                      usage_delta: dict | None,
                      *, cot: str = "default"):
    payload = {
        "model": model,
        "dataset": ds,
        "meta_sha256": meta_hash,
        "stats": {"text": text_stats, "multimodal": mm_stats, "delta": delta},
        "usage": {"text": text_usage, "multimodal": mm_usage, "delta": usage_delta},
    }
    f = outdir / f"dataset-{ds}_{model.replace('/', '_')}_cot{cot}.json"
    json.dump(payload, f.open("w"), indent=2)
    print(f"[saved] {f}")

def save_overview(outdir: Path, model: str, meta_hash: str,
                  overview: dict, cot: str = "default"):
    f = outdir / f"overview_{model.replace('/', '_')}_cot{cot}.json"
    json.dump({"model": model, "meta_sha256": meta_hash, **overview},
              f.open("w"), indent=2)
    print(f"[saved] {f}")

def save_error_file(outdir: Path, mode: str, model: str, meta_hash: str,
                     error_details: List[Dict[str, Any]], cot: str = "default"):
    """Save detailed error list for a particular mode into its own file."""
    if not error_details:
        # Nothing to save – avoid empty file clutter
        return
    payload = {
        "model": model,
        "mode": mode,
        "meta_sha256": meta_hash,
        "errors": error_details,
    }
    f = outdir / f"errors_{mode}_{model.replace('/', '_')}_cot{cot}.json"
    json.dump(payload, f.open("w"), indent=2)
    print(f"[saved] {f}")


def main(
    meta: Path = typer.Argument(..., exists=True, readable=True,
                                help="Path to metadata.json"),
    model: str = typer.Option(..., "--model", "-m", help="Model identifier"),
    mode: str  = typer.Option("text", "--mode",
                              help="text · multimodal · all"),
    only_run: str | None = typer.Option(None, "--only-run",
        help="Comma/space-sep list of datasets to run (others kept)"),
    no_run:  str | None = typer.Option(None, "--no-run",
        help="Comma/space-sep list of datasets to skip"),
    cot: str | None = typer.Option(
        None,
        "--cot",
        help="Chain-of-thought: true, false, default, or revert (opposite of default)."
    ),
    skip_nolima: bool = typer.Option(False, "--skip-nolima",
                                     hidden=True),
    skip_longbench: bool = typer.Option(False, "--skip-longbench",
                                        hidden=True),
    skip_babilong: bool = typer.Option(False, "--skip-babilong",
                                       hidden=True),
    skip_mmlupro: bool = typer.Option(False, "--skip-mmlupro",
                                      hidden=True),
    skip_mmluredux: bool = typer.Option(False, "--skip-mmluredux",
                                        hidden=True),
    skip_boolq: bool = typer.Option(False, "--skip-boolq",
                                    hidden=True),
    skip_gpqa: bool = typer.Option(False, "--skip-gpqa",
                                   hidden=True),
    workers: int = typer.Option(max(os.cpu_count() // 2, 1), "--workers", "-w"),
    debug:   bool = typer.Option(False, "--debug", "-v"),
    outdir:  Path = typer.Option(Path("readbench_results"), "--outdir"),
) -> None:

    sel_only = parse_csv_option(only_run)
    print(sel_only)
    sel_skip = parse_csv_option(no_run)

    if skip_nolima:     sel_skip.add("nolima")
    if skip_longbench:  sel_skip.add("longbench")
    if skip_babilong:   sel_skip.add("babilong")
    if skip_mmlupro:    sel_skip.add("mmlu_pro")
    if skip_mmluredux:  sel_skip.add("mmlu_redux")
    if skip_boolq:      sel_skip.add("boolq")
    if skip_gpqa:       sel_skip.add("gpqa")

    if sel_only:
        run_set = sel_only - sel_skip
    else:
        run_set = set(ALL_DATASETS) - sel_skip
    print(run_set)

    if debug:
        print(f"[debug] datasets to run: {sorted(run_set)}")

    if "rendered_images_ft12_" in meta.name:
        ppi_match = meta.name.split("rendered_images_ft12_")[1].split("ppi")[0]
        ppi = ppi_match if ppi_match.isdigit() else "93"
    elif "rendered_images_ft12-" in meta.name:
        ppi = "93"
    else:
        ppi = "unknown"
    
    split_map = {"nano": "nano", "extended": "extended"}
    split = "standard"  
    print(meta.name)
    for k, v in split_map.items():
        if k in meta.name:
            split = v
            break

    if cot is None:
        cot = "default"

    outdir = outdir / model / Path(ppi) / split / f'cot{cot}'
    outdir.mkdir(parents=True, exist_ok=True)

    archive_existing_results(outdir)

    meta_hash = sha256_file(meta)
    entries: list[dict] = json.load(meta.open())
    entries = [r for r in entries if r["dataset"] in run_set]

    if debug and workers > 1:
        print("[debug] forcing workers=1 so prints stay ordered")
        workers = 1

    random.shuffle(entries)

    modes_run = ["text", "multimodal"] if mode == "all" else [mode]
    results_by_mode: Dict[str, Dict[str, Any]] = {}
    errors_by_mode: Dict[str, Dict[str, int]] = {}

    for m in modes_run:
        acc, bucket_acc, q_acc, usage, det, err_cnt, err_details = \
            run_mode(entries, m, model, workers, debug, cot)

        save_mode_file(outdir, m, model, meta_hash,
                       acc, bucket_acc, q_acc, usage, det, err_cnt, cot)
        save_error_file(outdir, m, model, meta_hash, err_details, cot)

        results_by_mode[m] = dict(acc=acc, bucket=bucket_acc,
                                  q=q_acc, usage=usage)
        errors_by_mode[m] = err_cnt

    evaluated_datasets = set()
    if set(modes_run) == {"text", "multimodal"}:
        text_r = results_by_mode["text"]
        mm_r   = results_by_mode["multimodal"]
        datasets = sorted(set(text_r["acc"]) | set(mm_r["acc"]))
        for ds in datasets:
            evaluated_datasets.add(ds)
            save_dataset_file(
                outdir, ds, model, meta_hash,
                {"overall_accuracy": text_r["acc"].get(ds),
                 "length_accuracy":  text_r["bucket"].get(ds, {}),
                 "question_accuracy":text_r["q"].get(ds, {})},
                {"overall_accuracy": mm_r["acc"].get(ds),
                 "length_accuracy":  mm_r["bucket"].get(ds, {}),
                 "question_accuracy":mm_r["q"].get(ds, {})},
                {"overall_accuracy": (mm_r["acc"].get(ds) -
                                      text_r["acc"].get(ds))
                                     if (text_r["acc"].get(ds) is not None and
                                         mm_r["acc"].get(ds) is not None) else None,
                 "length_accuracy":  dict_delta(text_r["bucket"].get(ds, {}),
                                                mm_r["bucket"].get(ds, {})),
                 "question_accuracy":{}},   # fine-grained delta omitted for brevity
                text_r["usage"].get(ds, {}),
                mm_r["usage"].get(ds, {}),
                usage_delta(text_r["usage"].get(ds, {}),
                            mm_r["usage"].get(ds, {})),
                cot=cot,
            )
    else:
        m = modes_run[0]
        rs = results_by_mode[m]
        for ds in sorted(rs["acc"]):
            evaluated_datasets.add(ds)
            save_dataset_file(
                outdir, ds, model, meta_hash,
                {"overall_accuracy": rs["acc"][ds],
                 "length_accuracy": rs["bucket"].get(ds, {}),
                 "question_accuracy": rs["q"].get(ds, {})},
                None, None,
                rs["usage"].get(ds, {}),
                None,
                usage_delta(rs["usage"].get(ds, {}), {}),
                cot=cot
            )

    build_overview_from_datasets(
        outdir, model, meta_hash,
        datasets_ran=list(evaluated_datasets)
    )

    print("\nDone.")

if __name__ == "__main__":
    typer.run(main)