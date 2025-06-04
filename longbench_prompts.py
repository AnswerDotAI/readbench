"""
ADAPTED FROM THE ORIGINAL LONGBENCH REPOSITORY UNDER MIT LICENSE
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence, Union
import json

TextOrImg = Union[str, Path]

__all__ = ["build_prompt"]


def _read_txt(p: str | Path, max_chars: int | None = None) -> str:
    try:
        txt = Path(p).read_text(errors="ignore")
    except FileNotFoundError:
        return ""
    return txt if max_chars is None else txt[:max_chars]


_TEMPLATE_PATH = Path(__file__).resolve().parent / "eval_config" / "dataset2prompt.json"
_TEMPLATE_PATH_COT = Path(__file__).resolve().parent / "eval_config" / "dataset2prompt_cot.json"
try:
    _TEMPLATES: Dict[str, str] = json.loads(_TEMPLATE_PATH.read_text())
    _TEMPLATES_COT: Dict[str, str] = json.loads(_TEMPLATE_PATH_COT.read_text())
except FileNotFoundError:
    _TEMPLATES = {}



def _fill_template(tmpl: str, *, context: str, question: str) -> str:
    if "{context}" in tmpl:
        tmpl = tmpl.replace("{context}", context)
    if "{input}" in tmpl:
        tmpl = tmpl.replace("{input}", question)
    return tmpl

def build_prompt(rec: Dict, *, mode: str = "text", cot: bool = False) -> List[TextOrImg]:
    if mode not in {"text", "multimodal"}:
        raise ValueError("mode must be 'text' or 'multimodal'")

    question = (rec.get("prompt_text") or rec.get("question") or "").strip()

    subset_raw: str = rec.get("subset") or rec.get("lb_subset") \
                   or rec.get("dataset_name") or ""
    subset = subset_raw.replace("_e", "")

    if cot:
        template = _TEMPLATES_COT.get(subset)
    else:
        template = _TEMPLATES.get(subset)

    txt_path = rec.get("text_file")
    context_text = _read_txt(txt_path) if txt_path else ""

    segs: List[TextOrImg] = []

    if template:
        if mode == "text":
            prompt_str = _fill_template(template, context=context_text,
                                        question=question)
            return [prompt_str]

        imgs: Sequence[TextOrImg] = rec.get("images") or []
        if not imgs:
            raise KeyError("No 'images' available for multimodal prompt")

        if "{context}" in template:
            before_ctx, after_ctx = template.split("{context}", 1)
            before_ctx = before_ctx.replace("{input}", question)
            after_ctx  = after_ctx.replace("{input}", question)
            segs.extend([before_ctx])
            segs.extend(imgs)
            segs.append(after_ctx)
        else:
            segs.extend([template.replace("{input}", question)])
            segs.extend(imgs)
        return segs

    if mode == "text":
        if not context_text:
            raise KeyError("No 'text_file' available for text mode prompt")
        segs.append(
            "You will answer a question based on the following document "
            "excerpt. Read it carefully and provide a concise, correct "
            "answer.\n\n" + context_text + "\n\nQuestion: " + question + "\n\n"
            "Return only the answer – no extra explanation.")
        return segs

    imgs: Sequence[TextOrImg] = rec.get("images") or []
    if not imgs:
        raise KeyError("No 'images' available for multimodal prompt")
    segs.append(
        "You will answer a question based on the attached document "
        "pages. Review the pages, then answer concisely." if question else
        "Please review the attached document pages.")
    segs.extend(imgs)
    if question:
        segs.append(
            "Question: " + question + "\n\nReturn only the answer – no extra "
            "explanation.")
    return segs
