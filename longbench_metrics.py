"""
TAKEN FROM THE ORIGINAL LONGBENCH REPOSITORY UNDER MIT LICENSE
"""
from __future__ import annotations
import re, string
from collections import Counter
from typing import List, Mapping, Callable, Sequence

import jieba                       # Chinese word-segmentation
from fuzzywuzzy import fuzz        # simple token-sort ratio
from rouge import Rouge            # ROUGE-L scorer

__all__ = [
    "dataset2metric",
    "qa_f1_score", "qa_f1_zh_score", "rouge_score", "rouge_zh_score",
    "classification_score", "retrieval_score", "retrieval_zh_score",
    "count_score", "code_sim_score",
]

def _normalize_answer_en(s: str) -> str:
    def rm_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def rm_punc(t):     return "".join(ch for ch in t if ch not in string.punctuation)
    def white(t):       return " ".join(t.split())
    return white(rm_articles(rm_punc(s.lower())))

def _normalize_answer_zh(s: str) -> str:
    cn_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    all_punc = set(string.punctuation + cn_punc)
    def rm_punc(t):  return "".join(ch for ch in t if ch not in all_punc)
    def white(t):    return "".join(t.split())
    return white(rm_punc(s.lower()))

def _f1(pred_tokens: Sequence[str], gold_tokens: Sequence[str]) -> float:
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_tokens)
    rec  = num_same / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)


def qa_f1_score(pred: str, gold: str, **_) -> float:
    pred_tokens = _normalize_answer_en(pred).split()
    gold_tokens = _normalize_answer_en(gold).split()
    return _f1(pred_tokens, gold_tokens)


def qa_f1_zh_score(pred: str, gold: str, **_) -> float:
    pred_tokens = [ _normalize_answer_zh(t)
                    for t in jieba.cut(pred, cut_all=False) ]
    gold_tokens = [ _normalize_answer_zh(t)
                    for t in jieba.cut(gold, cut_all=False) ]
    pred_tokens  = [t for t in pred_tokens  if t]
    gold_tokens  = [t for t in gold_tokens  if t]
    return _f1(pred_tokens, gold_tokens)


def rouge_score(pred: str, gold: str, **_) -> float:
    try:
        scores = Rouge().get_scores([pred], [gold], avg=True)
        return scores["rouge-l"]["f"]
    except Exception:
        return 0.0


def rouge_zh_score(pred: str, gold: str, **_) -> float:
    pred_seg  = " ".join(jieba.cut(pred,  cut_all=False))
    gold_seg  = " ".join(jieba.cut(gold,  cut_all=False))
    return rouge_score(pred_seg, gold_seg)


def classification_score(pred: str, gold: str, *, all_classes: List[str] = None, **__) -> float:
    em_list = [c for c in (all_classes or []) if c in pred]
    # prune super-strings
    em_list = [c for c in em_list if not any((c in g and c != g) for g in (em_list))]
    if gold in em_list:
        return 1.0 / len(em_list)
    return 0.0


def retrieval_score(pred: str, gold: str, **_) -> float:
    gt_id = re.search(r"Paragraph (\d+)", gold)
    if not gt_id:
        return 0.0
    gt_id = gt_id.group(1)
    nums  = re.findall(r"\d+", pred)
    matches = sum(1 for n in nums if n == gt_id)
    return 0.0 if not nums else matches / len(nums)


def retrieval_zh_score(pred: str, gold: str, **_) -> float:
    gt_id = re.search(r"段落(\d+)", gold)
    if not gt_id:
        return 0.0
    gt_id = gt_id.group(1)
    nums  = re.findall(r"\d+", pred)
    matches = sum(1 for n in nums if n == gt_id)
    return 0.0 if not nums else matches / len(nums)


def count_score(pred: str, gold: str, **_) -> float:
    nums = re.findall(r"\d+", pred)
    matches = sum(1 for n in nums if n == str(gold))
    return 0.0 if not nums else matches / len(nums)


def code_sim_score(pred: str, gold: str, **_) -> float:
    # heuristically take the first non-comment, non-code-fenced line
    for line in pred.lstrip("\n").split("\n"):
        if not any(tok in line for tok in ("`", "#", "//")):
            pred_line = line
            break
    else:
        pred_line = ""
    return fuzz.ratio(pred_line, gold) / 100.0


dataset2metric: Mapping[str, Callable[..., float]] = {
    "narrativeqa":           qa_f1_score,
    "qasper":                qa_f1_score,
    "multifieldqa_en":       qa_f1_score,
    "multifieldqa_zh":       qa_f1_zh_score,
    "hotpotqa":              qa_f1_score,
    "2wikimqa":              qa_f1_score,
    "musique":               qa_f1_score,
    "dureader":              rouge_zh_score,
    "gov_report":            rouge_score,
    "qmsum":                 rouge_score,
    "multi_news":            rouge_score,
    "vcsum":                 rouge_zh_score,
    "trec":                  classification_score,
    "triviaqa":              qa_f1_score,
    "samsum":                rouge_score,
    "lsht":                  classification_score,
    "passage_retrieval_en":  retrieval_score,
    "passage_count":         count_score,
    "passage_retrieval_zh":  retrieval_zh_score,
    "lcc":                   code_sim_score,
    "repobench-p":           code_sim_score,
}