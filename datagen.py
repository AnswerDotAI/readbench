from __future__ import annotations

import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import math
import random
import typer
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
from PIL import Image, ImageDraw, ImageFont

DEFAULT_PPI = 92.9
BASE_FONT_PT = 12  


def get_font_and_dimensions(ppi: float):
    page_w = int(8.27 * ppi)
    page_h = int(11.69 * ppi)
    margin = int(ppi * 0.25)

    font_px = max(1, int(BASE_FONT_PT * ppi / 72))

    fallback = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    mac_paths = [
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    try:
        font = ImageFont.truetype("arial.ttf", font_px)
    except OSError:
        path = next((p for p in mac_paths if os.path.exists(p)), fallback)
        font = ImageFont.truetype(path, font_px)

    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    line_spacing = int(line_h * 0.2)

    return font, page_w, page_h, margin, line_h, line_spacing


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> List[str]:
    lines: List[str] = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        cur = ""
        for word in para.split():
            trial = (cur + ' ' + word).strip()
            if font.getlength(trial) <= max_w:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
    return lines


def _render(text: str, out_dir: Path, base: str, ppi: float) -> List[str]:
    font, page_w, page_h, margin, line_h, line_spacing = get_font_and_dimensions(ppi)
    max_w = page_w - 2 * margin
    lines = _wrap(text, font, max_w)

    y = margin
    page_idx = 0
    img = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(img)
    paths: List[str] = []
    max_line_w = 0.0

    def _save(im: Image.Image, last: bool, idx: int, content_y: int):
        crop_h = page_h if not last else min(page_h, content_y + margin)
        content_end_x = margin + min(max_line_w, max_w)
        crop_w = min(page_w, int(content_end_x + 4 * margin))
        im_crop = im.crop((0, 0, crop_w, crop_h))
        fp = out_dir / f"{base}_page{idx}.png"
        im_crop.save(fp, dpi=(int(ppi), int(ppi)), optimize=True, compress_level=3)
        paths.append(str(fp))

    for line in (*lines, "__END__"):
        if line == "__END__":
            _save(img, last=True, idx=page_idx, content_y=y)
            break
        if y + line_h > page_h - margin:
            _save(img, last=False, idx=page_idx, content_y=y)
            page_idx += 1
            img = Image.new("RGB", (page_w, page_h), "white")
            draw = ImageDraw.Draw(img)
            y = margin
            max_line_w = 0.0
        if "NEWLINE_CHAR" in line:
            parts = line.split("NEWLINE_CHAR")
            for sub in parts:
                if sub:
                    draw.text((margin, y), sub, fill="black", font=font)
                    w = font.getlength(sub)
                    if w > max_line_w:
                        max_line_w = w
                y += line_h + line_spacing
        else:
            draw.text((margin, y), line, fill="black", font=font)
            w = font.getlength(line)
            if w > max_line_w:
                max_line_w = w
            y += line_h + line_spacing

    return paths


def get_root_dir(ppi: float) -> Path:
    if abs(ppi - DEFAULT_PPI) < 1e-2:
        root = Path("rendered_images_ft12")
    else:
        root = Path(f"rendered_images_ft12_{int(ppi)}ppi")
    root.mkdir(exist_ok=True)
    return root

###############################################################################

def _mmlu_worker(args):
    idx, ex, img_dir, txt_dir, ppi = args
    qid = ex.get("question_id", idx)

    opts_txt = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(ex["options"]))

    imgs = _render(opts_txt, img_dir, str(qid), ppi)

    # ---------- write plain text ----------
    txt_dir.mkdir(exist_ok=True)
    txt_fp = txt_dir / f"{qid}.txt"
    txt_fp.write_text(opts_txt)

    ans_idx = int(ex.get("answer_index", ord(ex["answer"].strip()) - 65))
    return {
        "id": f"mmlu_pro_{qid}",
        "dataset": "mmlu_pro",
        "split": "test",
        "images": imgs,
        "text_file": str(txt_fp),
        "prompt_text": ex["question"],
        "ground_truth": chr(65 + ans_idx),
        "choices": ex["options"],
        "category": ex["category"],
    }


def process_mmlu_pro(split: str = "test", workers: int | None = None, ppi: float = DEFAULT_PPI):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    
    root = get_root_dir(ppi)
    
    img_dir = root / "mmlu_pro"
    txt_dir = root / "mmlu_pro_text"
    for d in (img_dir, txt_dir):
        d.mkdir(parents=True, exist_ok=True)

    workers = workers or max(1, mp.cpu_count() // 2)
    tasks = ((i, ex, img_dir, txt_dir, ppi) for i, ex in enumerate(ds))
    meta: List[Dict] = []

    print(f"MMLU‑Pro: rendering with {workers} workers at {ppi} PPI …")
    with ProcessPoolExecutor(workers) as pool:
        for rec in tqdm(pool.map(_mmlu_worker, tasks, chunksize=16), total=len(ds)):
            meta.append(rec)

    (img_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


LONG_BENCH_NAME = "THUDM/LongBench"
TOKEN_BINS = [
    (0, 512, "0-512"),
    (512, 1024, "512-1024"),
    (1024, 2048, "1024-2048"),
    (2048, 4096, "2048-4096"),
    (4096, 8192, "4096-8192"),
    (8192, 16384, "8192-16k"),
    (16384, float("inf"), "DISMISS"),
]

def _bin_len(n: int) -> str:
    for lo, hi, name in TOKEN_BINS:
        if lo <= n < hi:
            return name
    return "unknown"


def _long_worker(args):
    idx, ex, subset, img_dir, txt_dir, ppi = args
    base = f"{subset}_{idx}"
    bin_name = _bin_len(ex["length"])
    if bin_name == "DISMISS" or ex["language"] != "en":
        return None

    imgs = _render(ex["context"], img_dir, base, ppi)

    # ---------- write plain text ----------
    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_fp = txt_dir / f"{base}.txt"
    txt_fp.write_text(ex["context"])

    return {
        "_id": ex["_id"],
        "local_id": base,
        "dataset": "longbench",
        "subset": subset,
        "length": ex["length"],
        "length_bin": bin_name,
        "images": imgs,
        "text_file": str(txt_fp),
        "prompt_text": ex["input"],
        "ground_truth": ex["answers"],
    }


def process_longbench(workers: int | None = None, ppi: float = DEFAULT_PPI):
    # subsets = get_dataset_config_names(LONG_BENCH_NAME)
    subsets = ["2wikimqa_e",
    "hotpotqa_e",
    "multifieldqa_en_e",
    "narrativeqa",
    "passage_count_e",
    "triviaqa_e",
    ]

    workers = workers or max(1, mp.cpu_count() // 2)
    
    root = get_root_dir(ppi)

    for cfg in subsets:
        ds = load_dataset(LONG_BENCH_NAME, cfg, split="test")

        img_dir = root / f"longbench_{cfg}"
        txt_dir = root / "longbench_text" / cfg
        for d in (img_dir, txt_dir):
            d.mkdir(parents=True, exist_ok=True)

        tasks = ((i, ex, cfg, img_dir, txt_dir, ppi) for i, ex in enumerate(ds))
        meta: List[Dict] = []

        print(f"LongBench/{cfg}: {len(ds)} ex • {workers} workers at {ppi} PPI …")
        with ProcessPoolExecutor(workers) as pool:
            for rec in tqdm(pool.map(_long_worker, tasks, chunksize=16), total=len(ds)):
                if rec is None or rec["length_bin"] == "DISMISS":
                    continue
                meta.append(rec)

        (img_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))





BABILONG_LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k"]
BABILONG_QSETS = ["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10",]
BABILONG_NAME = "RMT-team/babilong"


def _babilong_worker(args):
    idx, ex, length_key, qset, img_dir, txt_dir, ppi = args
    base = f"{length_key}_{qset}_{idx}"

    # Render the story / haystack ("input")
    imgs = _render(ex["input"], img_dir, base, ppi)

    # Write plain text
    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_fp = txt_dir / f"{base}.txt"
    txt_fp.write_text(ex["input"])

    return {
        "id": f"babilong_{base}",
        "dataset": "babilong",
        "subset": qset,
        "length_bin": length_key,
        "images": imgs,
        "text_file": str(txt_fp),
        "prompt_text": ex["question"],
        "ground_truth": ex["target"],
    }


def process_babilong_niah(workers: int | None = None, ppi: float = DEFAULT_PPI):
    """Process Babilong NIAH for the chosen question sets and length buckets."""
    workers = workers or max(1, mp.cpu_count() // 2)
    root = get_root_dir(ppi)

    images_root = root
    text_root = root / "babilong_text"
    text_root.mkdir(parents=True, exist_ok=True)

    meta: List[Dict] = []

    total_tasks = 0
    task_iter = []
    for length_key in BABILONG_LENGTHS:
        for qset in BABILONG_QSETS:
            ds = load_dataset(BABILONG_NAME, length_key, split=qset)
            img_dir = images_root / f"babilong_{length_key}_{qset}"
            txt_dir = text_root / f"{length_key}_{qset}"
            img_dir.mkdir(parents=True, exist_ok=True)
            txt_dir.mkdir(parents=True, exist_ok=True)
            for idx, ex in enumerate(ds):
                task_iter.append((idx, ex, length_key, qset, img_dir, txt_dir, ppi))
            total_tasks += len(ds)

    print(
        f"BabiLong NIAH: {total_tasks} examples • {workers} workers at {ppi} PPI …"
    )

    with ProcessPoolExecutor(workers) as pool:
        for rec in tqdm(pool.map(_babilong_worker, task_iter, chunksize=16), total=total_tasks):
            meta.append(rec)

    # One metadata file at the root of babilong images
    (root / "babilong_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )


BOOLQ_NAME = "google/boolq"

def _boolq_worker(args):
    idx, ex, split, img_dir, txt_dir, ppi = args
    base = ex.get("id") or f"{split}_{idx}"

    imgs = _render(ex["passage"], img_dir, base, ppi)

    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_fp = txt_dir / f"{base}.txt"
    txt_fp.write_text(ex["passage"])

    return {
        "id": f"boolq_{base}",
        "dataset": "boolq",
        "split": split,
        "images": imgs,
        "text_file": str(txt_fp),
        "prompt_text": ex["question"],
        "ground_truth": bool(ex["answer"]),
    }


def process_boolq(split: str = "validation", workers: int | None = None,
                  ppi: float = DEFAULT_PPI):
    ds = load_dataset(BOOLQ_NAME, split=split)

    root = get_root_dir(ppi)
    img_dir = root / "boolq"
    txt_dir = root / "boolq_text"
    for d in (img_dir, txt_dir):
        d.mkdir(parents=True, exist_ok=True)

    workers = workers or max(1, mp.cpu_count() // 2)
    tasks = ((i, ex, split, img_dir, txt_dir, ppi) for i, ex in enumerate(ds))
    meta: List[Dict] = []

    print(f"BoolQ/{split}: {len(ds)} ex • {workers} workers at {ppi} PPI …")
    with ProcessPoolExecutor(workers) as pool:
        for rec in tqdm(pool.map(_boolq_worker, tasks, chunksize=16),
                        total=len(ds)):
            meta.append(rec)

    (img_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )


MMLU_REDUX_NAME = "edinburgh-dawg/mmlu-redux-2.0"


def _mmlur_worker(args):
    """Worker to render a single MMLU-Redux example."""
    idx, ex, subset, img_dir, txt_dir, ppi = args
    base = f"{subset}_{idx}"

    opts_txt = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(ex["choices"]))
    imgs = _render(opts_txt, img_dir, base, ppi)

    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_fp = txt_dir / f"{base}.txt"
    txt_fp.write_text(opts_txt)

    answer_idx = int(ex["answer"])  

    return {
        "id": f"mmlu_redux_{base}",
        "dataset": "mmlu_redux",
        "subset": subset,
        "split": "test",
        "images": imgs,
        "text_file": str(txt_fp),
        "prompt_text": ex["question"],
        "ground_truth": chr(65 + answer_idx),
        "choices": ex["choices"],
        "category": subset,
    }


def process_mmlu_redux(split: str = "test", workers: int | None = None, ppi: float = DEFAULT_PPI):
    """Render all 57 subsets of MMLU-Redux to images + plain text."""
    subsets = get_dataset_config_names(MMLU_REDUX_NAME)

    root = get_root_dir(ppi)

    workers = workers or max(1, mp.cpu_count() // 2)

    for subset in subsets:
        ds = load_dataset(MMLU_REDUX_NAME, subset, split=split)

        img_dir = root / f"mmlu_redux_{subset}"
        txt_dir = root / "mmlu_redux_text" / subset
        for d in (img_dir, txt_dir):
            d.mkdir(parents=True, exist_ok=True)

        tasks = ((i, ex, subset, img_dir, txt_dir, ppi) for i, ex in enumerate(ds))
        meta: List[Dict] = []

        print(f"MMLU-Redux/{subset}: {len(ds)} ex • {workers} workers at {ppi} PPI …")
        with ProcessPoolExecutor(workers) as pool:
            for rec in tqdm(pool.map(_mmlur_worker, tasks, chunksize=16), total=len(ds)):
                meta.append(rec)

        (img_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


GPQA_NAME   = "Idavidrein/gpqa"
GPQA_CONFIG = "gpqa_diamond"

def _gpqa_worker(args):
    idx, ex, img_dir, txt_dir, ppi = args
    qid = ex.get("Record ID", idx)

    opts      = [ex["Correct Answer"],
                 ex["Incorrect Answer 1"],
                 ex["Incorrect Answer 2"],
                 ex["Incorrect Answer 3"]]

    rng       = random.Random(hash(str(qid)) & 0xFFFFFFFF)
    order     = list(range(4));  rng.shuffle(order)
    shuffled  = [opts[i] for i in order]
    gt_letter = chr(65 + order.index(0))        # where the *correct* answer moved

    opts_txt  = "\n".join(f"{chr(65+i)}. {txt}" for i, txt in enumerate(shuffled))
    imgs      = _render(opts_txt, img_dir, str(qid), ppi)

    txt_dir.mkdir(exist_ok=True)
    txt_fp = txt_dir / f"{qid}.txt"
    txt_fp.write_text(opts_txt)

    return {
        "id":          f"gpqa_{qid}",
        "dataset":     "gpqa",
        "split":       "train",
        "images":      imgs,
        "text_file":   str(txt_fp),
        "prompt_text": ex["Question"],
        "ground_truth": gt_letter,
        "choices":     shuffled,
        "category":    ex.get("Subdomain") or ex.get("High-level domain"),
    }


def process_gpqa(workers: int | None = None,
                 ppi: float = DEFAULT_PPI,
                 split: str = "train"):
    ds = load_dataset(GPQA_NAME, GPQA_CONFIG, split=split)

    root    = get_root_dir(ppi)
    img_dir = root / "gpqa"
    txt_dir = root / "gpqa_text"
    for d in (img_dir, txt_dir):
        d.mkdir(parents=True, exist_ok=True)

    workers = workers or max(1, mp.cpu_count() // 2)
    tasks   = ((i, ex, img_dir, txt_dir, ppi) for i, ex in enumerate(ds))
    meta: List[Dict] = []

    print(f"GPQA: {len(ds)} questions • {workers} workers at {ppi} PPI …")
    with ProcessPoolExecutor(workers) as pool:
        for rec in tqdm(pool.map(_gpqa_worker, tasks, chunksize=16),
                        total=len(ds)):
            meta.append(rec)

    (img_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )


app = typer.Typer()

def parse_datasets(datasets_str: str) -> set[str]:
    """Parse comma-delimited dataset string into a set of valid dataset names."""
    if not datasets_str:
        return set()
    
    valid_datasets = {"mmlupro", "mmluredux", "longbench", "gpqa", "babilong", "boolq", "all"}
    
    requested = {name.strip().lower() for name in datasets_str.split(",")}
    
    if "all" in requested:
        return {"mmlupro", "mmluredux", "longbench", "gpqa", "babilong", "boolq"}
    
    invalid = requested - valid_datasets
    if invalid:
        print(f"❌ Invalid dataset names: {invalid}")
        print(f"Valid options: {', '.join(sorted(valid_datasets))}")
        exit(1)
    
    return requested


@app.command()
def main(
    datasets: str = typer.Option("all", "--datasets", help="Comma-separated list of datasets to generate (mmlupro,mmluredux,longbench,gpqa,babilong,boolq,all). Default: all"),
    ppi: float = typer.Option(DEFAULT_PPI, "--ppi", help="Pixels per inch (PPI) for image rendering. Default: 92.9"),
    workers: int = typer.Option(None, "--workers", "-w", help="Number of worker processes (default: half of CPU count)")
):
    to_process = parse_datasets(datasets)
    
    if not to_process:
        print("❌ No valid datasets specified.")
        return
    
    print(f"🎯 Processing datasets: {', '.join(sorted(to_process))}")
    
    if "boolq" in to_process:
        process_boolq(ppi=ppi, workers=workers) 
    if "gpqa" in to_process:
        process_gpqa(ppi=ppi, workers=workers)
    if "mmlupro" in to_process:
        process_mmlu_pro(ppi=ppi, workers=workers)
    if "babilong" in to_process:
        process_babilong_niah(ppi=ppi, workers=workers)
    if "longbench" in to_process:
        process_longbench(ppi=ppi, workers=workers)
    if "mmluredux" in to_process:
        process_mmlu_redux(ppi=ppi, workers=workers)
    
    print(f"✅ Selected dataset rendering complete at {ppi} PPI.")

if __name__ == "__main__":
    app()