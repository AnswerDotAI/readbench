from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

SEED = 42  
RNG: random.Random | None = None  # Will be initialised via _set_seed()
CAP_EXTENDED = 100
CAP_STANDARD = 35
CAP_NANO = 5
MIN_OVERALL_MINI = 490


def _set_seed(seed: int | None = None) -> None:
    global RNG
    s = seed if seed is not None else SEED
    RNG = random.Random(s)
    random.seed(s)  
    np.random.seed(s)
_set_seed()


def _sample_exact(records: List[dict], k: int | None) -> List[dict]:
    if k is None:
        return records
    return records if len(records) <= k else RNG.sample(records, k)


def _subset_key(r: dict) -> Tuple[Any, ...]:
    ds = r.get("dataset")
    if ds == "mmlu_pro":
        return ds, r["category"]
    if ds and ds.startswith("mmlu_redux"):
        return ds, r["category"]
    if ds and ds.startswith("longbench"):
        return ds, r.get("subset", ds), r.get("length_bin", None)
    if ds == "babilong":
        return ds, r["length_bin"], r["subset"]
    if ds == "gpqa":
        return ds, r["category"]
    return (ds,)


def _downsample(records: Iterable[dict], cap: int | None) -> List[dict]:
    """Down‑sample *records* so that each subset has at most CAP entries."""
    if cap is None:
        return list(records)
    buckets: Dict[Tuple[Any, ...], List[dict]] = defaultdict(list)
    for r in records:
        buckets[_subset_key(r)].append(r)
    sampled: List[dict] = []
    for key in sorted(buckets.keys()):
        recs = buckets[key]
        if key[0] == "boolq":
            boolq_cap = cap * 8
            sampled.extend(_sample_exact(recs, boolq_cap))
        elif key[0] == "gpqa":
            if MIN_OVERALL_MINI is not None and MIN_OVERALL_MINI >= 448:
                sampled.extend(recs)
            else:
                sampled.extend(_sample_exact(recs, cap))
        else:
            sampled.extend(_sample_exact(recs, cap))

    def _records_by_dataset(dataset_name: str) -> List[dict]:
        return [r for r in records if r.get("dataset") == dataset_name]

    # For BoolQ
    boolq_sampled = [r for r in sampled if r.get("dataset") == "boolq"]
    boolq_all = _records_by_dataset("boolq")
    if len(boolq_sampled) < MIN_OVERALL_MINI:
        boolq_sampled_ids = {id(r) for r in boolq_sampled}
        boolq_remaining = [r for r in boolq_all if id(r) not in boolq_sampled_ids]
        needed = MIN_OVERALL_MINI - len(boolq_sampled)
        if needed > 0 and boolq_remaining:
            to_sample = min(needed, len(boolq_remaining))
            boolq_extra = _sample_exact(boolq_remaining, to_sample)
            sampled.extend(boolq_extra)

    gpqa_sampled = [r for r in sampled if r.get("dataset") == "gpqa"]
    gpqa_all = _records_by_dataset("gpqa")
    if len(gpqa_sampled) < MIN_OVERALL_MINI:
        gpqa_sampled_ids = {id(r) for r in gpqa_sampled}
        gpqa_remaining = [r for r in gpqa_all if id(r) not in gpqa_sampled_ids]
        needed = MIN_OVERALL_MINI - len(gpqa_sampled)
        if needed > 0 and gpqa_remaining:
            to_sample = min(needed, len(gpqa_remaining))
            gpqa_extra = _sample_exact(gpqa_remaining, to_sample)
            sampled.extend(gpqa_extra)

    return sampled


def _filter_8k(records: Iterable[dict]) -> List[dict]:
    """Return records excluding entries exceeding the 8k window.

    Currently this translates to:
    • BabiLong: drop the *16k* length_bin.
    • LongBench: drop the *8192-16k* length_bin.
    """
    filtered: List[dict] = []
    for r in records:
        ds = r.get("dataset")
        length_bin = r.get("length_bin")

        lb = str(length_bin).lower()

        # Exclude BabiLong 16k bin (exact "16k" or variants such as "16k+"/"16k-32k")
        if ds == "babilong" and lb.startswith("16k"):
            continue

        # Exclude LongBench 8192-16k bin only (others are ≤ 8k).
        if ds and ds.startswith("longbench") and lb == "8192-16k":
            continue

        filtered.append(r)

    return filtered

def _filter_lb_bins(records: Iterable[dict]) -> List[dict]:
    filtered: List[dict] = []
    for r in records:
        ds = r.get("dataset")
        lb = r.get("length_bin")

        if ds and ds.startswith("longbench"):
            if lb == "0-512":
                continue
        filtered.append(r)

    return filtered


def _load_all_metadata(root: Path, include_boolq: bool = False) -> List[dict]:
    """Collect *all* ReadBench records from the render tree rooted at *root*."""
    metas: List[dict] = []

    metas += json.load((root / "mmlu_pro" / "metadata.json").open())

    for p in sorted(root.glob("mmlu_redux_*"), key=lambda x: x.name):
        if p.name.startswith("mmlu_redux_text"):
            continue
        meta_path = p / "metadata.json"
        if meta_path.exists():
            metas += json.load(meta_path.open())

    longbench_subsets = [
        "2wikimqa_e",
        "hotpotqa_e",
        "multifieldqa_en_e",
        "narrativeqa",
        "passage_count_e",
        "triviaqa_e",
    ]
    for subset in longbench_subsets:
        p = root / f"longbench_{subset}"
        if not p.exists():
            continue
        metas += json.load((p / "metadata.json").open())

    if include_boolq:
        boolq_meta_path = root / "boolq" / "metadata.json"
        if boolq_meta_path.exists():
            metas += json.load(boolq_meta_path.open())

    metas += json.load((root / "babilong_metadata.json").open())

    gpqa_meta_path = root / "gpqa" / "metadata.json"
    if gpqa_meta_path.exists():
        metas += json.load(gpqa_meta_path.open())

    return metas

def postfilter_lb_data(records: List[dict]) -> List[dict]:
    filtered: List[dict] = []
    for r in records:
        ds = r.get("dataset")
        if ds and ds.startswith("longbench"):
            subset = r.get("subset", "")
            if "multifield" in subset:
                continue
            filtered.append(r)
        else:
            filtered.append(r)
    return filtered

# ─────────────────────────────────────── Core build func ──

def build_eye_read(root: Path, cap: int | None = None, *, limit_8k: bool = False, postfilter_lb: bool = True, include_boolq: bool = False) -> List[dict]:
    """Return the ReadBench split defined by *cap* (None = full/extended)."""
    all_records = _load_all_metadata(root, include_boolq=include_boolq)
    all_records = _filter_lb_bins(all_records)

    if limit_8k:
        all_records = _filter_8k(all_records)

    downsampled = _downsample(all_records, cap)
    if postfilter_lb:
        downsampled = postfilter_lb_data(downsampled)
    return downsampled


# ───────────────────────────────────────────────────────── Writer/report ──

def _write_split(tag: str, records: List[dict], outdir: Path, prefix: str | None = None, eightk: bool = False) -> None:
    """Write *records* to *outdir* using ReadBench naming conventions.

    If *prefix* is provided, it is inserted after the "readbench" stem so that
    multiple variants (e.g. different PPIs) can coexist in the same output
    directory without collisions, e.g.::

        readbench-93ppi_metadata.json
        readbench-93ppi-mini_metadata.json
    """

    prefix_part = f"-{prefix}" if prefix else ""
    suffix = f"-{tag}" if tag else ""

    stem = "readbench_8k" if eightk else "readbench"
    fname = outdir / f"{stem}{prefix_part}{suffix}_metadata.json"

    # Persist JSON metadata
    fname.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    label_extra = "_8k" if eightk else ""
    label = f"{label_extra}{prefix_part}{suffix}" if (label_extra or prefix_part or suffix) else ""
    print(f"ReadBench{label:12}: {len(records):>6,} → {fname}")

    def _count_by_ds(coll: List[dict]) -> List[Tuple[str, int]]:
        unique_ds = {r["dataset"] for r in coll}
        return sorted(((ds, sum(1 for _ in coll if _["dataset"] == ds)) for ds in unique_ds), key=lambda x: x[0])

    for ds, n in _count_by_ds(records):
        print(f"  - {ds:<12}: {n:>6,}")


# ──────────────────────────────────────────────────────────────── CLI ────

def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build ReadBench splits from a render tree.")
    ap.add_argument("--root", type=Path, default=Path("rendered_images"), help="Render tree root (default: rendered_images)")
    ap.add_argument("--outdir", type=Path, default=Path("readbench_meta"), help="Output directory (default: readbench_meta)")
    ap.add_argument("--split", choices=["all", "extended", "standard", "nano", "custom"], default="all", help="Which split to build; 'all' builds every standard split.")
    ap.add_argument("--cap", type=int, default=None, help="Cap size for custom split (implies --split custom).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed to use (default: 42).")
    ap.add_argument("--do-all-ppi", action="store_true", help="Process all PPI variants: iterate over <root>*ppi directories and build splits for each.")
    ap.add_argument("--do-boolq", action="store_true", help="Include BoolQ dataset in the splits (default: False).")
    ap.add_argument("--no-8k", dest="limit_8k", action="store_false", help="Disable 8k-only variant: include BabiLong 16k & LongBench 8192-16k bins and do not prefix output filenames with _8k.")
    ap.set_defaults(limit_8k=True)
    args = ap.parse_args(argv)

    _set_seed(args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.cap is not None:
        args.split = "custom"

    caps = {
        "extended": CAP_EXTENDED,
        "standard": CAP_STANDARD,
        "nano": CAP_NANO,
        "custom": args.cap,
    }

    def _ppi_prefix(dir_name: str) -> str:
        """Return a short PPI identifier extracted from *dir_name*."""
        if dir_name.endswith("ppi") and "_" in dir_name:
            return dir_name.split("_")[-1]
        return dir_name

    def _build_one(root_dir: Path, tag: str):
        """Build *tag* split for *root_dir*."""
        _set_seed(args.seed)
        recs = build_eye_read(root_dir, caps[tag], limit_8k=args.limit_8k, include_boolq=args.do_boolq)
        prefix = _ppi_prefix(root_dir.name)
        _write_split(tag if tag != "standard" else "", recs, args.outdir, prefix=prefix, eightk=args.limit_8k)

    roots_to_process = []
    if args.do_all_ppi:
        base_root = args.root
        if base_root.exists() and base_root.is_dir() and base_root.name.endswith("ppi"):
            roots_to_process = [base_root]
        else:
            pattern = f"{base_root.name if base_root.name else 'rendered_images'}*ppi"
            roots_to_process = sorted(base_root.parent.glob(pattern))
        if not roots_to_process:
            print(f"[WARN] --do-all-ppi requested but no <root>*ppi directories found for pattern '{pattern}'.", file=sys.stderr)
            sys.exit(1)
    else:
        roots_to_process = [args.root]

    if args.split == "all":
        tags_to_process = ["extended", "standard", "nano"]
    else:
        tags_to_process = [args.split]

    for root_dir in roots_to_process:
        for tag in tags_to_process:
            _build_one(root_dir, tag)



if __name__ == "__main__":
    main()
