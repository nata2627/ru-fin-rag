#!/usr/bin/env python3
"""Tasks 12.16 and 12.18: the candidate's lists on dev and test.

The kernel reads the candidate weights from the output of the training kernel
(decision 34: live10x2, best dev step), so the weights are not uploaded again.
The dataset holds questions, hybrid top-50 and fragment texts, test labels stay here.

    python3 scripts/test12.py pack
    kaggle datasets create -p data/interim/live12/kaggle_test12/dataset
    kaggle kernels push -p data/interim/live12/kaggle_test12/kernel
    python3 scripts/top10_12.py --run dev=data/kaggle_out/ru-fin-test12-k/dev_runs.jsonl \\
        --run test=data/kaggle_out/ru-fin-test12-k/test_runs.jsonl
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from live12_runs import KAGGLE_USER, TOP, read_jsonl, write_jsonl  # noqa: E402
from train12 import KERNELS  # noqa: E402
from trunc12 import load_run  # noqa: E402

from rufin import punkt as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
PKG = os.path.join(ROOT, "data", "interim", "live12", "kaggle_test12")
SLUG = "ru-fin-test12"
CANDIDATE = KERNELS["live10x2"]
CODE = "rerank_test12.py"


def pack() -> None:
    rows = read_jsonl(os.path.join(QDIR, "live12.jsonl"))
    ds = os.path.join(PKG, "dataset")
    shutil.rmtree(PKG, ignore_errors=True)
    os.makedirs(ds)
    need = set()
    for split in ("dev", "test"):
        qs = [q for q in rows if q["split"] == split]
        hybrid = load_run("hybrid", {q["query_id"] for q in qs})
        write_jsonl(os.path.join(ds, f"{split}_queries.jsonl"),
                    [{"query_id": q["query_id"], "text": q["text"]} for q in qs])
        write_jsonl(os.path.join(ds, f"{split}_hybrid.jsonl"),
                    [{"query_id": q, "ranked": r[:TOP]} for q, r in sorted(hybrid.items())])
        need |= {c for r in hybrid.values() for c in r[:TOP]}
        print(f"{split}: {len(qs)} questions")
    dev_ids = {q["query_id"] for q in rows if q["split"] == "dev"}
    units = P.read_units(os.path.join(QDIR, "qrels_live12_punkt.tsv"))
    P.write_units(os.path.join(ds, "dev_punkt.tsv"), {q: units[q] for q in dev_ids})
    shutil.copy(os.path.join(ROOT, "src", "rufin", "punkt.py"), ds)
    n = 0
    with gzip.open(os.path.join(ds, "chunks.jsonl.gz"), "wt", encoding="utf-8") as f:
        for c in read_jsonl(os.path.join(ROOT, "data", "chunks", "base.jsonl")):
            if c["chunk_id"] in need:
                f.write(json.dumps({"chunk_id": c["chunk_id"], "text": c["text"]}, ensure_ascii=False) + "\n")
                n += 1
    if n != len(need):
        raise SystemExit(f"fragments missing from data/chunks/base.jsonl: {len(need) - n}")
    with open(os.path.join(ds, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"title": SLUG, "id": f"{KAGGLE_USER}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}, f, indent=1)
    kern = os.path.join(PKG, "kernel")
    os.makedirs(kern)
    shutil.copy(os.path.join(ROOT, "kaggle", CODE), kern)
    with open(os.path.join(kern, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"id": f"{KAGGLE_USER}/{SLUG}-k", "title": f"{SLUG}-k".replace("-", " "),
                   "code_file": CODE, "language": "python", "kernel_type": "script",
                   "is_private": True, "enable_gpu": True, "enable_tpu": False,
                   "enable_internet": False, "machine_shape": "",
                   "dataset_sources": [f"{KAGGLE_USER}/{SLUG}"],
                   "kernel_sources": [f"{KAGGLE_USER}/{CANDIDATE}"],
                   "competition_sources": [], "model_sources": []}, f, indent=1)
    print(f"package {os.path.relpath(PKG, ROOT)}: fragments {n}, candidate {CANDIDATE}")


if __name__ == "__main__":
    pack()
