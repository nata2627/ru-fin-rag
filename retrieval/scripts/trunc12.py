#!/usr/bin/env python3
"""Task 12.14: whole question against the question cut to 256 tokens, live12 dev.

The current reranker scores letters at 0.531 against 0.689 for the other test
questions. A long letter eats the 512-token window, so the fragment gets cut.
The cut length for training is chosen here, on dev only, before training.

    python3 scripts/trunc12.py pack     # dataset and kernel for kaggle/rerank_trunc12.py, no push
    python3 scripts/trunc12.py score    # after fetch: clause NDCG@10 of both modes
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from live12_runs import KAGGLE_USER, RERANKER_INPUT, TOP, read_jsonl, write_jsonl  # noqa: E402

from rufin import metrics as M  # noqa: E402
from rufin import punkt as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
RUN_DIRS = (os.path.join(ROOT, "data", "runs"), os.path.join(ROOT, "data", "interim", "live12", "runs"))
PKG = os.path.join(ROOT, "data", "interim", "live12", "kaggle_trunc")
SLUG = "ru-fin-trunc12"
KSLUG = f"{SLUG}-k"
FETCHED = os.path.join(ROOT, "data", "kaggle_out", KSLUG, "runs")
RESULT = os.path.join(ROOT, "data", "results", "trunc12.json")
MODES = ("full", "q256")


def dev_queries() -> list[dict]:
    rows = read_jsonl(os.path.join(QDIR, "live12.jsonl"))
    return [q for q in rows if q["split"] == "dev"]


def load_run(cfg: str, ids: set[str]) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    for d in RUN_DIRS:
        for r in read_jsonl(os.path.join(d, f"base__{cfg}.jsonl")):
            if r["query_id"] in ids:
                run.setdefault(r["query_id"], r["ranked"])
    if set(run) != ids:
        raise SystemExit(f"{cfg}: rows for {len(run)} of {len(ids)} dev questions")
    return run


def pack() -> None:
    dev = dev_queries()
    ids = {q["query_id"] for q in dev}
    hybrid = load_run("hybrid", ids)
    stored = load_run("hybrid-rerank", ids)
    ds, kern = os.path.join(PKG, "dataset"), os.path.join(PKG, "kernel")
    shutil.rmtree(PKG, ignore_errors=True)
    os.makedirs(ds)
    os.makedirs(kern)
    write_jsonl(os.path.join(ds, "queries.jsonl"), [{"query_id": q["query_id"], "text": q["text"]} for q in dev])
    write_jsonl(os.path.join(ds, "base__hybrid.jsonl"),
                [{"query_id": q, "ranked": r[:TOP]} for q, r in sorted(hybrid.items())])
    write_jsonl(os.path.join(ds, "base__hybrid-rerank.jsonl"),
                [{"query_id": q, "ranked": r[:10]} for q, r in sorted(stored.items())])
    need = {c for r in hybrid.values() for c in r[:TOP]}
    n = 0
    with gzip.open(os.path.join(ds, "chunks_candidates.jsonl.gz"), "wt", encoding="utf-8") as f:
        for c in read_jsonl(os.path.join(ROOT, "data", "chunks", "base.jsonl")):
            if c["chunk_id"] in need:
                f.write(json.dumps({"chunk_id": c["chunk_id"], "text": c["text"]}, ensure_ascii=False) + "\n")
                n += 1
    if n != len(need):
        raise SystemExit(f"candidates missing from data/chunks/base.jsonl: {len(need) - n}")
    with open(os.path.join(ds, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"title": SLUG, "id": f"{KAGGLE_USER}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}, f, indent=1)
    shutil.copy(os.path.join(ROOT, "kaggle", "rerank_trunc12.py"), kern)
    with open(os.path.join(kern, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"id": f"{KAGGLE_USER}/{KSLUG}", "title": KSLUG.replace("-", " "),
                   "code_file": "rerank_trunc12.py", "language": "python",
                   "kernel_type": "script", "is_private": True, "enable_gpu": True,
                   "enable_tpu": False, "enable_internet": False, "machine_shape": "p100",
                   "dataset_sources": [f"{KAGGLE_USER}/{SLUG}"], "kernel_sources": [],
                   "competition_sources": [], "model_sources": [RERANKER_INPUT]}, f, indent=1)
    pkg = os.path.relpath(PKG, ROOT)
    print(f"package {pkg}: {len(dev)} dev questions, letters {sum(q['letter'] for q in dev)}, "
          f"{len(hybrid) * TOP} pairs per mode, {n} chunks")
    print(f"not pushed; commands:\nkaggle datasets create -p {pkg}/dataset\n"
          f"kaggle kernels push -p {pkg}/kernel\n"
          f"python3 scripts/kaggle_run.py fetch {KSLUG} --dest data/kaggle_out/{KSLUG}")


def score() -> None:
    dev = dev_queries()
    units = P.read_units(os.path.join(QDIR, "qrels_live12_punkt.tsv"))
    runs = {}
    tokens = {}
    for mode in MODES:
        rows = read_jsonl(os.path.join(FETCHED, f"{mode}.jsonl"))
        runs[mode] = {r["query_id"]: r["ranked"] for r in rows}
        tokens.update({r["query_id"]: r["tokens"] for r in rows})
    ids = {q["query_id"] for q in dev}
    if any(set(r) != ids for r in runs.values()):
        raise SystemExit("fetched runs do not cover the dev questions")
    stored = load_run("hybrid-rerank", ids)
    groups = {"dev": sorted(ids),
              "dev письма": sorted(q["query_id"] for q in dev if q["letter"]),
              "dev длиннее 256 токенов": sorted(q for q in ids if tokens[q] > 256)}
    result = {"задача": "12.14", "метрика": "NDCG@10 по пунктам (rufin.punkt.metrics), dev сплита 12.8",
              "контроль full против сохраненного hybrid-rerank, одинаковых top-10":
                  {"совпало": sum(runs["full"][q][:10] == stored[q][:10] for q in ids), "из": len(ids)},
              "группы": {}}
    for name, qids in groups.items():
        per = {m: np.array([P.metrics(runs[m][q], units[q])["NDCG@10"] for q in qids]) for m in MODES}
        d, p = M.paired_diff_ci(per["q256"], per["full"])
        result["группы"][name] = {"вопросов": len(qids), **{m: round(float(per[m].mean()), 4) for m in MODES},
                                  "q256 - full": {"среднее": round(d.mean, 4), "lo": round(d.lo, 4),
                                                  "hi": round(d.hi, 4), "p": round(p, 4)}}
        print(f"{name:26s} n={len(qids):3d}  full {per['full'].mean():.3f}  q256 {per['q256'].mean():.3f}  "
              f"diff {d.mean:+.3f} [{d.lo:+.3f}; {d.hi:+.3f}] p={p:.3f}")
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(result["контроль full против сохраненного hybrid-rerank, одинаковых top-10"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pack", "score"))
    {"pack": pack, "score": score}[ap.parse_args().cmd]()


if __name__ == "__main__":
    main()
