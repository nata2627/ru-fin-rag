#!/usr/bin/env python3
"""Pool runs for a new batch of questions: bm25, dense-bge-m3, hybrid, depth 50.

The same pipeline that produced data/runs/base__{bm25,dense-bge-m3,hybrid}.jsonl:
bm25_run.py, dense_run.py over the stage A matrix of bge-m3, make_hybrid.py
(RRF k=60). Query vectors are encoded here on CPU in fp32: on the 20 control
live questions they reproduce the old dense-bge-m3 top-10 in 9 of 20 cases,
MPS fp16 in 6 of 20; the rest of the gap comes from the 01.10 environment.

The cross-encoder does not fit the Mac (5.4 min per 3 queries on CPU), so
--kaggle SLUG builds a dataset and a kernel for kaggle/rerank_live12.py.
Nothing is pushed. The kernel slug is SLUG-k: Kaggle answers 409 when it
equals the dataset slug.

The bge-m3 matrix comes from the stage A kernel output:
  kaggle kernels output naumaaaa/ru-fin -p <dir> --file-pattern 'base/bge-m3/'
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

TOP = 50
RERANKER_INPUT = "andreasbis/baai-bge-reranker-v2-m3/transformers/default/1"
KAGGLE_USER = "naumaaaa"


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def step(cmd: list[str]) -> None:
    print("$ " + " ".join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd),
          flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def encode_queries(queries: list[dict], out: str, batch_size: int) -> None:
    import numpy as np

    from rufin.retrieval.dense import Encoder
    from rufin.retrieval.model_specs import MODELS

    enc = Encoder(MODELS["bge-m3"], device="cpu", batch_size=batch_size, half=False)
    np.save(out, enc.encode([q["text"] for q in queries], is_query=True))
    with open(out.replace(".npy", ".ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(q["query_id"] for q in queries) + "\n")


def kaggle_package(slug: str, out: str, queries: str, control: str) -> str:
    pkg = os.path.join(out, "kaggle")
    ds, kern = os.path.join(pkg, "dataset"), os.path.join(pkg, "kernel")
    shutil.rmtree(pkg, ignore_errors=True)
    os.makedirs(ds)
    os.makedirs(kern)

    # Control: 20 old live questions with their original hybrid and reranked
    # rows. The kernel stops if the attached reranker does not reproduce them.
    ctl = read_jsonl(control)
    ids = {q["query_id"] for q in ctl}
    ctl_h = [r for r in read_jsonl(os.path.join(ROOT, "data", "runs", "base__hybrid.jsonl"))
             if r["query_id"] in ids]
    ctl_r = [{"query_id": r["query_id"], "ranked": r["ranked"]} for r in
             read_jsonl(os.path.join(ROOT, "data", "runs", "base__hybrid-rerank.jsonl"))
             if r["query_id"] in ids]
    write_jsonl(os.path.join(ds, "control_queries.jsonl"), ctl)
    write_jsonl(os.path.join(ds, "control_hybrid.jsonl"), ctl_h)
    write_jsonl(os.path.join(ds, "control_hybrid_rerank.jsonl"), ctl_r)

    hybrid = read_jsonl(os.path.join(out, "base__hybrid.jsonl"))
    write_jsonl(os.path.join(ds, "base__hybrid.jsonl"), hybrid)
    shutil.copy(queries, os.path.join(ds, "queries.jsonl"))

    need = {c for r in hybrid + ctl_h for c in r["ranked"][:TOP]}
    n = 0
    with gzip.open(os.path.join(ds, "chunks_candidates.jsonl.gz"), "wt", encoding="utf-8") as f:
        for c in read_jsonl(os.path.join(ROOT, "data", "chunks", "base.jsonl")):
            if c["chunk_id"] in need:
                f.write(json.dumps({"chunk_id": c["chunk_id"], "text": c["text"]},
                                   ensure_ascii=False) + "\n")
                n += 1
    if n != len(need):
        raise SystemExit(f"candidates missing from data/chunks/base.jsonl: {len(need) - n}")

    with open(os.path.join(ds, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"title": slug, "id": f"{KAGGLE_USER}/{slug}",
                   "licenses": [{"name": "CC0-1.0"}]}, f, indent=1)
    shutil.copy(os.path.join(ROOT, "kaggle", "rerank_live12.py"), kern)
    kslug = f"{slug}-k"
    with open(os.path.join(kern, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"id": f"{KAGGLE_USER}/{kslug}", "title": kslug.replace("-", " "),
                   "code_file": "rerank_live12.py", "language": "python",
                   "kernel_type": "script", "is_private": True, "enable_gpu": True,
                   "enable_tpu": False, "enable_internet": False, "machine_shape": "p100",
                   "dataset_sources": [f"{KAGGLE_USER}/{slug}"], "kernel_sources": [],
                   "competition_sources": [], "model_sources": [RERANKER_INPUT]},
                  f, indent=1, ensure_ascii=False)
    print(f"kaggle package: {os.path.relpath(pkg, ROOT)}, {len(hybrid)} queries, "
          f"{len(hybrid) * TOP} pairs, {n} chunks, control {len(ctl_h)}", flush=True)
    return kslug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="jsonl with query_id and text")
    ap.add_argument("--out", required=True, help="output folder for base__*.jsonl")
    ap.add_argument("--embeddings", default=os.path.join(ROOT, "data", "embeddings",
                                                         "base", "bge-m3"),
                    help="stage A bge-m3 matrix: vectors.npy and ids.txt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--kaggle", metavar="SLUG", default=None,
                    help="build the hybrid-rerank dataset SLUG and kernel SLUG-k, no push")
    ap.add_argument("--control", default=os.path.join(ROOT, "data", "interim", "live12",
                                                      "control", "queries.jsonl"))
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    queries = read_jsonl(args.queries)
    if len({q["query_id"] for q in queries}) != len(queries):
        raise SystemExit("duplicate query_id")
    qpath = os.path.join(out, "queries.jsonl")
    write_jsonl(qpath, [{"query_id": q["query_id"], "text": q["text"]} for q in queries])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    py = sys.executable
    t0 = time.time()

    step([py, os.path.join(ROOT, "scripts", "bm25_run.py"), "--queries", qpath,
          "--out", os.path.join(out, "base__bm25.jsonl"), "--top", str(TOP)])
    t1 = time.time()

    qvec = os.path.join(out, "qvec_bge-m3.npy")
    encode_queries(queries, qvec, args.batch_size)
    t2 = time.time()
    print(f"bge-m3 query vectors on CPU fp32: {t2 - t1:.0f} s", flush=True)
    step([py, os.path.join(ROOT, "scripts", "dense_run.py"), "--model", "bge-m3",
          "--embeddings", args.embeddings, "--queries", qpath, "--query-vectors", qvec,
          "--out", os.path.join(out, "base__dense-bge-m3.jsonl"), "--top", str(TOP)])
    step([py, os.path.join(ROOT, "scripts", "make_hybrid.py"), "--rundir", out,
          "--top", str(TOP)])

    for name in ("bm25", "dense-bge-m3", "hybrid"):
        rows = read_jsonl(os.path.join(out, f"base__{name}.jsonl"))
        short = sum(len(r["ranked"]) < TOP for r in rows)
        if len(rows) != len(queries) or short:
            raise SystemExit(f"{name}: {len(rows)} rows of {len(queries)}, "
                             f"shorter than {TOP}: {short}")
    print(f"runs ready: {len(queries)} queries, bm25 {t1 - t0:.0f} s, "
          f"total {time.time() - t0:.0f} s", flush=True)

    if args.kaggle:
        kslug = kaggle_package(args.kaggle, out, qpath, args.control)
        pkg = os.path.relpath(os.path.join(out, "kaggle"), ROOT)
        print("\nnot pushed; commands:")
        print(f"kaggle datasets create -p {pkg}/dataset")
        print(f"kaggle datasets status {KAGGLE_USER}/{args.kaggle}")
        print(f"kaggle kernels push -p {pkg}/kernel")
        print(f"python3 scripts/kaggle_run.py watch {kslug} --deadline-hours 1 "
              f"--every 120 --match \"reranked|control\"")
        print(f"python3 scripts/kaggle_run.py fetch {kslug} --dest data/kaggle_out/{kslug}")


if __name__ == "__main__":
    main()
