#!/usr/bin/env python3
"""Task 12.15: package the reranker training for Kaggle and score what comes back.

One dataset serves both kernels: the probe (96 training questions, 40 dev) gives
pairs per second and memory for the forecast, the full run trains on all of
live_synth.jsonl (decision 32). The kernel code is the same file, the probe
differs by one constant.

    python3 scripts/train12.py pack     # dataset and two kernels, no push
    python3 scripts/train12.py score    # after fetch of the full run: best step against hybrid-rerank on dev
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from live12_runs import KAGGLE_USER, RERANKER_INPUT, TOP, read_jsonl, write_jsonl  # noqa: E402
from trunc12 import dev_queries, load_run  # noqa: E402

from rufin import metrics as M  # noqa: E402
from rufin import punkt as P  # noqa: E402
from rufin.leak import forbidden_chunks  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
TRAIN = os.path.join(ROOT, "data", "interim", "live12", "train", "live_synth.jsonl")
PKG = os.path.join(ROOT, "data", "interim", "live12", "kaggle_train12")
SLUG = "ru-fin-train12"
KERNELS = {"probe": f"{SLUG}-probe", "full": f"{SLUG}-k", "live": f"{SLUG}-live", "lr3": f"{SLUG}-lr3",
           "live10": f"{SLUG}-live10", "live10x2": f"{SLUG}-live10x2"}
# After the full run lost to the base model (09.11): live questions only, and the
# same set at a lower rate for one epoch. One change per kernel against "full".
VARIANTS = {"probe": {"PROBE": "True"}, "full": {}, "live": {"LIVE_ONLY": "True"},
            "lr3": {"LR": "3e-6", "EPOCHS": "1"},
            # Live only won and was still rising at its last step: ten epochs with one
            # evaluation each, at the same rate and at double the rate.
            "live10": {"LIVE_ONLY": "True", "EPOCHS": "10", "EVALS_PER_EPOCH": "1"},
            "live10x2": {"LIVE_ONLY": "True", "EPOCHS": "10", "EVALS_PER_EPOCH": "1", "LR": "2e-5"}}
CODE = "train_rerank12.py"
RESULT = os.path.join(ROOT, "data", "results", "train12.json")


def pack() -> None:
    train = read_jsonl(TRAIN)
    forbidden = forbidden_chunks()
    used = {c for r in train for c in (*r["pos"], *r["neg"], *r["neg_random"])}
    if used & forbidden:
        raise SystemExit(f"training pairs on forbidden fragments: {len(used & forbidden)}")
    dev = dev_queries()
    ids = {q["query_id"] for q in dev}
    hybrid = load_run("hybrid", ids)
    units = P.read_units(os.path.join(QDIR, "qrels_live12_punkt.tsv"))

    ds = os.path.join(PKG, "dataset")
    shutil.rmtree(PKG, ignore_errors=True)
    os.makedirs(ds)
    shutil.copy(TRAIN, ds)
    write_jsonl(os.path.join(ds, "dev_queries.jsonl"), [{"query_id": q["query_id"], "text": q["text"]} for q in dev])
    write_jsonl(os.path.join(ds, "dev_hybrid.jsonl"),
                [{"query_id": q, "ranked": r[:TOP]} for q, r in sorted(hybrid.items())])
    P.write_units(os.path.join(ds, "dev_punkt.tsv"), {q: units[q] for q in ids})
    shutil.copy(os.path.join(ROOT, "src", "rufin", "punkt.py"), ds)
    need = used | {c for r in hybrid.values() for c in r[:TOP]}
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

    code = open(os.path.join(ROOT, "kaggle", CODE), encoding="utf-8").read()
    for const in ("PROBE = False", "LIVE_ONLY = False", "LR = 1e-5", "EPOCHS = 2", "EVALS_PER_EPOCH = 2"):
        if code.count(const + "\n") != 1:
            raise SystemExit(f"kaggle/{CODE}: {const} is not where the packer expects it")
    for mode, kslug in KERNELS.items():
        kern = os.path.join(PKG, mode)
        os.makedirs(kern)
        with open(os.path.join(kern, CODE), "w", encoding="utf-8") as f:
            body = code
            for name, value in VARIANTS[mode].items():
                body = re.sub(rf"^{name} = .*$", f"{name} = {value}", body, count=1, flags=re.M)
            f.write(body)
        # Empty machine_shape is the T4 x2 node: fp16 autocast needs Turing, P100 has no fp16 tensor cores.
        with open(os.path.join(kern, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"id": f"{KAGGLE_USER}/{kslug}", "title": kslug.replace("-", " "),
                       "code_file": CODE, "language": "python", "kernel_type": "script",
                       "is_private": True, "enable_gpu": True, "enable_tpu": False,
                       "enable_internet": False, "machine_shape": "",
                       "dataset_sources": [f"{KAGGLE_USER}/{SLUG}"], "kernel_sources": [],
                       "competition_sources": [], "model_sources": [RERANKER_INPUT]}, f, indent=1)
    pkg = os.path.relpath(PKG, ROOT)
    print(f"package {pkg}: train {len(train)} questions, dev {len(dev)}, fragments {n}, "
          f"forbidden among pairs 0 of {len(used)}")
    print(f"not pushed; commands:\nkaggle datasets create -p {pkg}/dataset")
    for mode, kslug in KERNELS.items():
        print(f"kaggle kernels push -p {pkg}/{mode}\n"
              f"python3 scripts/kaggle_run.py fetch {kslug} --dest data/kaggle_out/{kslug}")


def score(mode: str) -> None:
    out = os.path.join(ROOT, "data", "kaggle_out", KERNELS[mode])
    evals = read_jsonl(os.path.join(out, "evals.jsonl"))
    best = max(evals, key=lambda e: e["dev_ndcg10"])
    dev = dev_queries()
    ids = sorted(q["query_id"] for q in dev)
    units = P.read_units(os.path.join(QDIR, "qrels_live12_punkt.tsv"))
    base = load_run("hybrid-rerank", set(ids))
    cand = {r["query_id"]: r["ranked"] for r in read_jsonl(os.path.join(out, "dev_runs", f"step{best['step']}.jsonl"))}
    a = np.array([P.metrics(cand[q], units[q])["NDCG@10"] for q in ids])
    b = np.array([P.metrics(base[q], units[q])["NDCG@10"] for q in ids])
    d, p = M.paired_diff_ci(a, b)
    letters = [i for i, q in enumerate(sorted(dev, key=lambda q: q["query_id"])) if q["letter"]]
    result = {"задача": "12.15", "ядро": KERNELS[mode], "обучение": {"live_synth.jsonl, решение 32": VARIANTS[mode]},
              "метрика": "NDCG@10 по пунктам на dev сплита 12.8; dev же выбирал шаг, оценка оптимистична",
              "оценки по шагам": evals, "лучший шаг": best["step"],
              "dev": {"вопросов": len(ids), "кандидат": round(float(a.mean()), 4),
                      "hybrid-rerank": round(float(b.mean()), 4),
                      "разница": {"среднее": round(d.mean, 4), "lo": round(d.lo, 4), "hi": round(d.hi, 4),
                                  "p": round(p, 4)}},
              "dev письма": {"вопросов": len(letters), "кандидат": round(float(a[letters].mean()), 4),
                             "hybrid-rerank": round(float(b[letters].mean()), 4)}}
    with open(RESULT if mode == "full" else RESULT.replace(".json", f"-{mode}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"best step {best['step']}: dev {a.mean():.4f} vs hybrid-rerank {b.mean():.4f}, "
          f"diff {d.mean:+.4f} [{d.lo:+.4f}; {d.hi:+.4f}] p={p:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pack", "score"))
    ap.add_argument("--mode", choices=tuple(KERNELS), default="full", help="which kernel to score")
    args = ap.parse_args()
    pack() if args.cmd == "pack" else score(args.mode)


if __name__ == "__main__":
    main()
