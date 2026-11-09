#!/usr/bin/env python3
"""Task 12.16: unlabeled pairs in the candidate's top ten on dev and test.

The labels of stage 12 come from a pool of five configurations. A trained
reranker brings its own chunks into the top ten, and an unlabeled chunk counts
as zero, so the candidate would be scored below its real level. Before the gate
these pairs go to the same labeler with the same instruction (S29, item 4).

    python3 scripts/top10_12.py --run dev=data/kaggle_out/ru-fin-train12-k/dev_runs/step<best>.jsonl \\
        --run test=<test run of the best step>
    python3 scripts/razmetka_yadro.py prepare --pool data/interim/live12/pool_top10.tsv \\
        --work data/interim/live12/razmetka_top10 --queries data/queries/live12.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import live12_build as B  # noqa: E402
import razmetka_yadro as Y  # noqa: E402
from live12_runs import read_jsonl  # noqa: E402

ROOT = Y.ROOT
LIVE12 = os.path.join(ROOT, "data", "interim", "live12")
POOL = os.path.join(LIVE12, "pool_top10.tsv")
SPLITS = (B.DEV, B.TEST)
DEPTH = 10


def unlabeled(runs: dict[str, dict[str, list[str]]], split_of: dict[str, str],
              labeled: set[tuple[str, str]], depth: int = DEPTH) -> tuple[list[dict], dict]:
    """Pool rows for pairs nobody has graded, and per-split counts for the journal."""
    rows, seen, report = [], set(), {}
    for split, run in runs.items():
        ids = sorted(q for q in run if split_of.get(q) == split)
        top = [(q, c) for q in ids for c in run[q][:depth]]
        new = [k for k in top if k not in labeled]
        report[split] = {"вопросов": len(ids), "пар в десятке": len(top), "неразмеченных": len(new),
                         "вопросов с неразмеченными": len({q for q, _ in new})}
        for q, c in new:
            if (q, c) not in seen:
                seen.add((q, c))
                rows.append({"query_id": q, "chunk_id": c, "prioritet": Y.CORE, "istochnik": f"top10-{split}"})
    return rows, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="SPLIT=PATH",
                    help="jsonl with query_id and ranked, one per split")
    ap.add_argument("--out", default=POOL)
    args = ap.parse_args()
    paths = dict(r.split("=", 1) for r in args.run)
    if set(paths) - set(SPLITS):
        raise SystemExit(f"splits are {SPLITS}, got {sorted(paths)}")
    split_of = {q["query_id"]: q["split"] for q in read_jsonl(os.path.join(Y.QDIR, "live12.jsonl"))}
    runs = {s: {r["query_id"]: r["ranked"] for r in read_jsonl(p)} for s, p in paths.items()}
    for s, run in runs.items():
        want = {q for q, sp in split_of.items() if sp == s}
        if set(run) != want:
            raise SystemExit(f"{s}: run has {len(set(run) & want)} of {len(want)} questions")
    labels, _, incomplete, _ = B.gather()
    if incomplete:
        raise SystemExit(f"{len(incomplete)} questions are labeled only in part")
    rows, report = unlabeled(runs, split_of, set(labels))
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "chunk_id", "prioritet", "istochnik"],
                           delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"{len(rows)} pairs -> {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
