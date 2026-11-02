"""Fragments a training example may never touch.

One place for the rule, because two copies of it drifted on stage 12: the sampler read punkt
pieces from the old qrels_punkt.tsv, and the second fragment of a two-norm task was not checked
at all. A positive of any test counts, and so does every piece of a positive punkt: a question
written from the neighbouring piece of the same punkt still answers the test question.
"""
from __future__ import annotations

import csv
from pathlib import Path

QUERIES = Path(__file__).resolve().parents[2] / "data" / "queries"
GOLD_QRELS = ("qrels.tsv", "qrels_live_core.tsv", "qrels_live_extra.tsv", "qrels_live12.tsv")
PUNKT_QRELS = ("qrels_punkt.tsv", "qrels_live12_punkt.tsv")


def forbidden_chunks(queries: Path = QUERIES) -> frozenset[str]:
    out = set()
    for name in GOLD_QRELS:
        with open(queries / name, encoding="utf-8") as f:
            out |= {r["corpus-id"] for r in csv.DictReader(f, delimiter="\t")}
    for name in PUNKT_QRELS:
        with open(queries / name, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                out |= set(r["chunk-ids"].split(","))
    return frozenset(out)
