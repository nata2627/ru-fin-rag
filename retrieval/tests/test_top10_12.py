"""Доразметка десятки 12.16: в пул идут только неразмеченные пары своего сплита."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import top10_12 as T  # noqa: E402


def test_tolko_nerazmechennye_svoego_splita():
    runs = {"dev": {"q1": ["a", "b", "c"], "q2": ["a"]}, "test": {"q3": ["a", "d"]}}
    split_of = {"q1": "dev", "q2": "test", "q3": "test"}
    rows, report = T.unlabeled(runs, split_of, {("q1", "a"), ("q3", "a")}, depth=2)
    assert [(r["query_id"], r["chunk_id"]) for r in rows] == [("q1", "b"), ("q3", "d")]
    assert all(r["prioritet"] == T.Y.CORE for r in rows)
    assert report["dev"] == {"вопросов": 1, "пар в десятке": 2, "неразмеченных": 1, "вопросов с неразмеченными": 1}
