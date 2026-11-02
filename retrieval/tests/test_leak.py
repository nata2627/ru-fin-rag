"""Утечка эталона в обучение: кусок пункта-позитива и второй фрагмент «двух норм» запрещены."""
from __future__ import annotations

import csv
import pathlib
import random
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import synth12  # noqa: E402

from rufin.leak import GOLD_QRELS, PUNKT_QRELS, forbidden_chunks  # noqa: E402


@pytest.fixture
def queries(tmp_path: pathlib.Path) -> pathlib.Path:
    for name in GOLD_QRELS:
        (tmp_path / name).write_text("query-id\tcorpus-id\tscore\n", encoding="utf-8")
    (tmp_path / "qrels_live12.tsv").write_text("query-id\tcorpus-id\tscore\nexp0000\tA#0010\t1\n", encoding="utf-8")
    for name in PUNKT_QRELS:
        (tmp_path / name).write_text("query-id\tpunkt-id\tscore\tchunk-ids\tosnovanie\n", encoding="utf-8")
    (tmp_path / "qrels_live12_punkt.tsv").write_text(
        "query-id\tpunkt-id\tscore\tchunk-ids\tosnovanie\n"
        "exp0000\tA#куски0010-0011\t1\tA#0010,A#0011\tкуски одного длинного пункта\n", encoding="utf-8")
    return tmp_path


def test_kusok_punkta_zapreshen(queries):
    bad = forbidden_chunks(queries)
    assert {"A#0010", "A#0011"} <= bad
    assert "A#0012" not in bad


def test_vtoroy_fragment_ne_na_etalone(queries):
    chunks = {f"A#{i:04d}": {"chunk_id": f"A#{i:04d}", "act_id": "A", "position": i, "text": "x" * 300}
              for i in (1, 10, 11)}
    by_act = {"A": list(chunks)}
    bad = forbidden_chunks(queries)
    rng = random.Random(0)
    assert synth12.pick_second(chunks["A#0001"], chunks, by_act, bad, rng) is None
    assert synth12.pick_second(chunks["A#0001"], chunks, by_act, frozenset(), rng) in {"A#0010", "A#0011"}


@pytest.mark.data
def test_nastoyashchie_kuski_punktov_live12():
    bad = forbidden_chunks()
    with open(ROOT / "data" / "queries" / "qrels_live12_punkt.tsv", encoding="utf-8") as f:
        pieces = {c for r in csv.DictReader(f, delimiter="\t") for c in r["chunk-ids"].split(",")}
    assert pieces <= bad
