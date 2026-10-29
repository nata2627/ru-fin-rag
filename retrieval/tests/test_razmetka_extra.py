"""Второй набор живых: метки собираются из листа и в тест не попадают."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import razmetka_extra as X  # noqa: E402
import razmetka_yadro as Y  # noqa: E402

from rufin.benchmark import read_qrels  # noqa: E402


def test_metki_iz_lista():
    expected = {}
    for r in Y.read_tsv(X.SHEET):
        g = Y.grade(r["otvet1_celikom"], r["otvet2_vmeste"])
        if g > 0:
            expected.setdefault(r["query_id"], {})[r["chunk_id"]] = g
    assert {q: dict(r) for q, r in read_qrels(X.QRELS).items()} == expected


def test_voprosy_ne_v_teste():
    subsets = json.loads((ROOT / "data" / "queries" / "podvyborki.json").read_text(encoding="utf-8"))
    test = {q for ids in subsets.values() for q in ids}
    assert not {r["query_id"] for r in Y.read_tsv(X.SHEET)} & test


def test_blok_dokumenta_iz_fayla():
    result = json.loads(pathlib.Path(X.RESULT).read_text(encoding="utf-8"))
    doc = pathlib.Path(X.DOC).read_text(encoding="utf-8")
    assert doc.split(X.BLOCK_START, 1)[1].split(X.BLOCK_END, 1)[0].strip() == X.doc_block(result)
    assert result["позитивов"] == sum(len(r) for r in read_qrels(X.QRELS).values())
