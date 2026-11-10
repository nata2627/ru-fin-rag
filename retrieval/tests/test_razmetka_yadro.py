"""Разметка ядра: метки только дополняются, файл меток пересобирается из листа."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import razmetka_yadro as Y  # noqa: E402

from rufin.benchmark import read_qrels  # noqa: E402


def sheet() -> dict[tuple[str, str], int]:
    return {(r["query_id"], r["chunk_id"]): Y.grade(r["otvet1_celikom"], r["otvet2_vmeste"])
            for r in Y.read_tsv(Y.SHEET)}


def test_list_pokryvaet_vse_yadro():
    if not (pathlib.Path(Y.POOL).exists() and pathlib.Path(Y.SHEET).exists()):
        pytest.skip("пула или листа ядра нет, они вне git")
    core = {(r["query_id"], r["chunk_id"]) for r in Y.core_pairs()}
    assert set(sheet()) == core


def test_prezhnie_pozitivy_ostalis():
    old = read_qrels(str(ROOT / "data" / "queries" / "qrels.tsv"))
    core = read_qrels(Y.CORE_QRELS)
    lost = [(q, c) for q in core for c, s in old[q].items() if s > 0 and core[q].get(c, 0) < s]
    assert lost == []


def test_metki_peresobirayutsya_iz_lista():
    old = read_qrels(str(ROOT / "data" / "queries" / "qrels.tsv"))
    core = read_qrels(Y.CORE_QRELS)
    expected = {q: {c: s for c, s in old[q].items() if s > 0} for q in core}
    for (q, c), g in sheet().items():
        if g > 0:
            expected[q][c] = max(expected[q].get(c, 0), g)
    assert {q: dict(r) for q, r in core.items()} == expected


def test_blok_dokumenta_iz_fayla():
    result = json.loads(pathlib.Path(Y.RESULT).read_text(encoding="utf-8"))
    doc = pathlib.Path(Y.DOC).read_text(encoding="utf-8")
    assert doc.split(Y.BLOCK_START, 1)[1].split(Y.BLOCK_END, 1)[0].strip() == Y.doc_block(result)
    assert result["плотность живого теста"]["плотнее"]["меток"] == sum(
        1 for r in read_qrels(Y.CORE_QRELS).values() for s in r.values() if s > 0)
