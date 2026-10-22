"""Проверки докачки корпуса, которые можно сделать без сети.

Сам сбор с cbr.ru тестом не трогается: это чужой сайт, и дергать его из тестов
нельзя. Проверяется то, что ломается тихо: ключ акта, разница с корпусом и поля
фрагмента, которые должны совпасть с теми, что уже лежат в корпусе.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("update_corpus", ROOT / "scripts/update_corpus.py")
update_corpus = importlib.util.module_from_spec(spec)
sys.modules["update_corpus"] = update_corpus
spec.loader.exec_module(update_corpus)

from rufinrag.corpus import FIELDS  # noqa: E402

ACT = {
    "act_id": "9999-П_01102026",
    "number": "9999-П",
    "date": "01.10.2026",
    "type": "П",
    "kind": "положение",
    "title": "О порядке проверки ночного прогона",
    "issue": "Вестник Банка России № 1 (2026)",
    "issue_url": "https://www.cbr.ru/Queries/XsltBlock/File/59540/0000",
    "text": "Глава 1. Общие положения\n1.1. Настоящее Положение проверяет нарезку. " * 40,
}


def test_act_key_keeps_number_and_date():
    """Номер без даты не уникален: у методических рекомендаций нумерация годовая."""
    assert update_corpus.act_key("9999-П", "01.10.2026") == "9999-П_01102026"


def test_known_acts_reads_compressed_corpus(tmp_path):
    corpus = tmp_path / "chunks.jsonl.gz"
    with gzip.open(corpus, "wt", encoding="utf-8") as f:
        for act_id in ("590-П_28062017", "3054-У_04092013"):
            f.write(json.dumps({"act_id": act_id, "chunk_id": f"{act_id}#0000"}) + "\n")
    assert update_corpus.known_acts(corpus) == {"590-П_28062017", "3054-У_04092013"}


def test_chunk_rows_match_corpus_fields():
    rufin_chunking = pytest.importorskip("rufin.chunking")
    rows = update_corpus.chunk_rows(ACT, rufin_chunking.TokenRuler(None))
    assert rows
    for row in rows:
        assert tuple(row) == FIELDS
        assert row["act_id"] == ACT["act_id"]
        assert row["url"] == ACT["issue_url"]
        assert row["text"]
    assert [row["chunk_id"] for row in rows] == [
        f"{ACT['act_id']}#{i:04d}" for i in range(len(rows))
    ]


def test_skipped_acts_survive_between_runs(tmp_path):
    """Иначе прогон каждую ночь качает одни и те же 80 выпусков ради нуля актов."""
    path = tmp_path / "skipped.json"
    assert update_corpus.load_skipped(path) == {}
    update_corpus.save_skipped(path, {"590-П_28062017": {"reason": "мало кириллицы", "tries": 1}})
    assert update_corpus.load_skipped(path)["590-П_28062017"]["reason"] == "мало кириллицы"
