"""Лист пробы судьи: текст вопроса стоит в каждой строке.

В пуле текст вопроса заполнен только в первой строке группы, и
`gpu_judge.read_pool` раздаёт его вниз по файлу. Лист пробы идёт в порядке
листа человека, то есть строки перемешаны между группами, и раздача вниз
подставила бы судье соседний вопрос. Ошибка молчит: разметка получится, она
просто будет не про то.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location(
    "make_judge_probe", os.path.join(ROOT, "scripts", "make_judge_probe.py")
)
probe = importlib.util.module_from_spec(spec)
sys.modules["make_judge_probe"] = probe
spec.loader.exec_module(probe)


def write(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_text_is_taken_from_the_same_question(tmp_path, monkeypatch):
    pool = tmp_path / "pool.tsv"
    human = tmp_path / "human.tsv"
    out = tmp_path / "probe.tsv"
    write(
        pool,
        [
            {"query_id": "q1", "chunk_id": "a#1", "prioritet": "1", "vopros": "первый вопрос"},
            {"query_id": "q1", "chunk_id": "a#2", "prioritet": "2", "vopros": ""},
            {"query_id": "q2", "chunk_id": "b#1", "prioritet": "1", "vopros": "второй вопрос"},
            {"query_id": "q2", "chunk_id": "b#2", "prioritet": "2", "vopros": ""},
        ],
        ["query_id", "chunk_id", "prioritet", "vopros"],
    )
    # Порядок листа человека не повторяет порядок пула: именно на этом
    # раздача текста вниз по файлу и ломается.
    write(
        human,
        [
            {"query_id": "q2", "chunk_id": "b#2"},
            {"query_id": "q1", "chunk_id": "a#2"},
        ],
        ["query_id", "chunk_id"],
    )
    monkeypatch.setattr(
        sys, "argv", ["probe", "--human", str(human), "--pool", str(pool), "--out", str(out)]
    )
    probe.main()
    rows = list(csv.DictReader(open(out, encoding="utf-8"), delimiter="\t"))
    assert [r["vopros"] for r in rows] == ["второй вопрос", "первый вопрос"]


def test_pair_missing_from_the_pool_stops_the_run(tmp_path, monkeypatch):
    pool = tmp_path / "pool.tsv"
    human = tmp_path / "human.tsv"
    write(
        pool,
        [{"query_id": "q1", "chunk_id": "a#1", "prioritet": "1", "vopros": "вопрос"}],
        ["query_id", "chunk_id", "prioritet", "vopros"],
    )
    write(human, [{"query_id": "q1", "chunk_id": "нет такого"}], ["query_id", "chunk_id"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--human", str(human), "--pool", str(pool), "--out", str(tmp_path / "o.tsv")],
    )
    try:
        probe.main()
    except SystemExit as stop:
        assert "нет" in str(stop)
    else:
        raise AssertionError("пропавшая пара не остановила сборку листа")
