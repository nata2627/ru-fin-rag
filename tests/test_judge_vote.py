"""Голосование двух судей: в метрику идет только согласие.

Проверяется то, из-за чего правило и объявлено до разметки: согласованная
пара попадает в эталон, пара несогласия не попадает никуда, кроме списка
расхождений, знаменатель доли считается от числа пар, размеченных обоими,
а строка с неразобранным ответом разметкой не считается и в знаменатель
не идет.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location(
    "judge_vote", os.path.join(ROOT, "scripts", "judge_vote.py")
)
vote = importlib.util.module_from_spec(spec)
sys.modules["judge_vote"] = vote
spec.loader.exec_module(vote)

FIELDS = ["query_id", "chunk_id", "otvet_celikom", "otvet_vmeste", "tip",
          "otsenka_0_1_2", "syroy_otvet"]


def mark(qid, cid, score, kind="-"):
    """Строка листа судьи. Оценка и ответы согласованы правилом сборки."""
    if score == 2:
        first, second = "да", "-"
    else:
        first, second = "нет", ("да" if score == 1 else "нет")
    return {"query_id": qid, "chunk_id": cid, "otvet_celikom": first,
            "otvet_vmeste": second, "tip": kind, "otsenka_0_1_2": score,
            "syroy_otvet": f"1: {first} 2: {second}"}


def write(path, rows, fields=FIELDS):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run(tmp_path, monkeypatch, rows_a, rows_b):
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    write(a, rows_a)
    write(b, rows_b)
    out = tmp_path / "qrels_vote.tsv"
    disagreed = tmp_path / "disagreed.tsv"
    result = tmp_path / "vote.json"
    monkeypatch.setattr(sys, "argv", [
        "judge_vote", "--judge-a", str(a), "--judge-b", str(b),
        "--out", str(out), "--disagreed", str(disagreed),
        "--report", str(tmp_path / "vote.txt"), "--result", str(result),
    ])
    vote.main()
    qrels = [line.split("\t") for line in
             out.read_text(encoding="utf-8").splitlines()[1:]]
    disagreements = list(csv.DictReader(open(disagreed, encoding="utf-8"),
                                        delimiter="\t"))
    return qrels, disagreements, json.loads(result.read_text(encoding="utf-8"))


def test_agreement_goes_to_the_benchmark(tmp_path, monkeypatch):
    qrels, disagreements, result = run(
        tmp_path, monkeypatch,
        [mark("q1", "a#1", 2), mark("q1", "a#2", 1), mark("q2", "b#1", 0)],
        [mark("q1", "a#1", 2), mark("q1", "a#2", 1), mark("q2", "b#1", 0)],
    )
    assert qrels == [["q1", "a#1", "2"], ["q1", "a#2", "1"]]
    assert disagreements == []
    assert result["размечено обоими"] == 3
    assert result["совпало"] == 3
    assert result["разошлось"] == 0
    # Согласованный ноль это разметка, но в qrels нерелевантные пары
    # не перечисляются.
    assert result["согласованных по оценкам"]["0"] == 1
    assert result["в qrels"] == 2


def test_disagreement_is_dropped_and_listed(tmp_path, monkeypatch):
    qrels, disagreements, result = run(
        tmp_path, monkeypatch,
        [mark("q1", "a#1", 2), mark("q1", "a#2", 2), mark("q1", "a#3", 1)],
        [mark("q1", "a#1", 2), mark("q1", "a#2", 0), mark("q1", "a#3", 2)],
    )
    assert qrels == [["q1", "a#1", "2"]]
    assert {(r["chunk_id"], r["otsenka_a"], r["otsenka_b"]) for r in disagreements} == {
        ("a#2", "2", "0"), ("a#3", "1", "2")}
    assert result["разошлось"] == 2
    assert result["размечено обоими"] == 3
    # 1 против 2 это такое же несогласие, как 2 против 0, но грубое из них
    # только второе.
    assert result["разошлось на две ступени"] == 1
    assert result["матрица"]["2-0"] == 1 and result["матрица"]["1-2"] == 1


def test_denominator_counts_only_pairs_both_judges_marked(tmp_path, monkeypatch):
    qrels, disagreements, result = run(
        tmp_path, monkeypatch,
        [mark("q1", "a#1", 2), mark("q1", "a#2", 2), mark("q1", "a#3", 1)],
        [mark("q1", "a#1", 2), mark("q1", "a#4", 2)],
    )
    assert result["размечено обоими"] == 1
    assert result["размечено только A"] == 2
    assert result["размечено только B"] == 1
    # Пара без второго голоса в эталон не идет так же, как пара несогласия,
    # но несогласием не считается.
    assert qrels == [["q1", "a#1", "2"]]
    assert disagreements == []
    assert result["разошлось"] == 0


def test_unparsed_answer_is_not_annotation(tmp_path, monkeypatch):
    unparsed = {"query_id": "q1", "chunk_id": "a#2", "otvet_celikom": "",
                "otvet_vmeste": "", "tip": "", "otsenka_0_1_2": "",
                "syroy_otvet": "фрагмент относится к делу частично"}
    qrels, disagreements, result = run(
        tmp_path, monkeypatch,
        [mark("q1", "a#1", 2), unparsed],
        [mark("q1", "a#1", 2), mark("q1", "a#2", 1)],
    )
    assert result["судья A"]["пар размечено"] == 1
    assert result["судья A"]["строк пропущено"]["ответ не разобран"] == 1
    assert result["размечено обоими"] == 1
    assert result["совпало"] == 1
    assert qrels == [["q1", "a#1", "2"]]
    assert disagreements == []


def test_score_is_assembled_from_two_answers(tmp_path, monkeypatch):
    """Лист человека колонки оценки не имеет: оценка собирается из ответов."""
    human = [{"query_id": "q1", "chunk_id": "a#1", "otvet1_celikom": "нет",
              "otvet2_vmeste": "да", "tip": "часть"},
             {"query_id": "q1", "chunk_id": "a#2", "otvet1_celikom": "да",
              "otvet2_vmeste": "-", "tip": "-"}]
    path = tmp_path / "human.tsv"
    write(path, human, ["query_id", "chunk_id", "otvet1_celikom",
                        "otvet2_vmeste", "tip"])
    sheet = vote.read_sheet(str(path))
    assert sheet.marks == {("q1", "a#1"): 1, ("q1", "a#2"): 2}
    assert sheet.types == {("q1", "a#1"): "часть"}


def test_sheets_without_a_common_pair_stop_the_run(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as stop:
        run(tmp_path, monkeypatch,
            [mark("q1", "a#1", 2)], [mark("q2", "b#1", 2)])
    assert "не пересекаются" in str(stop.value)
