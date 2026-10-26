"""Разбор двадцати провалов сходится с правилом отбора, записанным до отбора.

Разбор это файл с ручными колонками, а отбор это код. Между ними легко
разъехаться: поменялась выдача, поменялся эталон, кто-то дописал строку.
Тогда таблица документа 16 начнёт описывать не те двадцать запросов, и
увидеть это по самой таблице нельзя. Поэтому отбор воспроизводится здесь.
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAZBOR = ROOT / "docs" / "raw" / "razbor_provalov.tsv"
RESULT = ROOT / "data" / "results" / "errors-zhivye.json"

# правило из `orchestration/EXPERIMENTS.md` части Б, раздел S15
RUN = ROOT / "data" / "runs" / "base__hybrid-rerank.jsonl"
SEED = 20261023
TAKE = 20

ПРИЧИНЫ = {
    "ответ найден, но не размечен",
    "эталон общая норма",
    "несовпадение слов",
    "связь двух норм",
    "эталон ниже десятки",
    "номером акта поиск не пользуется",
}
ГРУППЫ = {"помогла бы", "не помогла бы", "не узнать"}


@pytest.fixture
def разбор() -> list[dict]:
    with open(RAZBOR, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def отбор(data_dir: pathlib.Path) -> list[str]:
    qrels: dict[str, set[str]] = {}
    with open(data_dir / "queries" / "qrels.tsv", encoding="utf-8") as f:
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            if qid == "query-id":
                continue
            if int(score) > 0:
                qrels.setdefault(qid, set()).add(cid)
    run = {r["query_id"]: r["ranked"] for r in
           (json.loads(l) for l in open(RUN, encoding="utf-8"))}
    живые = set(json.load(open(data_dir / "queries" / "podvyborki.json",
                               encoding="utf-8"))["живые"])
    провалы = [q for q in sorted(живые & set(run))
               if qrels.get(q) and not (qrels[q] & set(run[q][:10]))]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(провалы), size=min(TAKE, len(провалы)), replace=False)
    return [провалы[i] for i in sorted(idx)]


def test_prichiny_i_gruppy_iz_spiska(разбор):
    assert len(разбор) == TAKE
    assert {r["причина"] for r in разбор} <= ПРИЧИНЫ
    assert {r["помогла_бы"] for r in разбор} <= ГРУППЫ
    assert all(r["заметка"].strip() for r in разбор)


@pytest.mark.data
def test_otbor_vosproizvoditsya(разбор, data_dir):
    if not RUN.exists():
        pytest.skip("выдачи нет: data/runs не версионируется, нужен прогон на видеокарте")
    assert sorted(r["query_id"] for r in разбор) == sorted(отбор(data_dir))


@pytest.mark.data
def test_chisla_otcheta_sovpadayut_s_razborom(разбор):
    if not RESULT.exists():
        pytest.skip("отчёт не посчитан: нужен `make razbor-provalov`")
    отчёт = json.load(open(RESULT, encoding="utf-8"))
    причины: dict[str, int] = {}
    группы: dict[str, int] = {}
    for r in разбор:
        причины[r["причина"]] = причины.get(r["причина"], 0) + 1
        группы[r["помогла_бы"]] = группы.get(r["помогла_бы"], 0) + 1
    assert отчёт["разбор"]["причины"] == причины
    assert отчёт["разбор"]["помогла бы человеческая манера вопроса"] == группы
    assert отчёт["разобрано"] == TAKE
    assert отчёт["семя отбора"] == SEED


def test_tablicy_dokumenta_sovpadayut_s_razborom(разбор):
    """Числа таблиц документа 16 берутся из разбора, а не набираются руками."""
    текст = (ROOT / "docs" / "16-razbor-provalov.md").read_text(encoding="utf-8")
    строки = [[c.strip() for c in line.strip().strip("|").split("|")]
              for line in текст.splitlines() if line.startswith("|")]

    def число_в_таблице(подпись: str) -> int | None:
        for ячейки in строки:
            if ячейки and ячейки[0].strip("*") == подпись:
                for c in ячейки[1:]:
                    if c.isdigit():
                        return int(c)
        return None

    for поле in ("причина", "помогла_бы"):
        счёт: dict[str, int] = {}
        for r in разбор:
            счёт[r[поле]] = счёт.get(r[поле], 0) + 1
        for подпись, n in счёт.items():
            assert число_в_таблице(подпись) == n, f"{подпись}: в документе не {n}"
