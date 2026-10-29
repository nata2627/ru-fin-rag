"""Проверки сплита по актам.

Утечка через сплит — ошибка, которая не падает и не видна по метрикам:
качество просто оказывается выше, чем есть, и объяснить это будет нечем.
Поэтому проверок четыре, и три из них — про данные, а не про код.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from rufin import split as S

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "queries" / "split.json"


@pytest.fixture
def сплит() -> dict:
    if not SPLIT.exists():
        pytest.skip("сплита нет: `make split`")
    return S.load(SPLIT)


def test_группы_не_пересекаются(сплит):
    assert S.overlaps(сплит) == []


def test_каждый_акт_ровно_в_одной_группе(сплит):
    из_групп = [a for acts in сплит["gruppy"].values() for a in acts]
    assert len(из_групп) == len(set(из_групп))
    assert set(сплит["gruppy"]) == set(S.GROUPS)


@pytest.mark.data
def test_акты_с_эталонами_теста_не_попали_в_обучение():
    """Эталон живого вопроса не может лежать в обучающем акте.

    Иначе прирост от обучения будет отчасти узнаванием того же текста,
    и отделить одно от другого будет нечем.
    """
    if not SPLIT.exists():
        pytest.skip("сплита нет")
    сплит = S.load(SPLIT)
    train = set(сплит["gruppy"][S.TRAIN])
    нарушения = []
    for имя in ("qrels.tsv", "qrels_by_clause.tsv"):
        path = ROOT / "data" / "queries" / имя
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            next(f)
            for line in f:
                qid, cid = line.split("\t")[:2]
                if qid.startswith(("exp", "man")) and S.act_of_chunk(cid) in train:
                    нарушения.append((qid, cid))
    assert not нарушения, f"эталоны теста в обучающих актах: {нарушения[:5]}"


@pytest.mark.data
def test_класс_документов_отложен_целиком():
    """Все Инструкции вне обучения — иначе перенос на новый жанр не измерить."""
    if not SPLIT.exists():
        pytest.skip("сплита нет")
    сплит = S.load(SPLIT)
    corpus = ROOT / "data" / "corpus" / "acts.jsonl"
    if not corpus.exists():
        pytest.skip("корпус не собран")
    train = set(сплит["gruppy"][S.TRAIN])
    with corpus.open(encoding="utf-8") as f:
        инструкции = {json.loads(line)["act_id"] for line in f
                      if line.strip() and json.loads(line)["type"] == "И"}
    assert not (инструкции & train)


@pytest.mark.data
def test_объединение_групп_совпадает_с_корпусом():
    if not SPLIT.exists():
        pytest.skip("сплита нет")
    сплит = S.load(SPLIT)
    corpus = ROOT / "data" / "corpus" / "acts.jsonl"
    if not corpus.exists():
        pytest.skip("корпус не собран")
    with corpus.open(encoding="utf-8") as f:
        все = {json.loads(line)["act_id"] for line in f if line.strip()}
    из_групп = {a for acts in сплит["gruppy"].values() for a in acts}
    assert из_групп == все


def test_акт_фрагмента_отрезается_по_решётке():
    assert S.act_of_chunk("590-П_28062017#0034") == "590-П_28062017"


def test_podlozhennoe_peresechenie_po_fragmentam_lovitsya():
    """Проверка по фрагментам ловит пересечение, которое проверка по актам
    пропускает: запрос подвыборки «синтетика» с префиксом syn в нее не входит."""
    train = [{"query_id": "tr1", "gold_chunk_id": "A#0001"},
             {"query_id": "tr2", "gold_chunk_id": "B#0001", "gold_chunk_id_2": "A#0007"},
             {"query_id": "tr3", "gold_chunk_id": "C#0004"}]
    test = {"A#0007", "A#0001", "Z#0000"}
    assert S.fragment_leaks(train, test) == [("tr1", "A#0001"), ("tr2", "A#0007")]
    kept, dropped = S.drop_leaked(train, test)
    assert [r["query_id"] for r in kept] == ["tr3"]
    assert [r["query_id"] for r in dropped] == ["tr1", "tr2"]


def test_para_fragmentov_tozhe_istochnik():
    assert S.sources_of({"chunk_id": "A#1", "chunk_id_2": "B#2"}) == {"A#1", "B#2"}


@pytest.mark.data
def test_obuchenie_bez_utechki_po_fragmentam():
    """Выборка без утечки (задача 11.2) не делит ни одного фрагмента с эталоном
    теста в единицах «пункт», а убранного ровно столько, сколько записано."""
    from rufin.punkt import read_units

    qdir = ROOT / "data" / "queries"
    path = qdir / "synthetic_train_noleak.jsonl"
    if not path.exists():
        pytest.skip("выборки без утечки нет: `python3 scripts/drop_leak.py`")
    эталон = {c for us in read_units(str(qdir / "qrels_punkt.tsv")).values()
              for u in us.values() if u.score > 0 for c in u.chunks}
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    assert S.fragment_leaks(rows, эталон) == []
    итог = json.loads((ROOT / "data" / "results" / "utechka.json").read_text(encoding="utf-8"))
    assert итог["обучение"]["осталось"] == len(rows)
    assert len(итог["фрагменты утечки"]) == итог["обучение"]["убрано"]


def test_blok_utechki_v_dokumente_iz_fayla():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import drop_leak as D

    итог = json.loads((ROOT / "data" / "results" / "utechka.json").read_text(encoding="utf-8"))
    doc = (ROOT / "docs" / "18-nabor-do-i-posle.md").read_text(encoding="utf-8")
    блок = doc.split(D.BLOCK_START, 1)[1].split(D.BLOCK_END, 1)[0]
    assert блок.strip() == D.doc_block(итог)
