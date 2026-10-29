"""Пункт как единица разметки: склейка кусков, подпункты и счет метрики."""
from __future__ import annotations

import math
import random

from rufin import metrics as M
from rufin import punkt as P
from rufin.explanations import starts_clause


def chunk(cid: str, pos: int, units: str, body: str = "", section: str = "Глава 3",
          start: int | None = None, end: int | None = None) -> dict:
    s = pos * 100 if start is None else start
    return {"chunk_id": f"A#{cid}", "act_id": "A", "position": pos, "section": section,
            "units": units, "char_start": s, "char_end": s + 100 if end is None else end, "body": body}


def test_kuski_odnogo_punkta_skleivayutsya():
    chunks = [chunk("0000", 0, "3.1–3.2"), chunk("0001", 1, "3.3"),
              chunk("0002", 2, "3.3", start=180), chunk("0003", 3, "3.3", start=270),
              chunk("0004", 4, "3.4")]
    groups = P.piece_groups(chunks)
    assert groups["A#0001"] == ["A#0001", "A#0002", "A#0003"]
    assert "A#0000" not in groups and "A#0004" not in groups


def test_odin_nomer_v_raznyh_razdelah_ne_skleivaetsya():
    chunks = [chunk("0000", 0, "1", section="Приложение 1"),
              chunk("0001", 1, "1", section="Приложение 2")]
    assert P.piece_groups(chunks) == {}


def test_razryv_otrezkov_ne_skleivaetsya():
    chunks = [chunk("0000", 0, "3.3", start=0, end=100), chunk("0001", 1, "3.3", start=500)]
    assert P.piece_groups(chunks) == {}


def test_podpunkt_da_ssylka_net():
    assert P.starts_clause_or_sub("3.12.2.8. Ссуда классифицируется", "3.12")
    assert P.starts_clause_or_sub("текст\n3.12. Банк", "3.12")
    assert not P.starts_clause_or_sub("в соответствии с пунктом 3.12.1 настоящего", "3.12")
    assert not P.starts_clause_or_sub("3.120. Другой пункт", "3.12")
    assert not P.starts_clause_or_sub("13.12. Другой пункт", "3.12")


def test_podpunkty_v_prilozhenii_ne_zaschityvayutsya():
    act = [chunk("0000", 0, "3.12", "3.12. Пункт"),
           chunk("0001", 1, "3.12.1–3.12.2", "3.12.1. Подпункт"),
           chunk("0009", 9, "3.12.1", "3.12.1. Строка таблицы", section="Приложение 2")]
    hit = P.clause_chunks(act, "3.12", {}, starts_clause)
    assert hit == {"A#0000", "A#0001"}


def test_prezhnie_pozitivy_ostayutsya_v_edinitsah():
    act = [chunk("0000", 0, "3.9.1–3.9.2", "3.9.1. a\n3.9.2. b"), chunk("0001", 1, "3.9.3", "3.9.3. c"),
           chunk("0002", 2, "3.10", "3.10. d")]
    qrels = {"q": {"A#0000": 2, "A#0001": 2, "A#0002": 1}}
    units = P.build_units(qrels, {"q": ["3.9.1", "3.9.2", "3.9.3"]}, {"A": act}, {"q": "A"}, {}, starts_clause)
    assert sorted(units["q"]) == ["A#0002", "A#п3.9.1", "A#п3.9.2", "A#п3.9.3"]
    for cid in qrels["q"]:
        assert any(cid in u.chunks for u in units["q"].values())


def test_idealnaya_vydacha_daet_edinitsu():
    """Фрагмент с двумя названными пунктами занимает две позиции, как в идеале.
    Первая версия счета давала здесь 0,765: вклад фрагмента был один, а идеал
    требовал две позиции."""
    units = {"п1": P.Unit("п1", 2, {"A#0000"}), "п2": P.Unit("п2", 2, {"A#0000"}),
             "п3": P.Unit("п3", 2, {"A#0001"})}
    m = P.metrics(["A#0000", "A#0001", "x"], units)
    assert m["NDCG@10"] == 1.0 and m["Recall@10"] == 1.0 and m["MRR@10"] == 1.0


def test_povtor_punkta_ne_zanimaet_pozitsiyu():
    units = {"куски": P.Unit("куски", 2, {"A#0001", "A#0002"}), "f": P.Unit("f", 1, {"A#0009"})}
    assert P.expand(["A#0001", "A#0002", "x", "A#0009"], units, 10) == [2, 0, 1]


def test_bez_kuskov_i_punktov_chisla_prezhnie():
    rng = random.Random(7)
    pool = [f"A#{i:04d}" for i in range(60)]
    for _ in range(200):
        rel = {c: rng.choice((1, 2)) for c in rng.sample(pool, rng.randint(1, 6))}
        ranked = rng.sample(pool, 30)
        units = {c: P.Unit(c, s, {c}) for c, s in rel.items()}
        new = P.metrics(ranked, units)
        assert math.isclose(new["NDCG@10"], M.ndcg_at_k(ranked, rel, 10))
        assert math.isclose(new["Recall@10"], M.recall_at_k(ranked, rel, 10))
        assert math.isclose(new["MRR@10"], M.mrr_at_k(ranked, rel, 10))


def test_zapis_i_chtenie(tmp_path):
    units = {"q": {"A#п3.12": P.Unit("A#п3.12", 2, {"A#0001", "A#0000"}, "пункт 3.12")}}
    path = tmp_path / "qrels_punkt.tsv"
    P.write_units(str(path), units)
    back = P.read_units(str(path))
    assert back["q"]["A#п3.12"].chunks == {"A#0000", "A#0001"}
    assert back["q"]["A#п3.12"].score == 2


def test_tablitsa_dokumenta_iz_fayla():
    """Таблица «до и после» в документе 18 совпадает с тем, что скрипт пишет из
    punkt-effect.json, и ни одна связка не упала с интервалом ниже нуля."""
    import json
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import punkt_effect as E

    result = json.loads((root / "data" / "results" / "punkt-effect.json").read_text(encoding="utf-8"))
    doc = (root / "docs" / "18-nabor-do-i-posle.md").read_text(encoding="utf-8")
    table = doc.split(E.TABLE_START, 1)[1].split(E.TABLE_END, 1)[0]
    assert table.strip() == E.doc_table(result)
    assert result["упало с интервалом ниже нуля"] == 0
    assert sorted(result["провалы"]["исчезли"]) == sorted(result["провалы"]["четыре провала метрики"])
