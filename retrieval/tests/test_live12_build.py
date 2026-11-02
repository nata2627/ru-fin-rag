"""Живой набор этапа 12: сплит не режет группы, вопрос без позитива в метрику не идет, метки только разметчика."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import live12_build as L  # noqa: E402

from rufin.benchmark import write_qrels  # noqa: E402


def question(qid: str, group: int, *, old: bool = False, route: str = "раздел", act: str = "A") -> dict:
    return {"query_id": qid, "text": "вопрос", "route": route, "act_id": act, "group": group,
            "old": old, "letter": False, "без590": True}


def test_gruppa_dubley_ne_rezhetsya():
    rows, n = [], 0
    for g in range(60):
        size = 1 + g % 3
        for _ in range(size):
            rows.append(question(f"q{n:03d}", g, old=g % 2 == 0, route=("раздел", "текст")[g % 4 // 2],
                                 act=("A", "B", "C")[g % 3]))
            n += 1
    split = L.split_dev_test(rows)
    for g in range(60):
        assert len({split[q["query_id"]] for q in rows if q["group"] == g}) == 1
    dev = sum(1 for s in split.values() if s == L.DEV)
    assert abs(dev - 0.25 * len(rows)) <= 3
    assert split == L.split_dev_test(rows)


def test_vopros_bez_pozitiva_ne_v_metrike_i_pechataetsya(capsys):
    queries = [question(f"q{i}", i) for i in range(8)]
    qrels = {f"q{i}": {f"A#{i:04d}": 2} for i in range(7)}
    split = L.assign(queries, qrels, set())
    assert split["q7"] == L.NO_ANSWER
    subsets = L.subsets_of(queries, split)
    assert subsets[L.NO_ANSWER] == ["q7"]
    assert "q7" not in subsets["живые тест"] + subsets["живые dev"]

    units = L.to_units(qrels, [])
    runs = {"bm25": {q["query_id"]: ["A#0000"] for q in queries}}
    per = L.ndcg_rows(runs, units, subsets["живые тест"])
    assert len(per["bm25"]) == len(subsets["живые тест"]) and not any(v != v for v in per["bm25"])
    with pytest.raises(ValueError):
        L.ndcg_rows(runs, units, ["q7"])

    L.print_no_answer([q for q in queries if q["query_id"] == "q7"], len(queries))
    out = capsys.readouterr().out
    assert "1 из 8" in out and "q7" in out


def test_metki_tolko_razmetchika(tmp_path):
    sheet = tmp_path / "sheet.tsv"
    sheet.write_text("query_id\tchunk_id\totvet1_celikom\totvet2_vmeste\ttip\tistochnik_v_pule\n"
                     "q1\tA#0001\tнет\tнет\t-\tпункт\n"
                     "q1\tA#0002\tнет\tда\tчасть\t-\n", encoding="utf-8")
    # The old union kept A#0001 as a positive; the labeler rejected it.
    write_qrels(str(tmp_path / "old.tsv"), {"q1": {"A#0001": 2, "A#0002": 1}})
    labels, _, _, _ = L.gather(sheets=(str(sheet),), works=(), optional=())
    qrels = L.labeler_qrels(labels)
    assert qrels == {"q1": {"A#0002": 1}}
    with pytest.raises(SystemExit, match="расхождение"):
        L.check_core(qrels, labels, {"q1"}, path=str(tmp_path / "old.tsv"), sheet=str(sheet))


def test_nepolnaya_razmetka_padaet(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "manifest.json").write_text(json.dumps({"партий": 2}), encoding="utf-8")
    (work / "pairs.tsv").write_text("nomer\tbatch\tquery_id\tchunk_id\n1\t001\tq1\tA#0001\n2\t002\tq2\tA#0002\n",
                                    encoding="utf-8")
    with pytest.raises(SystemExit, match="партий без ответа 2"):
        L.gather(sheets=(), works=(str(work),), optional=())
    _, report, incomplete, todo = L.gather(sheets=(), works=(str(work),), optional=(), allow_partial=True)
    assert todo == 2 and incomplete == {"q1", "q2"}


def test_gruppa_08_ne_rezhetsya():
    base = "кредитная организация формирует резерв ссуде заемщика третьей категории качества сроком"
    rows = [{**question(f"q{i:03d}", 100 + i), "text": f"{base} {tail}"}
            for i, tail in enumerate(("января", "марта"))]
    rows += [question(f"r{i:03d}", i, act=("A", "B")[i % 2]) for i in range(40)]
    for r in rows[2:]:
        r["text"] = f"вопрос номер {r['query_id']} отдельный"
    for r, g in zip(rows, L.near_groups(rows)):
        r["group"] = g
    assert rows[0]["group"] == rows[1]["group"]
    split = L.split_dev_test(rows)
    assert split["q000"] == split["q001"]
