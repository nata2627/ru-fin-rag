"""Замер качества ответа: что в нем ломается тихо.

Сам прогон идет отдельной командой и стоит шесть минут
(`scripts/measure_answer_quality.py`). Здесь проверяется то, чего в выводе не
видно.

Первое: знаменатель. Правило S24 говорит «выданные ответы», и если в
знаменатель попадут все запросы, доля корректных ссылок станет тихо ниже ровно
на отказы, то есть замер начнет наказывать слой ответа за то, за что его уже
меряет S21.

Второе: оценка человека собирается из двух его ответов, а не берется цифрой.
Перепутай колонки, и «отвечает вместе с другими» станет полным ответом.

Третье: человеческий раздел не выдумывается. Файлов части А на машине может не
быть, и тогда раздел либо переносится из прежнего отчета с пометкой, либо
прогон отказывается считать.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "measure_answer_quality", ROOT / "scripts/measure_answer_quality.py"
)
measure = importlib.util.module_from_spec(spec)
sys.modules["measure_answer_quality"] = measure
spec.loader.exec_module(measure)

REPORT = ROOT / "results/answer-quality.json"


def test_denominator_is_answers_not_queries():
    tally = Counter(
        {"queries": 10, "refused": 2, "answered": 8, "strict": 6, "citations": 40, "act": 7}
    )
    out = measure.report_of(tally)
    assert out["ссылка на релевантный фрагмент"] == {"hit": 6, "of": 8, "share": 0.75}
    assert out["ссылка на релевантный акт"]["of"] == 8
    assert out["citations_per_answer"] == 5.0


def test_empty_report_says_none_instead_of_zero():
    """Ноль ответов это «не измерено», а не «доля ноль»."""
    out = measure.report_of(Counter({"queries": 3, "refused": 3}))
    assert out["ссылка на релевантный фрагмент"]["share"] is None
    assert out["citations_per_answer"] is None


def test_human_grade_follows_the_two_questions():
    assert measure.human_grade({"otvet1_celikom": "да", "otvet2_vmeste": "-"}) == 2
    assert measure.human_grade({"otvet1_celikom": "нет", "otvet2_vmeste": "да"}) == 1
    assert measure.human_grade({"otvet1_celikom": "нет", "otvet2_vmeste": "нет"}) == 0
    assert measure.human_grade({"otvet1_celikom": " нет ", "otvet2_vmeste": " да "}) == 1


def test_human_section_is_absent_without_part_a(tmp_path):
    assert measure.human_section(tmp_path, {}) is None


def test_human_table_counts_both_directions(tmp_path):
    sheet = tmp_path / measure.HUMAN_SHEET
    sheet.parent.mkdir(parents=True)
    rows = [
        ("1", "да", "-", "q1", "c1"),
        ("2", "нет", "да", "q1", "c2"),
        ("3", "нет", "нет", "q2", "c3"),
    ]
    sheet.write_text(
        "nomer\totvet1_celikom\totvet2_vmeste\tquery_id\tchunk_id\n"
        + "".join("\t".join(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = tmp_path / measure.JUDGE_REPORT
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "правило": {"порог каппы с человеком": 0.4, "записано": "до данных"},
                "ручная разметка": {"кто разметил": "человек"},
                "судьи": [
                    {
                        "судья": measure.ACCEPTED_JUDGE,
                        "с человеком": {"каппа": 0.236, "интервал": [0.1, 0.3], "пар": 150},
                        "приёмка": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Эталон щедрее на паре q2/c3 (оценка 2 против нуля человека) и скромнее на
    # q1/c2 (ноль против единицы): обе стороны обязаны считаться, иначе отчет
    # покажет только ту ошибку, которую ждали.
    qrels = {"q1": {"c1": 2}, "q2": {"c3": 2}}
    out = measure.human_section(tmp_path, qrels)
    assert out["пар"] == 3
    assert out["запросов"] == 2
    assert out["эталон выше человека"] == {"hit": 1, "of": 3, "share": 0.3333}
    assert out["человек выше эталона"] == {"hit": 1, "of": 3, "share": 0.3333}
    assert out["полный ответ по эталону"] == {
        "пар": 2,
        "человек согласился полностью": 1,
        "человек увидел хотя бы часть": 1,
        "человек ответа не увидел": 1,
    }
    assert out["судья"]["применяется к ответам"] is False


def test_report_holds_together():
    if not REPORT.exists():
        pytest.skip("нет results/answer-quality.json, собрать: make answer-quality")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["rule"]["written_before"].endswith("S24")
    assert report["labeled_queries"] == 991
    for name, block in report["configs"].items():
        strict = block["ссылка на релевантный фрагмент"]["hit"]
        assert block["answered"] + block["refused"] == block["queries"] == 991, name
        # Мера по акту прощает соседний пункт того же документа, поэтому она не
        # может быть строже, а полный ответ по эталону это подмножество
        # релевантного. Разойдись эти неравенства, и считаются разные вещи.
        assert block["ссылка на релевантный акт"]["hit"] >= strict, name
        assert block["ссылка на фрагмент с полным ответом"]["hit"] <= strict, name
        assert block["первая ссылка релевантна"]["hit"] <= strict, name
        assert sum(v["queries"] for v in block["by_subsample"].values()) == 991, name
    cross = report["cross_support"]
    # Вариант из находки S24 считает опору лексической выдачей, поэтому отказов
    # у него обязано быть столько же, сколько у лексической связки: разойдись
    # это, и замер считает опору не тем, чем обещает.
    assert cross["refused"] == report["configs"]["bm25"]["refused"]
    assert cross["queries"] == 991
    assert cross["лишних поисков сервису"] == 0
    assert cross["опора не попала в цитаты"]["of"] == cross["answered"]
    # Две поправки к тому же пути ставят опорный фрагмент в цитаты, поэтому
    # набор цитат у них один и тот же, а отличается только порядок. Разойдись
    # строгая мера между ними, и замер считает разные наборы, а не порядок.
    first, last = cross["с опорой в цитатах"], cross["с опорой последней цитатой"]
    for block in (first, last):
        assert block["опора не попала в цитаты"]["hit"] == 0
        assert block["refused"] == cross["refused"]
    assert first["ссылка на релевантный фрагмент"] == last["ссылка на релевантный фрагмент"]
    assert first["первая ссылка релевантна"] != last["первая ссылка релевантна"]
    human = report["human"]
    assert human["пар"] == sum(human["таблица"].values())
    assert human["судья"]["приемка"] is False
    assert human["судья"]["каппа с человеком"] < human["судья"]["порог"]
