"""Числа приёмки судей в карточке набора берутся из файла замера.

Это главное число всего разбора разметки, и оно же самое заманчивое для правки
руками: порог 0,40 объявлен до данных, а каппа вышла ниже. Проверка сверяет
карточку с `data/results/judge_acceptance.json` и падает, если они разойдутся.
"""
from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ОТЧЁТ = ROOT / "data" / "results" / "judge_acceptance.json"
КАРТОЧКА = ROOT / "docs" / "DATASET_CARD.md"
ПОРОГ = 0.40


@pytest.fixture
def отчёт() -> dict:
    if not ОТЧЁТ.exists():
        pytest.skip("приёмка не считана: нужен `python scripts/judge_acceptance.py`")
    return json.loads(ОТЧЁТ.read_text(encoding="utf-8"))


def зп(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


@pytest.mark.data
def test_porog_i_ruchnaya_razmetka(отчёт):
    assert отчёт["правило"]["порог каппы с человеком"] == ПОРОГ
    ручная = отчёт["ручная разметка"]
    assert ручная["пар"] == 150
    assert sum(ручная["оценки"].values()) == 150
    текст = КАРТОЧКА.read_text(encoding="utf-8")
    assert f"{ручная['оценки']['0']} нулей" in текст
    assert f"{ручная['оценки']['1']} единицы" in текст
    assert f"{ручная['оценки']['2']} двоек" in текст


@pytest.mark.data
def test_kazhdyy_sudya_stoit_v_kartochke(отчёт):
    текст = КАРТОЧКА.read_text(encoding="utf-8")
    for судья in отчёт["судьи"]:
        с_человеком = судья["с человеком"]
        строка = (
            f"| `{судья['судья']}` | {судья['машина']} | {с_человеком['пар']} из 150 | "
            f"**{зп(с_человеком['каппа'])}** [{зп(с_человеком['интервал'][0])}; "
            f"{зп(с_человеком['интервал'][1])}] |"
        )
        assert строка in текст, f"в карточке нет строки судьи {судья['судья']}"


@pytest.mark.data
def test_verdikt_sleduet_iz_chisel(отчёт):
    """Вердикт не пишется словами мимо чисел: он пересчитывается здесь же."""
    for судья in отчёт["судьи"]:
        взято = (
            (судья["с человеком"] or {}).get("каппа", -1) >= ПОРОГ
            and (судья.get("с собой") or {}).get("каппа", -1)
            >= отчёт["правило"]["порог согласия двух прогонов"]
            and судья["стек вывода записан"]
        )
        assert судья["приёмка"] == взято, судья["судья"]
    assert отчёт["приёмка взята"] == any(r["приёмка"] for r in отчёт["судьи"])
    # Следствие отказа названо в карточке, а не только в журнале сессии.
    assert "пул глубины 50 (40 134 пары) не размечается" in КАРТОЧКА.read_text(encoding="utf-8")


@pytest.mark.data
def test_luchshiy_sudya_eto_luchshiy_po_chislu(отчёт):
    лучший = max(отчёт["судьи"], key=lambda r: (r["с человеком"] or {}).get("каппа", -1))
    assert отчёт["лучший судья"] == лучший["судья"]
    assert not лучший["приёмка"], "порог взят: карточку и вывод надо переписать"
