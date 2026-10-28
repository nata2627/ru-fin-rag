"""Таблица навыков: каждая строка проверяема по ссылке за минуту.

Требование постановки к этой таблице одно, и оно про читателя, а не про
формат: работодатель открывает ссылку и за минуту видит, где навык написан и
где посчитано его число. Путь в обратных кавычках этому не отвечает, его надо
скопировать и найти руками, поэтому пути стоят ссылками. Ссылка, ведущая в
пустоту, хуже отсутствующей, значит проверять надо существование цели.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

КОРЕНЬ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(КОРЕНЬ / "scripts"))

import skills_table  # noqa: E402

ССЫЛКА = re.compile(r"\[`([^`]+)`\]\(([^)]+)\)")


def test_u_kazhdoy_stroki_est_ssylka_i_chislo() -> None:
    for навык, где, число, файл in skills_table.строки():
        assert ССЫЛКА.search(skills_table.ссылками(где)), навык
        assert ССЫЛКА.search(skills_table.ссылками(файл)), навык
        assert re.search(r"\d", число), навык


def test_vse_ssylki_vedut_na_sushchestvuyushchie_puti() -> None:
    for навык, где, файл in ((н, г, ф) for н, г, _, ф in skills_table.строки()):
        for столбец in (skills_table.ссылками(где), skills_table.ссылками(файл)):
            for подпись, цель in ССЫЛКА.findall(столбец):
                assert (КОРЕНЬ / цель.rstrip("/")).exists(), f"{навык}: {цель}"
                assert подпись in цель or цель.endswith(подпись), f"{навык}: {подпись}"


def test_korotkoe_imya_dostraivaetsya_sosednim_katalogom() -> None:
    """`text.py` рядом с `src/.../bm25.py` это файл того же каталога.

    Проверка именно на это: без достройки каталогом короткое имя осталось бы
    без ссылки, то есть строка таблицы перестала бы отвечать требованию.
    """
    вышло = skills_table.ссылками("`src/rufinrag/retrieval/bm25.py`, `text.py`")
    assert "(src/rufinrag/retrieval/text.py)" in вышло


def test_nesushchestvuyushchii_put_ostaetsya_bez_ssylki() -> None:
    вышло = skills_table.ссылками("`src/rufinrag/no_such_file.py`")
    assert вышло == "`src/rufinrag/no_such_file.py`"
