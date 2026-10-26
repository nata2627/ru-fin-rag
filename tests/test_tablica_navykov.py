"""Перезапись таблицы навыков в README.

Два числа таблицы меняет сам ночной прогон: счетчик своих прогонов и запись
приемки. Поэтому прогон вписывает таблицу в README перед коммитом, и важно,
что замена берет ровно таблицу: заголовок раздела и текст под ней остаются,
а строки прежней таблицы не остаются ниже новой.
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("skills_table", ROOT / "scripts" / "skills_table.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

README = (
    "## Таблица навыков\n\nПояснение.\n\n"
    "| Навык | Где в коде | Чем подтвержден | Где посчитано |\n"
    "|---|---|---|---|\n"
    "| Старый навык | `a.py` | 8 прогонов | `results/a.json` |\n"
    "\nТекст после таблицы.\n"
)
ТАБЛИЦА = (
    "| Навык | Где в коде | Чем подтвержден | Где посчитано |\n"
    "|---|---|---|---|\n"
    "| Новый навык | `b.py` | 9 прогонов | `results/b.json` |\n"
)


def test_zamena_beret_rovno_tablicu():
    стало = T.вписать(README, ТАБЛИЦА)
    assert "Старый навык" not in стало
    assert "| Новый навык | `b.py` | 9 прогонов | `results/b.json` |" in стало
    assert стало.startswith("## Таблица навыков\n\nПояснение.\n\n")
    assert стало.endswith("\nТекст после таблицы.\n")


def test_povtornaya_zapis_nichego_ne_menyaet():
    один = T.вписать(README, ТАБЛИЦА)
    assert T.вписать(один, ТАБЛИЦА) == один


def test_tablica_repozitoriya_sovpadaet_s_readme():
    текст = (ROOT / "README.md").read_text(encoding="utf-8")
    assert T.вписать(текст, T.таблица()) == текст
