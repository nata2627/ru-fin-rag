"""Числа отчетов совпадают с файлами выводов, из которых они взяты.

Тексты README и docs набирают руками, и они расходятся с данными тихо:
перезаписали вывод другим прогоном, пересчитали, а в тексте остались прежние
цифры. Здесь три такие сверки: согласие судьи, таблица лексического разрыва
и число тестов, которого в README быть не должно.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
КАРТОЧКА = ROOT / "docs" / "DATASET_CARD.md"
РАЗБОР = ROOT / "docs" / "16-razbor-provalov.md"
СУДЬЯ = ROOT / "data" / "results" / "judge_acceptance.json"
РАЗРЫВ = ROOT / "data" / "results" / "lexical-gap-hybrid-rerank.json"


def зап(x: float, знаков: int = 3) -> str:
    """Число в записи документов: десятичная запятая, минус обычный."""
    return f"{x:.{знаков}f}".replace(".", ",")


def инт(m: float, lo: float, hi: float, знаков: int = 3) -> str:
    return f"{зап(m, знаков)} [{зап(lo, знаков)}; {зап(hi, знаков)}]"


@pytest.fixture(scope="module")
def судья() -> dict:
    return json.loads(СУДЬЯ.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def лучший(судья: dict) -> dict:
    return next(s for s in судья["судьи"] if s["судья"] == судья["лучший судья"])


@pytest.mark.parametrize("путь", [README, КАРТОЧКА], ids=["README", "DATASET_CARD"])
def test_soglasie_sudi_tekst_raven_faylu(путь, судья, лучший):
    """Каппа, порог, число пар и признак «приемка не взята» берутся из
    judge_acceptance.json: прежнее число 0,408 уже раз осталось в тексте
    после того, как его заменили измерением против человека."""
    текст = путь.read_text(encoding="utf-8")
    k = лучший["с человеком"]
    assert not судья["приёмка взята"]
    assert инт(k["каппа"], *k["интервал"]) in текст
    assert зап(судья["правило"]["порог каппы с человеком"], 2) in текст
    assert f"{судья['ручная разметка']['пар']} пар" in текст
    assert "не взят" in текст


@pytest.mark.parametrize("путь", [README, КАРТОЧКА], ids=["README", "DATASET_CARD"])
def test_soglasie_sudi_raspredelenie_otsenok(путь, судья):
    """Распределение оценок человека 96 / 44 / 10 в тексте то же, что в файле."""
    оценки = судья["ручная разметка"]["оценки"]
    текст = путь.read_text(encoding="utf-8")
    assert (f"{оценки['0']} нулей, {оценки['1']} единицы, {оценки['2']} двоек"
            in текст.replace("\n", " "))


def test_soglasie_sudi_vzveshennaya_i_sovpadenie(лучший):
    """Взвешенная каппа и «совпало дословно» стоят в таблице карточки и в README."""
    k = лучший["с человеком"]
    for путь in (README, КАРТОЧКА):
        текст = путь.read_text(encoding="utf-8").replace("\n", " ")
        assert зап(k["каппа взвешенная"]) in текст
        assert f"{k['совпало дословно']} из {k['пар']}" in текст


def test_soglasie_sudi_tablitsa_kartochki(судья):
    """Строка каждого судьи в таблице карточки: каппа с интервалом и
    знаменатель пар."""
    текст = КАРТОЧКА.read_text(encoding="utf-8")
    всего = судья["ручная разметка"]["пар"]
    for s in судья["судьи"]:
        k = s["с человеком"]
        строка = next(l for l in текст.splitlines()
                      if l.startswith(f"| `{s['судья']}`"))
        assert инт(k["каппа"], *k["интервал"]) in строка.replace("**", ""), s["судья"]
        assert f"{k['пар']} из {всего}" in строка, s["судья"]


def test_lexical_gap_istochnik_dokumenta():
    """Документ 16 называет источником выдачу гибрида с кросс-энкодером, и файл,
    на который он ссылается, посчитан по ней же (файл без суффикса посчитан
    по выдаче E4 и в этот документ не входит)."""
    разрыв = json.loads(РАЗРЫВ.read_text(encoding="utf-8"))
    текст = РАЗБОР.read_text(encoding="utf-8")
    assert разрыв["выдача для деления живых"].endswith("base__hybrid-rerank.jsonl")
    assert "lexical-gap-hybrid-rerank.json" in текст
    assert "lexical_gap_hybrid-rerank.txt" in текст
    assert (ROOT / "docs" / "raw" / "lexical_gap_hybrid-rerank.txt").exists()


def числа_клетки(клетка: str) -> list[float]:
    return [float(x.replace(",", ".")) for x in re.findall(r"-?\d+,\d+", клетка)]


def есть_тройка(текст: str, ждём: list[float]) -> bool:
    """В тексте есть запись «число [нижняя; верхняя]», равная ждём с допуском 0,001."""
    for m in re.finditer(r"-?\d+,\d+ \[-?\d+,\d+; -?\d+,\d+\]", текст):
        факт = числа_клетки(m.group(0))
        if all(abs(x - y) <= 0.0011 for x, y in zip(факт, ждём)):
            return True
    return False


def test_lexical_gap_tablitsa_urovney():
    """Четыре строки таблицы: число запросов и три доли с интервалами.
    Допуск 0,001: файл хранит четыре знака, текст набран с неокругленных."""
    разрыв = json.loads(РАЗРЫВ.read_text(encoding="utf-8"))
    текст = РАЗБОР.read_text(encoding="utf-8")
    for имя, у in разрыв["уровни"].items():
        строка = next(l for l in текст.splitlines() if l.startswith(f"| {имя} |"))
        клетки = [c.strip() for c in строка.strip("|").split("|")]
        assert клетки[1] == str(у["запросов"]), имя
        for клетка, ключ in zip(клетки[2:], ("со стеммером", "без стеммера",
                                             "первый найденный, со стеммером")):
            ждём = [у[ключ]["mean"], у[ключ]["lo"], у[ключ]["hi"]]
            факт = числа_клетки(клетка)
            assert len(факт) == 3, (имя, ключ)
            assert all(abs(x - y) <= 0.0011 for x, y in zip(факт, ждём)), (имя, ключ)


def test_lexical_gap_raznosti_i_parnyy_test():
    """Непарные разности и парное сравнение с первым найденным, включая знак."""
    разрыв = json.loads(РАЗРЫВ.read_text(encoding="utf-8"))
    текст = РАЗБОР.read_text(encoding="utf-8")
    for r in разрыв["разрыв"]:
        for ключ in ("со стеммером", "без стеммера"):
            d = r[ключ]
            assert есть_тройка(текст, [d["diff"], d["lo"], d["hi"]]), (r["слева"], ключ)
    провалы, остальные, синтетика = разрыв["первый найденный против эталона"]
    assert провалы["diff"] > 0 > остальные["diff"]
    знак = "+" if провалы["diff"] > 0 else ""
    assert (f"{знак}{зап(провалы['diff'])} [{зап(провалы['lo'])}; {зап(провалы['hi'])}], "
            f"p = {зап(провалы['p'])}") in текст
    assert инт(-остальные["diff"], -остальные["hi"], -остальные["lo"]) in текст
    плоский = текст.replace("\n", " ")
    assert f"На тех же {провалы['запросов']} провалах" in плоский
    assert f"На остальных {остальные['запросов']} живых" in плоский


def test_chislo_testov_ne_naberaetsya_rukami():
    """В README нет счета тестов вида «509 тестов»: он устаревает с каждым
    новым файлом (427 стояло при 509 на деле), а пересчитывает его pytest."""
    текст = README.read_text(encoding="utf-8")
    assert not re.search(r"\b\d+\s+тест", текст)
