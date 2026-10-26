"""Согласие разметчиков и то, на чём оно держится.

Каппа — единственное число, которым в проекте измеряется качество самой
разметки, и ошибиться в ней значит выдать разметку судьи за проверенную.
Отдельно проверяется, что инструкция человеку и подсказка судье — один
и тот же текст: разойдись они, каппа померила бы разницу в инструкциях.

С 24.10.2026 оценка не ставится, а собирается из двух ответов, поэтому
проверяется и сборка, и разбор ответа судьи. Разбор обязан возвращать
«ответа нет» вместо догадки: пара без разбора идёт в счётчик, а не
в оценку по умолчанию, иначе нераспознанные ответы тихо станут нулями.
"""
from __future__ import annotations

import pathlib

import pytest

from rufin.annotation import (
    ANSWER_FORMAT,
    RUBRIC,
    cohen_kappa,
    kappa_interval,
    kappa_verdict,
    parse_answers,
    scale_of,
    stratified_sample,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_полное_совпадение_даёт_единицу():
    assert cohen_kappa([0, 1, 2, 0], [0, 1, 2, 0]) == pytest.approx(1.0)


def test_согласие_на_уровне_случайного_даёт_ноль():
    """Обе разметки ставят одно и то же значение всем парам.

    Совпадение стопроцентное, но никакой информации в нём нет: каппа
    обязана это увидеть и вернуть ноль, а не единицу.
    """
    assert cohen_kappa([0] * 10, [0] * 10) == pytest.approx(1.0)
    # разметчики независимы и ставят метки в тех же долях
    a = [0, 0, 1, 1]
    b = [1, 1, 0, 0]
    assert cohen_kappa(a, b) < 0


def test_взвешенная_каппа_мягче_к_соседним_оценкам():
    """«2 против 1» — промах меньший, чем «2 против 0»."""
    близко = cohen_kappa([2, 2, 1, 1, 0, 0], [2, 1, 1, 2, 0, 0], weighted=True)
    далеко = cohen_kappa([2, 2, 1, 1, 0, 0], [2, 0, 1, 2, 0, 0], weighted=True)
    assert близко > далеко


def test_невзвешенная_каппа_не_различает_насколько_промах_велик():
    близко = cohen_kappa([2, 2, 1, 1, 0, 0], [2, 1, 1, 2, 0, 0])
    далеко = cohen_kappa([2, 2, 1, 1, 0, 0], [2, 0, 1, 2, 0, 0])
    assert близко == pytest.approx(далеко)


def test_разная_длина_списков_это_ошибка():
    with pytest.raises(ValueError):
        cohen_kappa([0, 1], [0])


def test_словесная_шкала_не_врёт_про_ноль():
    assert kappa_verdict(-0.1) == "хуже случайного"
    assert kappa_verdict(0.85) == "почти полное согласие"


def test_в_выборку_попадают_все_оценки_а_не_только_нули():
    пул = ([{"query_id": f"q{i}", "chunk_id": "c", "sud": 0} for i in range(300)]
           + [{"query_id": f"r{i}", "chunk_id": "c", "sud": 1} for i in range(20)]
           + [{"query_id": f"s{i}", "chunk_id": "c", "sud": 2} for i in range(10)])
    выборка = stratified_sample(пул, "sud", 30)
    assert len(выборка) == 30
    assert {r["sud"] for r in выборка} == {0, 1, 2}


def test_выборка_воспроизводится():
    пул = [{"query_id": f"q{i}", "chunk_id": "c", "sud": i % 3} for i in range(90)]
    assert stratified_sample(пул, "sud", 12) == stratified_sample(пул, "sud", 12)


def test_инструкция_в_документации_совпадает_с_подсказкой_судье():
    guide = (ROOT / "docs" / "ANNOTATION_GUIDE.md").read_text(encoding="utf-8")
    assert RUBRIC in guide, "docs/ANNOTATION_GUIDE.md разошёлся с annotation.RUBRIC"


def test_сборка_оценки_повторяет_правило_из_инструкции():
    assert scale_of(True, False) == 2
    assert scale_of(True, True) == 2
    assert scale_of(False, True) == 1
    assert scale_of(False, False) == 0


def test_правило_сборки_напечатано_в_инструкции():
    for строка in ('вопрос 1 «да»', 'вопрос 2 «да»', 'оценка 2', 'оценка 1',
                   'оценка 0'):
        assert строка in RUBRIC, f"в инструкции нет «{строка}»"


def test_разбор_трёх_строк_ответа():
    assert parse_answers("1: нет\n2: да\nтип: общая") == (False, True, "общая")
    assert parse_answers("1: да\n2: -\nтип: -") == (True, False, "-")
    assert parse_answers("1: нет\n2: нет\nтип: -") == (False, False, "-")


def test_тип_без_единицы_отбрасывается():
    """Тип имеет смысл только там, где оценка собралась в единицу. Судья
    иногда ставит его и к нулю; такой тип в разметку не идёт."""
    assert parse_answers("1: нет\n2: нет\nтип: часть") == (False, False, "-")
    assert parse_answers("1: да\n2: -\nтип: общая") == (True, False, "-")


def test_неразобранный_ответ_это_не_ноль():
    assert parse_answers("") is None
    assert parse_answers("затрудняюсь ответить") is None
    assert parse_answers("1: нет") is None, "второй ответ обязателен при «нет»"
    assert parse_answers("2: да\nтип: общая") is None


def test_разбор_переживает_болтливость_судьи():
    ответ = "Оценка:\n1: нет\n2: да\nтип: часть\nпояснение не требуется"
    assert parse_answers(ответ) == (False, True, "часть")


def test_интервал_каппы_накрывает_точечное_значение():
    судья = [0, 1, 2] * 20
    человек = [0, 1, 1] * 20
    низ, верх = kappa_interval(судья, человек, reps=400)
    assert низ <= cohen_kappa(судья, человек) <= верх


def test_меньшая_выборка_даёт_более_широкий_интервал():
    """Это не свойство бутстрэпа, а ответ на вопрос «хватит ли 100 пар
    вместо 150»: цена сокращения называется числом, а не на глаз."""
    судья = [0, 0, 1, 1, 2, 2] * 25
    человек = [0, 1, 1, 2, 2, 0] * 25
    шире = kappa_interval(судья, человек, size=100, reps=600)
    уже = kappa_interval(судья, человек, size=150, reps=600)
    assert (шире[1] - шире[0]) > (уже[1] - уже[0])


def test_формат_ответа_судьи_не_попал_в_инструкцию():
    """Инструкция одна, а формат ответа разный: человек пишет в колонки,
    судья строками. Попади формат в инструкцию, человек читал бы указание
    не про себя."""
    assert ANSWER_FORMAT not in RUBRIC


def test_otvet_bez_nomerov_razbiraetsya():
    """Так отвечает Qwen2.5-32B-AWQ: номера строк он не ставит.

    24.10.2026 на этом не разобралось 150 ответов из 150, то есть весь прогон
    судьи. Нумерация это оформление, а не ответ.
    """
    assert parse_answers("нет да тип: часть ответа") == (False, True, "часть")
    assert parse_answers("нет нет тип: -") == (False, False, "-")
    assert parse_answers("да - -") == (True, False, "-")


def test_poryadok_otvetov_bez_nomerov_znachit():
    """Первое «да/нет» это первый вопрос, второе второй. Иначе оценка перевернется."""
    assert parse_answers("да нет тип: -")[0] is True
    assert parse_answers("нет да тип: иное")[0] is False


def test_odin_otvet_bez_nomerov_ne_razbiraetsya():
    """Половина ответа это не ответ: пара идет в счетчик неразобранного."""
    assert parse_answers("да") is None
    assert parse_answers("ответить не могу") is None


def test_slova_iz_tipa_ne_schitayutsya_otvetami():
    """«общая норма» и «часть ответа» стоят после слова тип и в ответы не идут."""
    assert parse_answers("да - тип: общая норма") == (True, False, "-")


def test_nomerovannyy_otvet_odnoy_strokoy_razbiraetsya():
    """Так ответ лежит в листе после склейки строк, и так его читает пересчет."""
    assert parse_answers("1: нет 2: да тип: часть ответа") == (False, True, "часть")
    assert parse_answers("1: нет 2: нет тип: -") == (False, False, "-")
    assert parse_answers("1: да 2: - тип: -") == (True, False, "-")
