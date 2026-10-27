"""Урезание корпуса в замере забывания: размер задается снаружи и пишется в итог.

Замер 27.10.2026 пришлось считать на 2 000 документах вместо 20 000, потому что
процессор арендованной машины кодировал 1,3 текста в секунду. Урезание меняет
саму задачу: дистракторов меньше, значит NDCG выше у всех моделей сразу.
Поэтому размер обязан быть параметром, а не константой модуля, иначе число из
прогона невозможно повторить кодом репозитория.

Эталонные документы входят в урезанный корпус всегда. Если бы они терялись,
метрика упала бы у всех моделей одинаково и сказала бы о забывании ровно
ничего, причем выглядело бы это как удачный замер.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

import gpu_forget as F  # noqa: E402


def задача(документов: int = 500, запросов: int = 20) -> tuple[dict, dict, dict]:
    corpus = {f"d{i}": {"title": "", "text": f"текст документа {i}"}
              for i in range(документов)}
    queries = {f"q{i}": f"вопрос {i}" for i in range(запросов)}
    qrels = {f"q{i}": {f"d{i * 7}": 1} for i in range(запросов)}
    return corpus, queries, qrels


def test_razmer_korpusa_beretsya_iz_parametra() -> None:
    corpus, queries, qrels = задача()
    for cap in (50, 120, 300):
        док, зап, разметка = F.prepare(corpus, queries, qrels, 20, cap=cap)
        assert len(док) == cap
        assert len(зап) == 20


def test_etalony_vsegda_v_urezannom_korpuse() -> None:
    corpus, queries, qrels = задача()
    док, зап, разметка = F.prepare(corpus, queries, qrels, 20, cap=25)
    лежат = {d["id"] for d in док}
    эталоны = {d for q in разметка for d, g in разметка[q].items() if g > 0}
    assert эталоны <= лежат
    assert эталоны


def test_odin_seed_odin_korpus() -> None:
    corpus, queries, qrels = задача()
    первый = F.prepare(corpus, queries, qrels, 20, cap=100)
    второй = F.prepare(corpus, queries, qrels, 20, cap=100)
    assert [d["id"] for d in первый[0]] == [d["id"] for d in второй[0]]


def test_measure_prinimaet_cap() -> None:
    """Параметр должен быть у measure, иначе прогон правит его обёрткой.

    Замер 27.10.2026 так и шёл: двухстрочная подмена `F.prepare` в скретчпаде.
    Число из такого прогона кодом репозитория не повторяется.
    """
    import inspect

    параметры = inspect.signature(F.measure).parameters
    assert "cap" in параметры
    assert параметры["cap"].default == F.CORPUS_CAP
