"""Метрики сервиса обязаны давать те же числа, что часть А.

Докстринг модуля обещал этот тест, а теста не было: нашел разбор кода. Проверка
идет по замороженным выдачам бенчмарка, для которых числа части А опубликованы
(`docs/06-rezultaty-etap-2.md`: BM25 0,571, плотный поиск 0,548).
"""

from __future__ import annotations

import math

from rufinrag.benchmark import load_qrels, load_run
from rufinrag.metrics import evaluate, mrr_at_k, ndcg_at_k, queries_scored, recall_at_k


def test_frozen_runs_reproduce_part_a():
    qrels = load_qrels()
    assert queries_scored(qrels) == 991
    assert round(evaluate(load_run("bm25"), qrels)["NDCG@10"], 3) == 0.571
    assert round(evaluate(load_run("dense"), qrels)["NDCG@10"], 3) == 0.548


def test_query_without_relevant_is_left_out_of_the_denominator():
    """Отличие от части А, названное явно: там такой запрос входил в MRR нулем.

    Сейчас в разметке таких запросов нет (991 из 991 с релевантными), но если
    они появятся, среднее не должно зависеть от числа вопросов, на которые
    корпус ответить не может.
    """
    qrels = {"a": {"c1": 2}, "b": {"c9": 0}}
    run = {"a": ["c1"], "b": ["c2"]}
    scored = evaluate(run, qrels)
    assert scored["Recall@10"] == 1.0
    assert scored["MRR@10"] == 1.0
    assert queries_scored(qrels) == 1


def test_grades_weigh_in_ndcg():
    assert ndcg_at_k(["c1"], {"c1": 2}) == 1.0
    assert ndcg_at_k(["c2", "c1"], {"c1": 2, "c2": 1}) < 1.0
    assert recall_at_k(["c1"], {"c1": 1, "c2": 1}, 10) == 0.5
    assert mrr_at_k(["x", "c1"], {"c1": 1}) == 0.5
    assert recall_at_k(["c1"], {"c9": 0}, 10) is None
    assert ndcg_at_k(["c1"], {"c9": 0}) is None
    assert not math.isnan(mrr_at_k([], {"c1": 1}))
