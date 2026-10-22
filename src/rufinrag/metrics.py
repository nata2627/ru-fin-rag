"""Метрики качества выдачи: Recall@k, MRR@10, NDCG@10.

Формулы повторяют `rufin.metrics` части А намеренно, вплоть до обращения с
непроверенными парами: оценки в разметке градуированы (0 это проверенная
нерелевантность, она отличается от отсутствия строки), и NDCG считается с
весом `2^оценка - 1`. Сервис держит свою копию, а не тянет пакет части А,
потому что по этим числам идет ночная регрессия, и она не должна падать от
того, что в чужом репозитории поменяли сигнатуру.

Совпадение проверяется тестом: на замороженной выдаче `run-bm25.jsonl` здесь
должно получаться 0,484 NDCG@10, то же число, что опубликовано в части А.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

DEPTH = 10


def recall_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float | None:
    good = {chunk_id for chunk_id, grade in rel.items() if grade > 0}
    if not good:
        return None
    return len(good & set(ranked[:k])) / len(good)


def mrr_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = DEPTH) -> float:
    for position, chunk_id in enumerate(ranked[:k], start=1):
        if rel.get(chunk_id, 0) > 0:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int = DEPTH) -> float | None:
    dcg = sum(
        (2 ** rel.get(chunk_id, 0) - 1) / math.log2(position + 1)
        for position, chunk_id in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(position + 1) for position, grade in enumerate(ideal, 1))
    return dcg / idcg if idcg else None


def evaluate(
    run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]]
) -> dict[str, float]:
    """Средние по тем запросам, у которых есть хотя бы один релевантный фрагмент.

    Запросы без релевантных в знаменатель не попадают: иначе метрика зависит от
    того, сколько в наборе вопросов, на которые корпус ответить не может.
    """
    scored: dict[str, list[float]] = {
        "Recall@1": [],
        "Recall@5": [],
        "Recall@10": [],
        "MRR@10": [],
        "NDCG@10": [],
    }
    for query_id, rel in qrels.items():
        ranked = list(run.get(query_id, ()))
        for k in (1, 5, 10):
            value = recall_at_k(ranked, rel, k)
            if value is not None:
                scored[f"Recall@{k}"].append(value)
        ndcg = ndcg_at_k(ranked, rel)
        if ndcg is not None:
            scored["NDCG@10"].append(ndcg)
            scored["MRR@10"].append(mrr_at_k(ranked, rel))
    return {name: round(sum(values) / len(values), 4) for name, values in scored.items() if values}


def queries_scored(qrels: Mapping[str, Mapping[str, int]]) -> int:
    """Знаменатель метрик: сколько запросов имеют хотя бы один релевантный фрагмент."""
    return sum(1 for rel in qrels.values() if any(grade > 0 for grade in rel.values()))
