"""Метрики качества выдачи: Recall@k, MRR@10, NDCG@10.

Формулы повторяют `rufin.metrics` части А: оценки в разметке градуированы (0 это
проверенная нерелевантность, она отличается от отсутствия строки), и NDCG
считается с весом `2^оценка - 1`. Сервис держит свою копию, а не тянет пакет
части А, потому что по этим числам идет ночная регрессия, и она не должна
падать от того, что в чужом репозитории поменяли сигнатуру.

Одно отличие в агрегации, названное явно. Запрос, у которого нет ни одного
релевантного фрагмента, здесь не входит в знаменатель ни одной метрики, а в
части А он входил в среднее MRR нулем. Сейчас расхождение спит: в разметке все
991 запроса имеют релевантные, и числа совпадают до третьего знака. Появятся
такие запросы, и метрика не должна зависеть от того, сколько в наборе вопросов,
на которые корпус ответить не может.

Совпадение проверяется тестом: на замороженной выдаче `run-bm25.jsonl` здесь
должно получаться 0,571 NDCG@10, то же число, что опубликовано в части А после
двух правок токенизации от 23.10.2026: решения 16 (нормализация формы слова
стеммером, линия до него была 0,484, после 0,567) и починки разбора токена,
который рассыпал номер информационного письма.
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


def per_query(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    metric: str = "NDCG@10",
) -> tuple[list[str], list[float]]:
    """Значение метрики по каждому запросу: без них нет ни интервала, ни пары."""
    pick = {
        "NDCG@10": lambda ranked, rel: ndcg_at_k(ranked, rel),
        "MRR@10": lambda ranked, rel: mrr_at_k(ranked, rel),
        "Recall@10": lambda ranked, rel: recall_at_k(ranked, rel, DEPTH),
    }[metric]
    ids: list[str] = []
    values: list[float] = []
    for query_id in sorted(qrels):
        value = pick(list(run.get(query_id, ())), qrels[query_id])
        if value is not None:
            ids.append(query_id)
            values.append(value)
    return ids, values


def bootstrap_ci(values: Sequence[float], runs: int = 10000, seed: int = 0) -> dict[str, float]:
    """Интервал среднего бутстрэпом по запросам, как в части А."""
    import numpy as np

    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(sample, size=(runs, sample.size), replace=True).mean(axis=1)
    return {
        "mean": round(float(sample.mean()), 4),
        "lo": round(float(np.quantile(means, 0.025)), 4),
        "hi": round(float(np.quantile(means, 0.975)), 4),
    }


def paired_diff_ci(
    better: Sequence[float], worse: Sequence[float], runs: int = 10000, seed: int = 0
) -> dict[str, float]:
    """Разница на одних и тех же запросах: парный бутстрэп и перестановочный тест.

    Парный вариант обязателен: конфигурации меряются на общем наборе, их ошибки
    скоррелированы, и независимые интервалы завысили бы неопределенность.
    """
    import numpy as np

    diff = np.asarray(better, dtype=float) - np.asarray(worse, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(runs, diff.size))].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(runs, diff.size))
    permuted = (diff * signs).mean(axis=1)
    return {
        "mean": round(float(diff.mean()), 4),
        "lo": round(float(np.quantile(means, 0.025)), 4),
        "hi": round(float(np.quantile(means, 0.975)), 4),
        "p": round(float((np.abs(permuted) >= abs(diff.mean())).mean()), 4),
    }


def queries_scored(qrels: Mapping[str, Mapping[str, int]]) -> int:
    """Знаменатель метрик: сколько запросов имеют хотя бы один релевантный фрагмент."""
    return sum(1 for rel in qrels.values() if any(grade > 0 for grade in rel.values()))
