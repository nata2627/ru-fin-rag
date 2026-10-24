"""Объединение выдач обратными рангами.

Перенесено из части А (`src/rufin/retrieval/hybrid.py`). RRF складывает не
оценки, а обратные ранги: оценки BM25 и косинусной близости живут в разных
шкалах, и любая нормировка «в ноль-единицу» зависит от разброса конкретного
запроса. Константа 60 из той же части А, иначе числа сервиса и бенчмарка
перестанут сходиться.
"""

from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60


def rrf(
    runs: Sequence[Sequence[tuple[str, float]]],
    k: int = RRF_K,
    top: int = 100,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Слияние обратными рангами. Вес списка по умолчанию единица у каждого.

    Вес нужен расширению по цитированию: список из соседних по графу актов
    добавляется как подсказка, а не как равноправная ступень, и при весе 1,0 его
    первый фрагмент встает в выдаче рядом с первым фрагментом базовой выдачи.
    Сколько весит подсказка, решает замер, а не код (EXPERIMENTS, S20).
    """
    if weights is None:
        weights = (1.0,) * len(runs)
    if len(weights) != len(runs):
        raise ValueError(f"весов {len(weights)}, выдач {len(runs)}")
    scores: dict[str, float] = {}
    for run, weight in zip(runs, weights, strict=True):
        for rank, (doc_id, _) in enumerate(run, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top]
