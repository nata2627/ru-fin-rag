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
    runs: Sequence[Sequence[tuple[str, float]]], k: int = RRF_K, top: int = 100
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for run in runs:
        for rank, (doc_id, _) in enumerate(run, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top]
