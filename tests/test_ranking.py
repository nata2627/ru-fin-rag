"""Отбор лучших k: ничьи, края, испорченные оценки.

Эти проверки появились после того, как прогон в Linux CI разошелся с маком на
равных оценках BM25. Остальные края (k вне диапазона, NaN) нашел разбор кода, в
прогоне они не срабатывали.
"""

from __future__ import annotations

import numpy as np
import pytest

from rufinrag.retrieval.ranking import top_k


def test_ties_are_broken_by_index():
    scores = np.array([1.0, 5.0, 5.0, 5.0, 2.0])
    assert list(top_k(scores, 3)) == [1, 2, 3]


def test_order_is_by_score_then_index():
    scores = np.array([0.1, 0.9, 0.5, 0.9])
    assert list(top_k(scores, 4)) == [1, 3, 2, 0]


def test_k_larger_than_corpus_returns_everything():
    scores = np.array([0.2, 0.4, 0.3])
    assert list(top_k(scores, 10)) == [1, 2, 0]


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="больше нуля"):
        top_k(np.array([0.1, 0.2]), 0)
    with pytest.raises(ValueError, match="больше нуля"):
        top_k(np.array([0.1, 0.2]), -1)


def test_nan_scores_do_not_empty_the_window():
    """Оценка, которой нет, худшая. Раньше окно схлопывалось и выдача была пустой."""
    scores = np.array([np.nan, 0.5, np.nan, 0.7, 0.1])
    found = top_k(scores, 3)
    assert list(found) == [3, 1, 4]
    assert len(top_k(np.full(100, np.nan), 5)) == 5
