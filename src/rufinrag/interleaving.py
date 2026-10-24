"""Интерливинг команд: слияние двух выдач и перевес по одному запросу.

Жило в `scripts/plan_sample_size.py`, где считались только знаменатели, и
модуль перевеса брался нарочно: правило остановки писалось до того, как кто-то
увидит, какая связка выигрывает. Правило закоммичено, поэтому знак теперь
нужен, и слияние переехало сюда, чтобы у задач 6.1 и 6.2 оно было одно.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

DEPTH = 10


def merge_team_draft(
    left: Sequence[str], right: Sequence[str], rng: random.Random, depth: int = DEPTH
) -> tuple[list[str], list[int]]:
    """Слияние двух выдач интерливингом команд: список и хозяин каждой позиции.

    Правило из исходной работы: следующую позицию берет та команда, у которой
    позиций меньше, а при равенстве монета. Поэтому ни одна конфигурация не
    получает систематического преимущества в начале списка, и предпочтение
    нельзя будет объяснить тем, кто стоял первым.
    """
    merged: list[str] = []
    owners: list[int] = []
    taken: set[str] = set()
    counts = [0, 0]
    cursors = [0, 0]
    lists = (left, right)
    while len(merged) < depth:
        for side in (0, 1):
            while cursors[side] < len(lists[side]) and lists[side][cursors[side]] in taken:
                cursors[side] += 1
        has = [cursors[side] < len(lists[side]) for side in (0, 1)]
        if not any(has):
            break
        if not (has[0] and has[1]):
            side = 0 if has[0] else 1
        elif counts[0] < counts[1]:
            side = 0
        elif counts[1] < counts[0]:
            side = 1
        else:
            side = 0 if rng.random() < 0.5 else 1
        chunk_id = lists[side][cursors[side]]
        taken.add(chunk_id)
        merged.append(chunk_id)
        owners.append(side)
        counts[side] += 1
    return merged, owners


def credits(
    left: Sequence[str], right: Sequence[str], rel: Mapping[str, int], rng: random.Random
) -> tuple[int, int]:
    """Сколько размеченных релевантных фрагментов привела каждая сторона.

    За клик принято попадание размеченного релевантного фрагмента в первую
    десятку слитого списка. Это допущение офлайн интерливинга: предполагается,
    что пользователь нажал бы ровно на то, что разметка считает релевантным.
    На живых людях оно не проверено, потому что живых людей у сервиса нет.
    """
    merged, owners = merge_team_draft(left, right, rng)
    credit = [0, 0]
    for chunk_id, side in zip(merged, owners, strict=True):
        if rel.get(chunk_id, 0) > 0:
            credit[side] += 1
    return credit[0], credit[1]


def margin(
    left: Sequence[str], right: Sequence[str], rel: Mapping[str, int], rng: random.Random
) -> int:
    """Перевес правой стороны над левой: больше нуля значит выиграла правая."""
    first, second = credits(left, right, rel, rng)
    return second - first
