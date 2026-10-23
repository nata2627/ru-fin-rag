"""Подъем мелкого фрагмента к родительскому куску базовой нарезки.

Нужно для сравнения нарезок по уровню метрики, а не по разрыву между
методами. Прямое сравнение уровней запрещено (`scripts/chunking_effect.py`):
при дроблении цель уменьшается, и попасть в нее труднее обоим методам.
Если же искать по мелким фрагментам, а в выдачу отдавать родительский
фрагмент базовой нарезки, единица оценки у обеих ветвей одна и та же,
разметка одна и та же, и уровни сравнимы.

Родитель определяется по символьным границам в тексте акта: у каждого
фрагмента есть `act_id`, `char_start` и `char_end`, и обе нарезки режут один
и тот же текст. Родитель это фрагмент базовой нарезки того же акта с
наибольшим пересечением. Базовые фрагменты идут с перекрытием, поэтому
пересечений бывает несколько, и «наибольшее» здесь не формальность:
при ничьей берется фрагмент, который в акте стоит раньше.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    chunk_id: str
    start: int
    end: int


class ParentIndex:
    """Соответствие «фрагмент любой нарезки -> родитель базовой нарезки»."""

    def __init__(self, parents: dict[str, list[Span]]):
        self.by_act: dict[str, list[Span]] = {}
        self.starts: dict[str, list[int]] = {}
        self.widest: dict[str, int] = {}
        for act, spans in parents.items():
            ordered = sorted(spans, key=lambda s: (s.start, s.end))
            self.by_act[act] = ordered
            self.starts[act] = [s.start for s in ordered]
            self.widest[act] = max((s.end - s.start for s in ordered), default=0)

    @classmethod
    def from_records(cls, records) -> "ParentIndex":
        parents: dict[str, list[Span]] = {}
        for rec in records:
            parents.setdefault(rec["act_id"], []).append(
                Span(rec["chunk_id"], rec["char_start"], rec["char_end"]))
        return cls(parents)

    def parent_of(self, act_id: str, start: int, end: int) -> str | None:
        spans = self.by_act.get(act_id)
        if not spans:
            return None
        starts = self.starts[act_id]
        # кандидаты: все, чье начало левее конца искомого куска, но не дальше
        # самого широкого родителя влево. Перебирать весь акт нельзя: в самом
        # большом из них 2314 базовых фрагментов и впятеро больше мелких.
        hi = bisect.bisect_right(starts, end)
        lo = bisect.bisect_left(starts, start - self.widest[act_id])
        best, best_overlap = None, 0
        for span in spans[lo:hi]:
            overlap = min(end, span.end) - max(start, span.start)
            if overlap > best_overlap:
                best, best_overlap = span, overlap
        if best is not None:
            return best.chunk_id
        # пересечения нет вовсе: такое бывает у куска, который целиком попал
        # в текст, не покрытый базовой нарезкой. Берем ближайшего по началу,
        # иначе позиция выдачи просто исчезнет и ветвь получит преимущество.
        place = min(bisect.bisect_left(starts, start), len(spans) - 1)
        nearest = min(spans[max(0, place - 1):place + 2],
                      key=lambda s: abs(s.start - start))
        return nearest.chunk_id

    def lift(self, ranked: list[str], meta: dict[str, tuple[str, int, int]],
             top: int) -> list[str]:
        """Поднять выдачу к родителям, сняв повторы и сохранив порядок.

        Повторы снимаются: несколько мелких фрагментов одного родителя это
        одна позиция в выдаче, а не несколько. Из-за этого список короче
        исходного, поэтому читать его надо с глубины, достаточной чтобы
        набрать `top` родителей.
        """
        out: list[str] = []
        seen: set[str] = set()
        for chunk_id in ranked:
            act, start, end = meta[chunk_id]
            parent = self.parent_of(act, start, end)
            if parent is None or parent in seen:
                continue
            seen.add(parent)
            out.append(parent)
            if len(out) == top:
                break
        return out
