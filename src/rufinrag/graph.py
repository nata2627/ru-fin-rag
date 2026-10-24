"""Граф ссылок между актами корпуса и расширение выдачи по цитированию.

Акты ЦБ ссылаются друг на друга номером, и это уже измерено с другой стороны:
в части А из 676 позиций первой десятки BM25, взятых не из названного вопросом
акта, 231 содержала его номер в тексте (EXPERIMENTS, S8). Там перекрестная
ссылка была помехой, которую снял фильтр по номеру. Здесь та же связь берется
нарочно: от найденных фрагментов идем по ссылкам и добавляем в выдачу фрагменты
соседних актов.

Дуга строится по номеру с буквенным суффиксом (`references.act_numbers`).
Номер самого акта в дугу не превращается: шапка акта стоит в тексте каждого его
фрагмента, поэтому свой номер встречается в корпусе 62 606 раз и ссылкой не
является.

Номер не всегда указывает на один акт: методические рекомендации нумеруются
заново каждый год, и «10-МР» носят четыре разных документа. Такой номер дает
дуги во все одноименные акты, а число таких упоминаний печатается отдельно:
расширение выдачи работает на полноту, а выбор года по тексту ссылки это
отдельная задача, которая здесь не решается.

Граф собирается из корпуса в памяти и на диске не хранится: сборка это один
проход регулярным выражением по текстам, и держать рядом с корпусом еще один
файл, который может от него отстать, дороже, чем пересобрать.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .corpus import Corpus
from .references import count_act_numbers

DIRECTIONS = ("forward", "backward", "both")

# Буквенные суффиксы, которыми кончаются номера настоящих актов: федеральные
# законы и виды актов ЦБ. Нужны одной оценке: выражение части А берет любое
# «цифры-буквы», поэтому среди ссылок за пределы корпуса есть и не акты («1-я
# группа» дает «1-Я», латинские буквы приходят из формул). Список заведомо
# неполон: «4015-I» это закон об организации страхового дела, и в него не
# попадает. Поэтому число по списку это оценка снизу, и так оно и называется.
ACT_SUFFIXES = ("ФЗ", "ФКЗ", "У", "П", "И", "Т", "ОД", "МР", "ОР", "ИН")


class GraphError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class GraphStats:
    """Числа графа. Публикуются как есть, вместе с оговоркой про выражение.

    `outside_mentions` считает все упоминания номеров, которых в корпусе нет, и
    в них попадают не только акты: выражение части А берет любое «цифры-буквы»,
    то есть «1-я группа» тоже. Поэтому рядом печатается `top_outside`, и по нему
    видно, что это за номера.
    """

    acts: int
    chunks: int
    mentions: int
    references: int
    references_inside: int
    references_outside: int
    edges: int
    acts_with_references: int
    acts_cited: int
    ambiguous_references: int
    outside_numbers: int
    mentions_inside: int
    mentions_outside: int
    mentions_outside_act_like: int
    build_seconds: float
    top_cited: list[tuple[str, int]] = field(default_factory=list)
    top_outside: list[tuple[str, int]] = field(default_factory=list)
    outside_by_suffix: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class ActGraph:
    forward: dict[str, tuple[str, ...]]
    backward: dict[str, tuple[str, ...]]
    act_chunks: dict[str, np.ndarray]
    stats: GraphStats

    def neighbors(self, act_ids: Sequence[str], direction: str) -> list[str]:
        """Соседние акты в порядке первого появления: порядок семян важен."""
        if direction not in DIRECTIONS:
            raise GraphError(f"направление {direction!r}, известны: {', '.join(DIRECTIONS)}")
        seeds = set(act_ids)
        out: list[str] = []
        for act_id in act_ids:
            found: tuple[str, ...] = ()
            if direction in ("forward", "both"):
                found += self.forward.get(act_id, ())
            if direction in ("backward", "both"):
                found += self.backward.get(act_id, ())
            for other in found:
                # Акт семени уже представлен в базовой выдаче, добавлять его
                # фрагменты значит усиливать то, что и так найдено.
                if other not in seeds and other not in out:
                    out.append(other)
        return out


def build_graph(corpus: Corpus, top: int = 20) -> ActGraph:
    t0 = time.monotonic()
    number_to_acts: dict[str, list[str]] = defaultdict(list)
    act_number: dict[str, str] = {}
    act_rows: dict[str, list[int]] = defaultdict(list)
    for row, chunk in enumerate(corpus.chunks):
        act_id = chunk.act_id
        act_rows[act_id].append(row)
        if act_id not in act_number:
            act_number[act_id] = chunk.act.number
            number_to_acts[chunk.act.number].append(act_id)

    mentions: dict[tuple[str, str], int] = defaultdict(int)
    for chunk, text in zip(corpus.chunks, corpus.iter_texts(), strict=True):
        act_id = chunk.act_id
        own = act_number[act_id]
        for number, count in count_act_numbers(text).items():
            if number != own:
                mentions[(act_id, number)] += count

    forward: dict[str, list[str]] = defaultdict(list)
    backward: dict[str, list[str]] = defaultdict(list)
    cited: dict[str, int] = defaultdict(int)
    outside: dict[str, int] = defaultdict(int)
    inside_references = ambiguous = 0
    mentions_inside = mentions_outside = 0
    for (act_id, number), count in mentions.items():
        targets = number_to_acts.get(number)
        if not targets:
            outside[number] += count
            mentions_outside += count
            continue
        inside_references += 1
        mentions_inside += count
        if len(targets) > 1:
            ambiguous += 1
        for target in targets:
            if target == act_id:
                continue
            if target not in forward[act_id]:
                forward[act_id].append(target)
                backward[target].append(act_id)
                cited[target] += 1

    stats = GraphStats(
        acts=len(act_number),
        chunks=len(corpus.chunks),
        mentions=mentions_inside + mentions_outside,
        references=len(mentions),
        references_inside=inside_references,
        references_outside=len(mentions) - inside_references,
        edges=sum(len(v) for v in forward.values()),
        acts_with_references=sum(1 for v in forward.values() if v),
        acts_cited=len(cited),
        ambiguous_references=ambiguous,
        outside_numbers=len(outside),
        mentions_inside=mentions_inside,
        mentions_outside=mentions_outside,
        mentions_outside_act_like=sum(
            count for number, count in outside.items() if _suffix(number) in ACT_SUFFIXES
        ),
        build_seconds=time.monotonic() - t0,
        top_cited=_top(cited, top),
        top_outside=_top(outside, top),
        outside_by_suffix=_top(_by_suffix(outside), top),
    )
    return ActGraph(
        forward={k: tuple(v) for k, v in forward.items() if v},
        backward={k: tuple(v) for k, v in backward.items() if v},
        act_chunks={k: np.asarray(v, dtype=np.int64) for k, v in act_rows.items()},
        stats=stats,
    )


def _top(counts: dict[str, int], top: int) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda row: (-row[1], row[0]))[:top]


def _suffix(number: str) -> str:
    return number.rsplit("-", 1)[-1] if "-" in number else number


def _by_suffix(counts: dict[str, int]) -> dict[str, int]:
    """Упоминания за пределами корпуса по буквенному суффиксу номера.

    Печатается рядом с числом таких упоминаний, потому что выражение части А
    берет любое «цифры-буквы», и в этот счет попадают не только акты: «1-я
    группа» дает номер «1-Я». По разбивке видно, сколько в нем законов (ФЗ),
    сколько отмененных актов ЦБ (П, У, И) и сколько мусора.
    """
    out: dict[str, int] = {}
    for number, count in counts.items():
        out[_suffix(number)] = out.get(_suffix(number), 0) + count
    return out


@dataclass(frozen=True, slots=True)
class Expansion:
    """Что расширение добавило. Нужно замеру: без этого прибавку нечем объяснить."""

    neighbor_acts: int
    added: int
    seed_acts: tuple[str, ...]


def cited_candidates(
    query: str,
    base: Sequence[tuple[str, float]],
    corpus: Corpus,
    graph: ActGraph,
    bm25,
    direction: str = "forward",
    seeds: int = 5,
    depth: int = 10,
) -> tuple[list[tuple[str, float]], Expansion]:
    """Второй список кандидатов: лучшие по BM25 фрагменты соседних по графу актов.

    Соседи берутся от актов первых `seeds` фрагментов базовой выдачи. Внутри
    соседних актов порядок решает тот же BM25, что и в первой ступени: иначе
    в выдачу поехал бы первый фрагмент акта, то есть его титульный лист.
    """
    seed_acts: list[str] = []
    for chunk_id, _ in base[:seeds]:
        act_id = corpus.get(chunk_id).act_id
        if act_id not in seed_acts:
            seed_acts.append(act_id)
    neighbors = graph.neighbors(seed_acts, direction)
    if not neighbors or depth <= 0:
        return [], Expansion(len(neighbors), 0, tuple(seed_acts))

    rows = np.concatenate([graph.act_chunks[act_id] for act_id in neighbors])
    scores = bm25.scores(query)[rows]
    # Ничьи решаются по номеру фрагмента в корпусе, как в `retrieval.ranking`:
    # порядок выдачи не должен зависеть от сборки numpy.
    order = np.lexsort((rows, -scores))[:depth]
    picked = [
        (corpus.chunks[int(rows[i])].chunk_id, float(scores[i])) for i in order if scores[i] > 0
    ]
    return picked, Expansion(len(neighbors), len(picked), tuple(seed_acts))


def reserve_merge(
    base: Sequence[tuple[str, float]],
    extra: Sequence[tuple[str, float]],
    reserve: int,
    top: int = 10,
) -> list[tuple[str, float]]:
    """Отдать расширению последние `reserve` мест выдачи, остальное оставить базе.

    Второй режим расширения, и он появился из замера первого: слияние обратными
    рангами переставляет всю первую десятку даже с маленьким весом, потому что
    оценки RRF на глубине 50 различаются в 1,8 раза (EXPERIMENTS, S20). Здесь
    базовый порядок не трогается вовсе, а цена расширения видна глазами: это
    ровно `reserve` мест из `top`.
    """
    if reserve < 0:
        raise GraphError(f"резерв {reserve}, нужно не меньше нуля")
    keep = max(top - reserve, 0)
    head = [row for row in base[:keep]]
    taken = {chunk_id for chunk_id, _ in head}
    tail = [row for row in extra if row[0] not in taken][:reserve]
    taken.update(chunk_id for chunk_id, _ in tail)
    rest = [row for row in base[keep:] if row[0] not in taken]
    return (head + tail + rest)[:top]
