"""Эталонный набор: запросы, разметка, выдачи для сверки.

Набор приходит из части А и здесь заморожен: по нему идет регрессия качества
после каждого обновления корпуса, поэтому он версионируется вместе с кодом и
руками не размечается (SPEC, раздел 4).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import data_dir
from .corpus import read_jsonl

BENCHMARK_DIR = "benchmark"


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    text: str
    origin: str
    podvyborka: str


def benchmark_dir(base: Path | None = None) -> Path:
    return (base or data_dir()) / BENCHMARK_DIR


def load_queries(base: Path | None = None) -> dict[str, Query]:
    path = benchmark_dir(base) / "queries.jsonl"
    return {
        row["query_id"]: Query(
            query_id=row["query_id"],
            text=row["text"],
            origin=row["origin"],
            podvyborka=row["podvyborka"],
        )
        for row in read_jsonl(path)
    }


def load_qrels(base: Path | None = None) -> dict[str, dict[str, int]]:
    """Разметка: запрос, фрагмент, оценка. Нулевые оценки сохраняются.

    Нулевая оценка это проверенная нерелевантность, и она отличается от
    отсутствия строки, то есть от непроверенной пары.
    """
    path = benchmark_dir(base) / "qrels.tsv"
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 3:
                out[row[0]][row[1]] = int(row[2])
    return dict(out)


def load_run(name: str, base: Path | None = None) -> dict[str, list[str]]:
    """Выдача прогона: идентификатор запроса и упорядоченный список фрагментов."""
    path = benchmark_dir(base) / f"run-{name}.jsonl"
    return {row["query_id"]: list(row["ranked"]) for row in read_jsonl(path)}
