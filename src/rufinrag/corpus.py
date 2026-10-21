"""Корпус фрагментов: чтение, метаданные, ссылка на источник.

Фрагменты лежат в репозитории сервиса сжатыми (DECISIONS, решение 7), потому
что публичного источника корпуса нет: датасет части А не опубликован, а сбор
с cbr.ru при каждом развертывании это часы прогона и нагрузка на госсайт.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import data_dir

CORPUS_FILE = "corpus/chunks.jsonl.gz"

# Поля, которые сервис отдает наружу и хранит. Остальные поля части А
# (`units`, `issue`) в выдаче не нужны.
FIELDS = (
    "chunk_id",
    "act_id",
    "number",
    "date",
    "type",
    "title",
    "url",
    "section",
    "position",
    "char_start",
    "char_end",
    "text",
)


class CorpusError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    act_id: str
    number: str
    date: str
    type: str
    title: str
    url: str
    section: str
    position: int
    char_start: int
    char_end: int
    text: str

    @property
    def act_label(self) -> str:
        """Человеческое имя акта: «Указание Банка России № 3054-У от 04.09.2013»."""
        kinds = {
            "У": "Указание",
            "П": "Положение",
            "И": "Инструкция",
            "ОД": "Приказ",
        }
        kind = kinds.get(self.type, "Акт")
        return f"{kind} Банка России № {self.number} от {self.date}"


@dataclass
class Corpus:
    chunks: list[Chunk]
    source: Path

    def __post_init__(self) -> None:
        self._by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    def get(self, chunk_id: str) -> Chunk:
        try:
            return self.chunks[self._by_id[chunk_id]]
        except KeyError as exc:
            raise CorpusError(f"нет фрагмента {chunk_id!r} в корпусе {self.source}") from exc

    @property
    def version(self) -> str:
        """Версия корпуса для ответа API и для строки лога.

        Число фрагментов и число актов: этого достаточно, чтобы заметить, что
        выдача получена на другом корпусе, а считается это без чтения файла
        заново.
        """
        acts = {c.act_id for c in self.chunks}
        return f"chunks{len(self.chunks)}-acts{len(acts)}"


def corpus_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / CORPUS_FILE


def read_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_corpus(path: Path | None = None) -> Corpus:
    path = path or corpus_path()
    if not path.exists():
        raise CorpusError(
            f"нет корпуса {path}. Собрать из части А: python scripts/import_corpus.py"
        )
    chunks = [
        Chunk(
            chunk_id=row["chunk_id"],
            act_id=row["act_id"],
            number=row.get("number", ""),
            date=row.get("date", ""),
            type=row.get("type", ""),
            title=row.get("title", ""),
            url=row.get("url", ""),
            section=row.get("section", ""),
            position=int(row.get("position", 0)),
            char_start=int(row.get("char_start", 0)),
            char_end=int(row.get("char_end", 0)),
            text=row["text"],
        )
        for row in read_jsonl(path)
    ]
    if not chunks:
        raise CorpusError(f"корпус {path} пуст")
    return Corpus(chunks=chunks, source=path)
