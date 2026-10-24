"""Корпус фрагментов: чтение, метаданные, текст по требованию.

Фрагменты лежат в репозитории сервиса сжатыми (DECISIONS, решение 7), потому
что публичного источника корпуса нет: датасет части А не опубликован, а сбор с
cbr.ru при каждом развертывании это часы прогона и нагрузка на госсайт.

Тексты держатся в памяти сжатыми блоками, а не строками. Причина в бесплатном
хостинге: у Render на бесплатном тарифе 512 МБ, а сервис с текстами как строки
занимал 527 МБ. Замер 21.10.2026 на 62 594 фрагментах: строками 219 МБ, одним
буфером в utf-8 200 МБ, блоками по 64 фрагмента со сжатием zlib 23 МБ при
доступе к одному фрагменту 0,196 мс. Выдача из десяти фрагментов стоит около
двух миллисекунд, и это дешевле, чем не влезть в тариф.
"""

from __future__ import annotations

import gzip
import json
import zlib
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import data_dir

CORPUS_FILE = "corpus/chunks.jsonl.gz"
BLOCK_SIZE = 64
COMPRESS_LEVEL = 6
CACHE_BLOCKS = 16
SEPARATOR = "\x00"

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

# Вид акта по буквенному суффиксу номера. Виды взяты из корпуса, а не из
# общего знания: в нем встречаются У (29 139 фрагментов), П (27 102),
# И (4075), МР (2198) и ОР (92). Без МР и ОР их 2290 фрагментов назывались в
# выдаче «Акт Банка России», хотя в тексте фрагмента напечатан вид.
ACT_KINDS = {
    "У": "Указание",
    "П": "Положение",
    "И": "Инструкция",
    "ОД": "Приказ",
    "МР": "Методические рекомендации",
    "ОР": "Официальное разъяснение",
}


class CorpusError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Act:
    act_id: str
    number: str
    date: str
    type: str
    title: str
    url: str

    @property
    def label(self) -> str:
        """Человеческое имя акта: «Указание Банка России № 3054-У от 04.09.2013»."""
        return f"{ACT_KINDS.get(self.type, 'Акт')} Банка России № {self.number} от {self.date}"


@dataclass(frozen=True, slots=True)
class Chunk:
    """Фрагмент без текста: поля акта лежат в `Act`, текст в `TextStore`.

    Так сделано ради памяти: полей акта 1554 набора на 62 594 фрагмента, и
    хранить их в каждом фрагменте значит держать те же строки сорок раз.
    """

    chunk_id: str
    act: Act
    section: str
    position: int
    char_start: int
    char_end: int

    @property
    def act_id(self) -> str:
        return self.act.act_id


class TextStore:
    """Тексты фрагментов в сжатых блоках. Распакованный блок кэшируется.

    Блоки собираются по ходу чтения файла, а не из готового списка строк: иначе
    пик памяти на старте равен размеру всех текстов, то есть тому, от чего
    блоки и спасают.
    """

    def __init__(self, blocks: list[bytes], count: int, block_size: int = BLOCK_SIZE):
        self.blocks = blocks
        self.count = count
        self.block_size = block_size
        self._cache: OrderedDict[int, list[str]] = OrderedDict()

    @property
    def size_bytes(self) -> int:
        return sum(len(b) for b in self.blocks)

    def save(self, path: Path) -> None:
        """Блоки на диск, чтобы холодный старт не тратил секунды на сжатие."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"RFTS")
            f.write(len(self.blocks).to_bytes(4, "little"))
            f.write(self.count.to_bytes(4, "little"))
            f.write(self.block_size.to_bytes(4, "little"))
            for block in self.blocks:
                f.write(len(block).to_bytes(4, "little"))
            for block in self.blocks:
                f.write(block)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as f:
            if f.read(4) != b"RFTS":
                raise CorpusError(f"{path}: не файл блоков текста")
            count_blocks = int.from_bytes(f.read(4), "little")
            count = int.from_bytes(f.read(4), "little")
            block_size = int.from_bytes(f.read(4), "little")
            sizes = [int.from_bytes(f.read(4), "little") for _ in range(count_blocks)]
            blocks = [f.read(size) for size in sizes]
        return cls(blocks, count=count, block_size=block_size)

    @property
    def size_mb(self) -> float:
        return sum(len(b) for b in self.blocks) / 1048576

    def get(self, index: int) -> str:
        if not 0 <= index < self.count:
            raise CorpusError(f"нет фрагмента с номером {index}, всего {self.count}")
        block_index, offset = divmod(index, self.block_size)
        block = self._cache.get(block_index)
        if block is None:
            block = zlib.decompress(self.blocks[block_index]).decode().split(SEPARATOR)
            self._cache[block_index] = block
            if len(self._cache) > CACHE_BLOCKS:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(block_index)
        return block[offset]


class TextBuilder:
    """Накопитель блоков: принимает тексты по одному, держит только текущий блок."""

    def __init__(self, block_size: int = BLOCK_SIZE):
        self.block_size = block_size
        self.blocks: list[bytes] = []
        self._current: list[str] = []
        self.count = 0

    def add(self, text: str) -> None:
        self._current.append(text)
        self.count += 1
        if len(self._current) == self.block_size:
            self._flush()

    def _flush(self) -> None:
        if self._current:
            self.blocks.append(
                zlib.compress(SEPARATOR.join(self._current).encode(), COMPRESS_LEVEL)
            )
            self._current = []

    def build(self) -> TextStore:
        self._flush()
        return TextStore(self.blocks, count=self.count, block_size=self.block_size)


@dataclass
class Corpus:
    chunks: list[Chunk]
    texts: TextStore
    source: Path

    def __post_init__(self) -> None:
        self._by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    def index_of(self, chunk_id: str) -> int:
        try:
            return self._by_id[chunk_id]
        except KeyError as exc:
            raise CorpusError(f"нет фрагмента {chunk_id!r} в корпусе {self.source}") from exc

    def get(self, chunk_id: str) -> Chunk:
        return self.chunks[self.index_of(chunk_id)]

    def text(self, chunk_id: str) -> str:
        return self.texts.get(self.index_of(chunk_id))

    def iter_texts(self) -> Iterator[str]:
        """Тексты по порядку: нужны сборке индекса, которая читает их один раз."""
        for i in range(len(self.chunks)):
            yield self.texts.get(i)

    @property
    def version(self) -> str:
        """Версия корпуса для ответа API и для строки лога.

        Число фрагментов и число актов: этого достаточно, чтобы заметить, что
        выдача получена на другом корпусе.
        """
        acts = {c.act.act_id for c in self.chunks}
        return f"chunks{len(self.chunks)}-acts{len(acts)}"


def corpus_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / CORPUS_FILE


def read_jsonl(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_corpus(path: Path | None = None, texts_path: Path | None = None) -> Corpus:
    """Прочитать корпус. Если блоки текста уже собраны, взять их с диска.

    На холодном старте это разница между секундами сжатия и чтением готового
    файла, а бесплатный хостинг просыпается на каждом запросе после простоя.
    """
    path = path or corpus_path()
    if not path.exists():
        raise CorpusError(
            f"нет корпуса {path}. Собрать из части А: python scripts/import_corpus.py"
        )
    ready = TextStore.load(texts_path) if texts_path and texts_path.exists() else None
    acts: dict[str, Act] = {}
    chunks: list[Chunk] = []
    builder = TextBuilder()
    for row in read_jsonl(path):
        act_id = row["act_id"]
        act = acts.get(act_id)
        if act is None:
            act = acts[act_id] = Act(
                act_id=act_id,
                number=row.get("number", ""),
                date=row.get("date", ""),
                type=row.get("type", ""),
                title=row.get("title", ""),
                url=row.get("url", ""),
            )
        chunks.append(
            Chunk(
                chunk_id=row["chunk_id"],
                act=act,
                section=row.get("section", ""),
                position=int(row.get("position", 0)),
                char_start=int(row.get("char_start", 0)),
                char_end=int(row.get("char_end", 0)),
            )
        )
        if ready is None:
            builder.add(row["text"])
    if not chunks:
        raise CorpusError(f"корпус {path} пуст")
    texts = ready or builder.build()
    if texts.count != len(chunks):
        raise CorpusError(
            f"{texts_path}: текстов {texts.count}, фрагментов в корпусе {len(chunks)}. "
            "Пересобрать: python scripts/build_index.py"
        )
    return Corpus(chunks=chunks, texts=texts, source=path)
