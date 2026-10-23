"""Аналитика над журналом запросов: одни модели, два режима хранения.

Модели витрины лежат в проекте dbt (`dbt/models`), и других копий у них нет.
На стройке и в прогоне CI их собирает dbt в ClickHouse сервером (профиль `logs`
в `docker-compose.yml`): слой `staging` чистит события, слой `marts` отвечает на
вопросы, схема и тесты лежат рядом с моделями. В бесплатном режиме сервера нет
и не будет: события это файлы JSONL в репозитории (DECISIONS, решение 12), dbt
по ним работать не может, и тот же текст модели исполняет `chdb`, движок
ClickHouse внутри процесса.

Поэтому здесь нет ни одного запроса витрины, здесь исполнитель моделей dbt без
dbt: `ref` разворачивается в общее табличное выражение, `source` в тот источник
событий, который есть у режима (подзапрос `file(...)` или таблица на сервере), и
модель становится одним запросом. Что dbt и `chdb` считают по моделям одно и то
же, проверяет `scripts/build_mart.py`, а не глаза.

`quantileExact`, а не `quantile`: приближенный квантиль ClickHouse считает по
выборке из потока, и два режима дали бы разные числа на одних данных по
причине, которая не имеет отношения к хранению.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

# Колонки события в том виде, в каком их читает аналитика. Одно описание на два
# режима: по нему строится и структура для `file()`, и таблица на сервере,
# иначе типы разойдутся и агрегаты начнут печататься по-разному.
EVENT_COLUMNS = (
    ("kind", "String"),
    ("request_id", "String"),
    ("received_at", "String"),
    ("query_normalized", "String"),
    ("query_chars", "UInt32"),
    ("outcome", "String"),
    ("status", "UInt16"),
    ("config", "String"),
    ("config_version", "String"),
    ("corpus_version", "String"),
    ("timings_ms", "Map(String, Float64)"),
    ("hits", "UInt32"),
    ("rating", "String"),
)

SCHEMA = "rufinrag"
TABLE = f"{SCHEMA}.query_events"

DBT_PROJECT = Path(__file__).resolve().parents[2] / "dbt"
MARTS = "marts"
STAGING = "staging"

# Настройки стоят в тексте запроса, а не в клиенте: иначе один режим читал бы
# события с настройками, а другой без них.
# `skip_unknown_fields`: в строке события есть поля, которых витрине не надо
# (`results`, `reason`), и схема события растет быстрее витрины.
# `defaults_for_omitted_fields`: у оценки нет полей запроса, у отвергнутого
# запроса нет задержек, и это нормальная строка, а не битая.
SETTINGS = (
    "input_format_skip_unknown_fields = 1, "
    "input_format_defaults_for_omitted_fields = 1, "
    "join_use_nulls = 0"
)

OUTPUT_FORMAT = "TabSeparatedWithNames"

# Место источника событий в тексте модели. Строка, а не разметка jinja: дальше
# текст идет в ClickHouse, и остаток jinja в нем должен быть виден как ошибка.
SOURCE_PLACEHOLDER = "__journal_query_events__"

_REF = re.compile(r"\{\{\s*ref\(\s*'([a-z0-9_]+)'\s*\)\s*\}\}")
_SOURCE = re.compile(r"\{\{\s*source\(\s*'journal'\s*,\s*'query_events'\s*\)\s*\}\}")


class WarehouseError(Exception):
    pass


@dataclass(frozen=True)
class Model:
    """Модель dbt без разметки: текст, слой и имена моделей, от которых зависит."""

    name: str
    layer: str
    body: str
    refs: tuple[str, ...]


def load_models(project: Path | None = None) -> dict[str, Model]:
    directory = (project or DBT_PROJECT) / "models"
    if not directory.is_dir():
        raise WarehouseError(f"нет моделей dbt в {directory}")
    models: dict[str, Model] = {}
    for path in sorted(directory.glob("*/*.sql")):
        text = path.read_text(encoding="utf-8")
        refs = tuple(dict.fromkeys(_REF.findall(text)))
        body = _SOURCE.sub(SOURCE_PLACEHOLDER, _REF.sub(r"\1", text)).strip()
        if "{{" in body:
            raise WarehouseError(
                f"в модели {path.name} осталась разметка jinja, бесплатный режим ее не исполняет"
            )
        models[path.stem] = Model(path.stem, path.parent.name, body, refs)
    if not models:
        raise WarehouseError(f"в {directory} нет ни одной модели")
    return models


def upstream(models: dict[str, Model], name: str) -> list[str]:
    """Модели, от которых зависит `name`, в порядке, годном для цепочки CTE."""
    order: list[str] = []
    done: set[str] = set()
    path: list[str] = []

    def walk(current: str) -> None:
        if current in path:
            raise WarehouseError("модели ссылаются по кругу: " + " -> ".join([*path, current]))
        if current in done:
            return
        path.append(current)
        for ref in models[current].refs:
            if ref not in models:
                raise WarehouseError(f"{current} ссылается на модель {ref}, которой нет")
            walk(ref)
        path.pop()
        done.add(current)
        if current != name:
            order.append(current)

    walk(name)
    return order


class _Marts(Mapping[str, str]):
    """Модели слоя `marts`: имя и текст.

    Читается при первом обращении, а не при импорте модуля: в образе сервиса
    проекта dbt нет (туда едут пакет, конфигурация и индекс), и модуль обязан
    импортироваться там без него.
    """

    def __init__(self) -> None:
        self._bodies: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._bodies is None:
            models = load_models()
            self._bodies = {
                name: model.body for name, model in models.items() if model.layer == MARTS
            }
        return self._bodies

    def __getitem__(self, name: str) -> str:
        return self._load()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())


MART = _Marts()


def structure() -> str:
    return ", ".join(f"{name} {kind}" for name, kind in EVENT_COLUMNS)


def columns() -> str:
    return ",\n    ".join(f"{name} {kind}" for name, kind in EVENT_COLUMNS)


def compose(name: str, source: str) -> str:
    """Модель dbt как один запрос: зависимости становятся CTE, `source` подзапросом."""
    models = load_models()
    if name not in models:
        raise WarehouseError(f"нет модели {name}, есть {', '.join(sorted(models))}")
    chain = [
        f"{dep} AS (\n{_resolve(models[dep].body, source)}\n)" for dep in upstream(models, name)
    ]
    body = _resolve(models[name].body, source)
    if not chain:
        return f"{body}\nSETTINGS {SETTINGS}"
    # Своя секция `with` у модели сливается с цепочкой зависимостей: секция в
    # запросе бывает одна, а своя у моделей есть.
    own, body = _merge_with(body)
    return "WITH " + ",\n".join(chain) + (",\n" if own else "\n") + body + f"\nSETTINGS {SETTINGS}"


def _resolve(body: str, source: str) -> str:
    return body.replace(SOURCE_PLACEHOLDER, f"(\n{source.strip()}\n)")


def _merge_with(body: str) -> tuple[bool, str]:
    """Снять с модели ключевое слово `with`, если оно у нее есть."""
    lines = body.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        if not re.match(r"with\s", stripped, flags=re.IGNORECASE):
            return False, body
        lines[index] = re.sub(r"^(\s*)with\s+", r"\1", line, flags=re.IGNORECASE)
        return True, "\n".join(lines)
    return False, body


def parse_table(text: str) -> tuple[list[str], list[list[str]]]:
    """Разбор `TabSeparatedWithNames` в заголовок и строки.

    Сравнение режимов идет по тексту, а разбор нужен тому, кто печатает отчет.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return [], []
    return lines[0].split("\t"), [line.split("\t") for line in lines[1:]]


@dataclass
class FilesWarehouse:
    """Бесплатный режим: события читаются из файлов JSONL, сервера нет.

    Движок тот же ClickHouse, он работает внутри процесса python, поэтому
    модель витрины на бесплатном тарифе и на стройке пишется один раз.
    """

    directory: Path
    pattern: str = "queries-*.jsonl"

    def source(self) -> str:
        # Путь абсолютный: `file()` разбирает его относительно рабочей папки
        # процесса, а прогон CI и прогон с мака запускаются из разных мест.
        glob = str(Path(self.directory).resolve() / self.pattern)
        return f"SELECT * FROM file('{glob}', JSONEachRow, '{structure()}')"

    def run(self, name: str) -> str:
        import chdb

        result = chdb.query(compose(name, self.source()), OUTPUT_FORMAT)
        return str(result)


@dataclass
class ServerWarehouse:
    """Режим стройки: ClickHouse сервером из compose, профиль `logs`.

    Клиент это HTTP-интерфейс ClickHouse на urllib, а не драйвер: драйвер стал
    бы зависимостью ради трех запросов, а интерфейс здесь это POST с текстом
    запроса в теле.
    """

    url: str = "http://127.0.0.1:8123"
    timeout: int = 60

    def execute(
        self, sql: str, body: bytes | None = None, fmt: str = OUTPUT_FORMAT, **settings: str
    ) -> str:
        params = {"default_format": fmt, **settings}
        if body is not None:
            params["query"] = sql
            data = body
        else:
            data = sql.encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/?{urllib.parse.urlencode(params)}", data=data, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8")

    def scalar(self, sql: str) -> str:
        return self.execute(sql, fmt="TabSeparated").strip()

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/ping", timeout=5) as response:
                return response.read().decode("utf-8").strip() == "Ok."
        except OSError:
            return False

    def load(self, directory: Path, pattern: str = "queries-*.jsonl") -> int:
        """Залить файлы журнала в таблицу, предварительно ее пересоздав.

        Таблица пересоздается, а не дополняется: сверка режимов обязана читать
        ровно те же события, что лежат в файлах, а дополнение оставило бы в ней
        прошлый прогон и расхождение объяснялось бы не кодом, а состоянием.
        """
        files = sorted(Path(directory).resolve().glob(pattern))
        if not files:
            raise WarehouseError(f"нет файлов {pattern} в {directory}")
        self.execute(f"CREATE DATABASE IF NOT EXISTS {SCHEMA}")
        self.execute(f"DROP TABLE IF EXISTS {TABLE}")
        self.execute(
            f"CREATE TABLE {TABLE} (\n    {columns()}\n) "
            "ENGINE = MergeTree ORDER BY (received_at, request_id)"
        )
        lines = 0
        for path in files:
            raw = path.read_bytes()
            self.execute(
                f"INSERT INTO {TABLE} FORMAT JSONEachRow",
                body=raw,
                input_format_skip_unknown_fields="1",
                input_format_defaults_for_omitted_fields="1",
            )
            lines += sum(1 for line in raw.splitlines() if line.strip())
        return lines

    def source(self) -> str:
        return f"SELECT * FROM {TABLE}"

    def run(self, name: str) -> str:
        return self.execute(compose(name, self.source()))

    def read_model(self, name: str) -> str:
        """Вывод модели, которую собрал dbt, прямо из базы."""
        return self.execute(f"SELECT * FROM {SCHEMA}.{name}")


def mart(warehouse: FilesWarehouse | ServerWarehouse) -> dict[str, str]:
    return {name: warehouse.run(name) for name in sorted(MART)}


def report(rows: dict[str, str]) -> dict:
    """Витрина в виде, пригодном для файла результата."""
    out: dict[str, dict] = {}
    for name, text in rows.items():
        header, body = parse_table(text)
        out[name] = {
            "columns": header,
            "rows": [dict(zip(header, row, strict=True)) for row in body],
        }
    return out


def dump(rows: dict[str, str], path: Path) -> None:
    path.write_text(json.dumps(report(rows), ensure_ascii=False, indent=2) + "\n", "utf-8")
