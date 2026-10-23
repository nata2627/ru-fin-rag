"""Аналитика над журналом запросов: один набор запросов, два режима хранения.

На стройке события лежат в ClickHouse сервером (профиль `logs` в
`docker-compose.yml`), в бесплатном режиме сервера нет и не будет: события это
файлы JSONL в репозитории (DECISIONS, решение 12), и считает по ним `chdb`,
тот же движок ClickHouse внутри процесса. Чтобы два режима не разъехались,
текст запросов витрины здесь один (`MART`), а различается только то, откуда
берется отношение `events`: подзапрос `file(...)` или таблица на сервере.
Совпадение проверяется тестом `tests/test_warehouse.py`, а не глазами.

Эти же запросы станут моделями dbt в задаче 4.4, поэтому они написаны как
представления над `events`, без ссылок на имя файла или базы внутри текста.

`quantileExact`, а не `quantile`: приближенный квантиль ClickHouse считает по
выборке из потока, и два режима дали бы разные числа на одних данных по
причине, которая не имеет отношения к хранению.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Колонки события в том виде, в каком их читает аналитика. Одно описание на два
# режима: по нему строится и структура для `file()`, и таблица на сервере,
# иначе типы разойдутся и агрегаты начнут печататься по-разному.
EVENT_COLUMNS = (
    ("kind", "String"),
    ("request_id", "String"),
    ("received_at", "String"),
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

TABLE = "rufinrag.query_events"

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

# Трафик по суткам. Сутки берутся из начала `received_at` строкой: файл журнала
# уже нарезан по суткам UTC (events.py), и разбор в дату с часовым поясом
# добавил бы ошибку там, где ее нет.
TRAFFIC_BY_DAY = """
SELECT
    substring(received_at, 1, 10) AS day,
    count() AS queries,
    countIf(outcome = 'ok') AS answered,
    countIf(outcome = 'rejected') AS rejected,
    countIf(outcome = 'unavailable') AS unavailable
FROM events
WHERE kind = 'query'
GROUP BY day
ORDER BY day
"""

# Задержка по ступеням. Ступени приходят картой, а не колонками: их состав
# зависит от связки поиска (у bm25 нет плотной ступени, у hybrid-rerank есть
# переранжирование), и колонка под каждую ступень половину времени стояла бы
# пустой.
STAGE_LATENCY = """
SELECT
    stage,
    count() AS runs,
    round(quantileExact(0.5)(ms), 3) AS p50_ms,
    round(quantileExact(0.95)(ms), 3) AS p95_ms,
    round(max(ms), 3) AS max_ms
FROM (
    SELECT
        arrayJoin(timings_ms) AS pair,
        pair.1 AS stage,
        pair.2 AS ms
    FROM events
    WHERE kind = 'query' AND outcome = 'ok'
)
GROUP BY stage
ORDER BY stage
"""

# Доля запросов с оценкой по суткам. Числитель и знаменатель стоят рядом
# числами (AGENTS.md, раздел 2), долю считает тот, кто печатает текст.
RATED_SHARE = """
SELECT
    day,
    count() AS answered,
    countIf(ratings > 0) AS rated
FROM (
    SELECT substring(received_at, 1, 10) AS day, request_id
    FROM events
    WHERE kind = 'query' AND outcome = 'ok'
) AS q
LEFT JOIN (
    SELECT request_id, count() AS ratings
    FROM events
    WHERE kind = 'feedback'
    GROUP BY request_id
) AS r USING (request_id)
GROUP BY day
ORDER BY day
"""

# Соединение оценок с запросами по `request_id`. Оценка без своего запроса это
# не мусор, который можно выбросить молча: процесс сервиса не проверяет
# существование идентификатора (api.py), и такая строка приходит и от опечатки
# клиента, и от оценки запроса, чьи сутки уже уехали в другой файл. Поэтому
# сирота считается числом рядом с общим числом оценок.
FEEDBACK_MATCH = """
SELECT
    count() AS feedback_events,
    countIf(known = 1) AS matched,
    countIf(known = 0) AS orphans,
    countIf(rating = 'up') AS up,
    countIf(rating = 'down') AS down
FROM (
    SELECT request_id, rating
    FROM events
    WHERE kind = 'feedback'
) AS f
LEFT JOIN (
    SELECT DISTINCT request_id, 1 AS known
    FROM events
    WHERE kind = 'query'
) AS q USING (request_id)
"""

MART = {
    "traffic_by_day": TRAFFIC_BY_DAY,
    "stage_latency": STAGE_LATENCY,
    "rated_share": RATED_SHARE,
    "feedback_match": FEEDBACK_MATCH,
}

OUTPUT_FORMAT = "TabSeparatedWithNames"


def structure() -> str:
    return ", ".join(f"{name} {kind}" for name, kind in EVENT_COLUMNS)


def columns() -> str:
    return ",\n    ".join(f"{name} {kind}" for name, kind in EVENT_COLUMNS)


def compose(name: str, source: str) -> str:
    """Запрос витрины поверх своего источника событий."""
    return f"WITH events AS (\n{source.strip()}\n){MART[name].strip()}\nSETTINGS {SETTINGS}"


def parse_table(text: str) -> tuple[list[str], list[list[str]]]:
    """Разбор `TabSeparatedWithNames` в заголовок и строки.

    Сравнение режимов идет по тексту, а разбор нужен тому, кто печатает отчет.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return [], []
    return lines[0].split("\t"), [line.split("\t") for line in lines[1:]]


class WarehouseError(Exception):
    pass


@dataclass
class FilesWarehouse:
    """Бесплатный режим: события читаются из файлов JSONL, сервера нет.

    Движок тот же ClickHouse, он работает внутри процесса python, поэтому
    запрос витрины на бесплатном тарифе и на стройке пишется один раз.
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
        self.execute("CREATE DATABASE IF NOT EXISTS rufinrag")
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


def mart(warehouse: FilesWarehouse | ServerWarehouse) -> dict[str, str]:
    return {name: warehouse.run(name) for name in MART}


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
