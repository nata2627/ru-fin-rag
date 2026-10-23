"""Журнал запросов: одна строка JSON на событие.

Поле содержит то, что написано на этикетке (AGENTS.md, раздел 2), поэтому в
событии запроса нет оценки пользователя: она приходит позже ответа, и поле в
строке запроса всегда стояло бы пустым. Оценка это отдельное событие с тем же
`request_id`, витрина соединяет их по нему. По той же причине поля, которых не
было в этом запросе, отсутствуют, а не равны `null`: у отвергнутого запроса
выдачи и задержек по ступеням не существует.

Записи живут в репозитории сервиса (DECISIONS, решение 12), то есть видны в
истории git. Поэтому в строке только то, что пользователь сам отправил в
открытый сервис, и технические поля: ни заголовков, ни адресов, ни куки.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .config import data_dir, env
from .retrieval.text import tokenize
from .search import SearchResult

SCHEMA_VERSION = 1

# Поля, которые есть в каждом событии запроса, чем бы он ни кончился.
QUERY_FIELDS = (
    "schema",
    "kind",
    "request_id",
    "received_at",
    "query",
    "query_chars",
    "top",
    "service_version",
    "outcome",
    "status",
)
# Поля, которые добавляет дошедший до поиска запрос.
ANSWERED_FIELDS = (
    "query_normalized",
    "config",
    "config_version",
    "corpus_version",
    "timings_ms",
    "hits",
    "results",
    "degraded",
)
FEEDBACK_FIELDS = ("schema", "kind", "request_id", "received_at", "rating")

RATINGS = ("up", "down")

# Предел длины текста в строке лога. Отвергнутый запрос бывает на мегабайт
# (QUALITY, раздел 3), а строки лога уезжают в репозиторий.
MAX_LOGGED_CHARS = 4000

log = logging.getLogger(__name__)


def new_request_id() -> str:
    return uuid.uuid4().hex


def utc_now() -> str:
    """Момент события в UTC, с миллисекундами."""
    return datetime.now(tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _query_head(query: str) -> dict:
    head = {"query": query[:MAX_LOGGED_CHARS], "query_chars": len(query)}
    if len(query) > MAX_LOGGED_CHARS:
        head["query_truncated_to"] = MAX_LOGGED_CHARS
    return head


def answered(
    request_id: str, received_at: str, top: int, result: SearchResult, service_version: str
) -> dict:
    """Событие запроса, который прошел поиск."""
    return {
        "schema": SCHEMA_VERSION,
        "kind": "query",
        "request_id": request_id,
        "received_at": received_at,
        **_query_head(result.query),
        # То, что увидела лексическая ступень: токены ее же токенизатора, а не
        # отдельная «нормализация для лога», которая разъехалась бы с поиском.
        "query_normalized": " ".join(tokenize(result.query)),
        "top": top,
        "service_version": service_version,
        "outcome": "ok",
        "status": 200,
        "config": result.config,
        "config_version": result.config_version,
        "corpus_version": result.corpus_version,
        "timings_ms": result.timings_ms,
        "hits": len(result.hits),
        "results": [
            {
                "rank": hit.rank,
                "chunk_id": hit.chunk_id,
                "act_id": hit.act_id,
                "score": round(hit.score, 6),
            }
            for hit in result.hits
        ],
        "degraded": result.degraded,
    }


def rejected(
    request_id: str,
    received_at: str,
    query: str,
    top: int | None,
    status: int,
    reason: str,
    service_version: str,
) -> dict:
    """Событие запроса, который до поиска не дошел: грязный ввод или сервис не готов."""
    return {
        "schema": SCHEMA_VERSION,
        "kind": "query",
        "request_id": request_id,
        "received_at": received_at,
        **_query_head(query),
        "top": top,
        "service_version": service_version,
        "outcome": "unavailable" if status >= 500 else "rejected",
        "status": status,
        "reason": reason[:500],
    }


def feedback(request_id: str, rating: str, received_at: str | None = None) -> dict:
    if rating not in RATINGS:
        raise ValueError(f"оценка {rating!r} вне {RATINGS}")
    return {
        "schema": SCHEMA_VERSION,
        "kind": "feedback",
        "request_id": request_id,
        "received_at": received_at or utc_now(),
        "rating": rating,
    }


@dataclass
class EventLog:
    """Дописывание событий в файл дня.

    Файл на сутки по UTC: так ночной прогон забирает закрытые сутки целиком, а
    дневной файл не растет без предела. Один обработчик держится открытым, и
    каждая строка сбрасывается на диск: бесплатный хостинг усыпляет процесс без
    предупреждения, и буфер в памяти пропал бы вместе с ним.
    """

    directory: Path
    written: int = 0
    failed: int = 0
    last_error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _handle: object | None = field(default=None, repr=False)
    _day: str = field(default="", repr=False)

    @classmethod
    def from_env(cls) -> EventLog | None:
        """Журнал по переменным окружения. `RUFINRAG_EVENT_LOG=off` выключает запись."""
        if env("RUFINRAG_EVENT_LOG").lower() == "off":
            return None
        override = env("RUFINRAG_EVENT_DIR")
        return cls(Path(override).expanduser() if override else data_dir() / "events")

    def path_for(self, day: str) -> Path:
        return self.directory / f"queries-{day}.jsonl"

    def append(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        day = event["received_at"][:10]
        with self._lock:
            try:
                handle = self._open(day)
                handle.write(line + "\n")
                handle.flush()
            except OSError as exc:
                # Запрос уже обслужен, и ронять ответ из-за журнала незачем.
                # Но молчать тоже нельзя: отказ виден числом в `/health` и
                # строкой в выводе процесса.
                self.failed += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("не записано событие %s: %s", event.get("request_id"), exc)
                self._close()
            else:
                self.written += 1

    def _open(self, day: str):
        if self._handle is None or day != self._day:
            self._close()
            self.directory.mkdir(parents=True, exist_ok=True)
            self._handle = self.path_for(day).open("a", encoding="utf-8")
            self._day = day
        return self._handle

    def _close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None
                self._day = ""

    def close(self) -> None:
        with self._lock:
            self._close()

    def stats(self) -> dict:
        body = {"dir": str(self.directory), "written": self.written, "failed": self.failed}
        if self.failed:
            body["last_error"] = self.last_error
        return body
