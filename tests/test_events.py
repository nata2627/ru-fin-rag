"""Схема журнала запросов и полнота полей (AGENTS.md, раздел 2).

`received_at` это момент события, а не момент записи файла: именно эта
подмена в части А привела к тому, что во всех строках стояло одно время.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from rufinrag import events
from rufinrag.config import data_dir
from rufinrag.retrieval.text import tokenize
from rufinrag.search import Hit, SearchResult

ROOT = Path(__file__).resolve().parents[1]


def _hit(rank: int) -> Hit:
    return Hit(
        chunk_id=f"chunk-{rank}",
        score=1.0 / rank,
        rank=rank,
        act_id="act-1",
        act="Указание Банка России № 1-У от 01.01.2020",
        title="Акт для теста",
        section="1.1",
        url="https://www.cbr.ru/act-1",
        text="текст фрагмента",
    )


def _result(query: str = "норматив достаточности капитала H1") -> SearchResult:
    return SearchResult(
        query=query,
        config="hybrid",
        config_version="hybrid+hybrid+cand50+rufin",
        corpus_version="chunks62594-acts1554",
        hits=[_hit(1), _hit(2)],
        timings_ms={"bm25": 1.2, "total": 1.2},
        degraded=[],
    )


def test_answered_event_has_every_field_and_no_none():
    event = events.answered(
        request_id=events.new_request_id(),
        received_at=events.utc_now(),
        top=10,
        result=_result(),
        service_version="0.1.0",
    )
    for name in (*events.QUERY_FIELDS, *events.ANSWERED_FIELDS):
        assert name in event, f"нет поля {name}"
        assert event[name] is not None, f"поле {name} равно None"


def test_rejected_event_has_query_fields_and_reason_but_no_answered_fields():
    event = events.rejected(
        request_id=events.new_request_id(),
        received_at=events.utc_now(),
        query="норматив H1",
        top=10,
        status=422,
        reason="query: поле пустое",
        service_version="0.1.0",
    )
    for name in events.QUERY_FIELDS:
        assert name in event, f"нет поля {name}"
        assert event[name] is not None, f"поле {name} равно None"
    assert event["reason"]
    for name in events.ANSWERED_FIELDS:
        assert name not in event, f"у отвергнутого запроса не должно быть поля {name}"


def test_feedback_event_has_exactly_feedback_fields():
    event = events.feedback(events.new_request_id(), "up")
    assert set(event.keys()) == set(events.FEEDBACK_FIELDS)
    for name in events.FEEDBACK_FIELDS:
        assert event[name] is not None


def test_query_normalized_matches_lexical_tokenizer():
    query = "Норматив Н1 достаточности КАПИТАЛА, 590-П"
    event = events.answered(
        request_id=events.new_request_id(),
        received_at=events.utc_now(),
        top=10,
        result=_result(query=query),
        service_version="0.1.0",
    )
    assert event["query_normalized"] == " ".join(tokenize(query))


def test_long_query_is_truncated_in_the_log_line_but_length_is_real():
    query = "а" * 1_000_000
    event = events.rejected(
        request_id=events.new_request_id(),
        received_at=events.utc_now(),
        query=query,
        top=10,
        status=422,
        reason="слишком длинный запрос",
        service_version="0.1.0",
    )
    assert len(event["query"]) <= events.MAX_LOGGED_CHARS
    assert event["query_chars"] == 1_000_000
    assert event["query_truncated_to"] == events.MAX_LOGGED_CHARS


def test_received_at_is_the_moment_of_collection_not_of_writing(tmp_path):
    log = events.EventLog(tmp_path)
    collected_at = events.utc_now()
    time.sleep(0.2)
    event = events.feedback(events.new_request_id(), "up", received_at=collected_at)

    log.append(event)

    written = json.loads(log.path_for(collected_at[:10]).read_text(encoding="utf-8").strip())
    assert written["received_at"] == collected_at
    assert written["received_at"] != events.utc_now()


def test_two_events_collected_apart_get_different_received_at(tmp_path):
    log = events.EventLog(tmp_path)
    first_collected = events.utc_now()
    time.sleep(0.2)
    second_collected = events.utc_now()
    assert first_collected != second_collected

    log.append(events.feedback(events.new_request_id(), "up", received_at=first_collected))
    log.append(events.feedback(events.new_request_id(), "down", received_at=second_collected))

    lines = log.path_for(first_collected[:10]).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["received_at"] for line in lines] == [
        first_collected,
        second_collected,
    ]


def test_days_split_into_separate_files_by_utc_date(tmp_path):
    log = events.EventLog(tmp_path)
    log.append(events.feedback("a" * 32, "up", received_at="2026-10-22T23:59:59.000Z"))
    log.append(events.feedback("b" * 32, "down", received_at="2026-10-23T00:00:01.000Z"))

    first_day = log.path_for("2026-10-22").read_text(encoding="utf-8").splitlines()
    second_day = log.path_for("2026-10-23").read_text(encoding="utf-8").splitlines()
    assert len(first_day) == 1
    assert len(second_day) == 1


def test_write_failure_is_counted_not_swallowed(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("тут обычный файл, а не каталог", encoding="utf-8")
    log = events.EventLog(blocked)

    log.append(events.feedback(events.new_request_id(), "up"))

    assert log.failed == 1
    assert log.last_error
    stats = log.stats()
    assert stats["failed"] == 1
    assert stats["last_error"]


def test_each_line_is_parseable_json_with_cyrillic_not_escaped(tmp_path):
    log = events.EventLog(tmp_path)
    event = events.rejected(
        request_id=events.new_request_id(),
        received_at=events.utc_now(),
        query="норматив достаточности капитала",
        top=10,
        status=422,
        reason="запрос состоит из пробелов",
        service_version="0.1.0",
    )
    log.append(event)

    raw = log.path_for(event["received_at"][:10]).read_text(encoding="utf-8")
    lines = [line for line in raw.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "норматив" in raw
    assert "\\u" not in raw
    parsed = json.loads(lines[0])
    assert parsed["request_id"] == event["request_id"]


def test_from_env_off_disables_the_log(monkeypatch):
    monkeypatch.setenv("RUFINRAG_EVENT_LOG", "off")
    assert events.EventLog.from_env() is None


def test_default_event_dir_is_the_service_data_dir(monkeypatch):
    """В рабочем режиме события пишутся в data/events и едут в репозиторий."""
    monkeypatch.delenv("RUFINRAG_EVENT_LOG", raising=False)
    monkeypatch.delenv("RUFINRAG_EVENT_DIR", raising=False)
    log = events.EventLog.from_env()
    assert log is not None
    assert log.directory == data_dir() / "events"


def test_the_test_run_writes_events_outside_the_repo(event_dir_outside_the_repo):
    """Обратная сторона предыдущего теста: прогон тестов в дерево не пишет.

    Один раз уже написал: сверка задачи 4.1 нашла в data/events 77 событий
    тестов. Каталог ставит tests/conftest.py на весь прогон, поэтому правило
    держится и на тестах, которые поднимают сервис подпроцессом.
    """
    log = events.EventLog.from_env()
    assert log is not None
    assert log.directory == event_dir_outside_the_repo
    assert ROOT not in log.directory.parents


def test_from_env_uses_event_dir_override(tmp_path, monkeypatch):
    monkeypatch.delenv("RUFINRAG_EVENT_LOG", raising=False)
    monkeypatch.setenv("RUFINRAG_EVENT_DIR", str(tmp_path / "custom-events"))
    log = events.EventLog.from_env()
    assert log is not None
    assert log.directory == tmp_path / "custom-events"


# Тесты уровня API: свой TestClient с журналом в каталоге прогона (tests/
# conftest.py), чтобы контрактные тесты не писали события в data/events
# репозитория.


def _today_lines(event_dir) -> list[str]:
    path = event_dir / f"queries-{events.utc_now()[:10]}.jsonl"
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def api_client(event_dir_outside_the_repo):
    from rufinrag.corpus import corpus_path

    if not corpus_path().exists():
        pytest.skip("нет корпуса, собрать: python scripts/import_corpus.py")

    # Каталог журнала свой фикстура не ставит: его ставит conftest на весь
    # прогон. Своя переменная здесь однажды уже снялась в teardown вместе с
    # общей, и события следующего теста ушли в рабочее дерево.
    os.environ["RUFINRAG_DISABLE_DENSE"] = "1"
    from fastapi.testclient import TestClient

    from rufinrag.api import app

    with TestClient(app) as test_client:
        yield test_client, event_dir_outside_the_repo
    os.environ.pop("RUFINRAG_DISABLE_DENSE", None)


def test_successful_search_writes_exactly_one_event_with_matching_request_id(api_client):
    client, event_dir = api_client
    before = len(_today_lines(event_dir))

    response = client.post("/search", json={"query": "норматив достаточности капитала H1"})
    body = response.json()

    lines = _today_lines(event_dir)
    assert len(lines) == before + 1
    logged = json.loads(lines[-1])
    assert logged["request_id"] == body["request_id"]
    assert logged["outcome"] == "ok"


def test_rejected_search_is_logged_with_reason(api_client):
    client, event_dir = api_client
    before = len(_today_lines(event_dir))

    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 422

    lines = _today_lines(event_dir)
    assert len(lines) == before + 1
    logged = json.loads(lines[-1])
    assert logged["outcome"] == "rejected"
    assert logged["reason"]


def test_feedback_with_known_request_id_is_logged_and_accepted(api_client):
    client, event_dir = api_client
    search_response = client.post("/search", json={"query": "норматив H1"})
    request_id = search_response.json()["request_id"]
    before = len(_today_lines(event_dir))

    response = client.post("/feedback", json={"request_id": request_id, "rating": "up"})
    assert response.status_code == 202

    lines = _today_lines(event_dir)
    assert len(lines) == before + 1
    logged = json.loads(lines[-1])
    assert logged["kind"] == "feedback"
    assert logged["request_id"] == request_id
    assert logged["rating"] == "up"


def test_feedback_with_garbage_request_id_is_rejected_by_schema(api_client):
    client, event_dir = api_client
    before = len(_today_lines(event_dir))

    response = client.post("/feedback", json={"request_id": "не похоже на id", "rating": "up"})

    assert response.status_code == 422
    assert len(_today_lines(event_dir)) == before


def test_health_reports_event_log_counts(api_client):
    client, event_dir = api_client
    client.post("/search", json={"query": "норматив H1"})

    body = client.get("/health").json()

    assert body["event_log"]["dir"] == str(event_dir)
    assert body["event_log"]["written"] >= 1
