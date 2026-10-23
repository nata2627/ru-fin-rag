"""Поток журнала с хостинга в репозиторий: рост и отсутствие дублей.

Два свойства, из-за которых этот путь вообще написан. Строк в репозитории после
выгрузки становится больше, а повторная выгрузка не добавляет ничего: сервис
отдает те же строки (он ничего не удаляет), и дубли опознаются по ключу
события, а не по позиции в файле.

Выгрузка по настоящей сети проверяется репетицией `scripts/rehearse_drain.py`:
она поднимает `uvicorn` процессом, а числа кладет в `results/events-rehearsal.json`.
Здесь клиент ходит в приложение напрямую, поэтому прогон тестов быстрый.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rufinrag import api, events
from rufinrag.corpus import corpus_path
from rufinrag.search import Hit, SearchResult

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("drain_events", ROOT / "scripts/drain_events.py")
drain_events = importlib.util.module_from_spec(spec)
sys.modules["drain_events"] = drain_events
spec.loader.exec_module(drain_events)

TOKEN = "f" * 32
ENV = ("RUFINRAG_EVENT_DIR", "RUFINRAG_EVENTS_TOKEN", "RUFINRAG_DISABLE_DENSE")


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    """Сервис с живым журналом в каталоге процесса и заданным секретом выдачи."""
    if not corpus_path().exists():
        pytest.skip("нет корпуса, собрать: python scripts/import_corpus.py")
    directory = tmp_path_factory.mktemp("service-disk")
    previous = {name: os.environ.get(name) for name in ENV}
    os.environ.update(
        RUFINRAG_EVENT_DIR=str(directory),
        RUFINRAG_EVENTS_TOKEN=TOKEN,
        RUFINRAG_DISABLE_DENSE="1",
    )
    with TestClient(api.app) as client:
        yield client, directory
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def line(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


def query_line(received_at: str, request_id: str | None = None) -> str:
    result = SearchResult(
        query="норматив достаточности капитала H1",
        config="bm25",
        config_version="bm25+cand50",
        corpus_version="chunks62594-acts1554",
        hits=[
            Hit(
                chunk_id="chunk-1",
                score=1.0,
                rank=1,
                act_id="act-1",
                act="Указание Банка России",
                title="Акт",
                section="1.1",
                url="https://www.cbr.ru/act-1",
                text="фрагмент",
            )
        ],
        timings_ms={"bm25": 1.0, "total": 1.2},
        degraded=[],
    )
    return line(
        events.answered(request_id or events.new_request_id(), received_at, 10, result, "0.1.0")
    )


def lines_in(directory: Path) -> int:
    return drain_events.survey(directory)["lines"]


def test_dump_does_not_exist_without_a_secret(service, monkeypatch):
    """Ручка над текстами чужих запросов выключена, пока секрета нет."""
    client, _ = service
    monkeypatch.delenv("RUFINRAG_EVENTS_TOKEN")
    assert client.get("/events").status_code == 404


def test_wrong_secret_is_refused(service):
    client, _ = service
    assert client.get("/events").status_code == 403
    assert client.get("/events", headers={"X-Events-Token": "a" * 32}).status_code == 403


def test_dump_gives_the_lines_the_service_wrote(service):
    client, directory = service
    answer = client.post("/search", json={"query": "порядок формирования резервов", "top": 3})
    rated = client.post(
        "/feedback", json={"request_id": answer.json()["request_id"], "rating": "up"}
    )
    assert rated.status_code == 202

    response = client.get("/events", headers={"X-Events-Token": TOKEN})
    assert response.status_code == 200
    body = [row for row in response.text.splitlines() if row.strip()]
    assert int(response.headers["X-Events-Lines"]) == len(body)
    assert response.headers["X-Events-Truncated"] == "0"
    assert int(response.headers["X-Events-Files"]) == len(list(directory.glob("*.jsonl")))
    assert sum(1 for row in body if answer.json()["request_id"] in row) == 2


def test_dump_carries_nothing_but_the_query_the_answer_and_technical_fields(service):
    """Решение 12: ни заголовков, ни адресов, ни куки, даже если клиент их прислал."""
    client, _ = service
    client.post(
        "/search",
        json={"query": "идентификация клиента", "top": 3},
        headers={"User-Agent": "rufinrag-test", "Cookie": "session=secret", "X-Real-Ip": "8.8.8.8"},
    )
    response = client.get("/events", headers={"X-Events-Token": TOKEN})
    known = {
        *events.QUERY_FIELDS,
        *events.ANSWERED_FIELDS,
        *events.FEEDBACK_FIELDS,
        "reason",
        "query_truncated_to",
    }
    for row in response.text.splitlines():
        if not row.strip():
            continue
        assert set(json.loads(row)) <= known, f"лишнее поле в строке журнала: {row[:200]}"
    assert "8.8.8.8" not in response.text
    assert "session=secret" not in response.text


def test_dump_is_bounded_and_says_so(service, monkeypatch):
    client, _ = service
    monkeypatch.setattr(api, "MAX_DRAIN_LINES", 1)
    response = client.get("/events", headers={"X-Events-Token": TOKEN})
    assert response.headers["X-Events-Lines"] == "1"
    assert response.headers["X-Events-Truncated"] == "1"


def test_lines_grow_and_a_second_drain_adds_nothing(tmp_path):
    directory = tmp_path / "events"
    offered = [query_line(f"2026-10-23T1{index}:00:00.000Z") for index in range(5)]

    first = drain_events.merge(offered, directory)
    assert first == {
        "added": 5,
        "duplicates": 0,
        "files_touched": ["queries-2026-10-23.jsonl"],
    }
    assert lines_in(directory) == 5

    second = drain_events.merge(offered, directory)
    assert second["added"] == 0
    assert second["duplicates"] == 5
    assert second["files_touched"] == []
    assert lines_in(directory) == 5


def test_a_drain_after_new_traffic_adds_only_what_is_new(tmp_path):
    directory = tmp_path / "events"
    offered = [query_line(f"2026-10-23T1{index}:00:00.000Z") for index in range(5)]
    drain_events.merge(offered, directory)

    grown = [*offered, query_line("2026-10-23T19:00:00.000Z")]
    third = drain_events.merge(grown, directory)
    assert third["added"] == 1
    assert third["duplicates"] == 5
    assert lines_in(directory) == 6


def test_the_query_and_its_rating_are_two_events_with_one_request_id(tmp_path):
    """Ключ дубля это не `request_id` сам по себе: оценка приходит с тем же."""
    directory = tmp_path / "events"
    request_id = events.new_request_id()
    offered = [
        query_line("2026-10-23T10:00:00.000Z", request_id),
        line(events.feedback(request_id, "up", "2026-10-23T10:00:01.000Z")),
    ]
    assert drain_events.merge(offered, directory)["added"] == 2
    assert drain_events.merge(offered, directory)["added"] == 0
    survey = drain_events.survey(directory)
    assert survey["query_events"] == 1
    assert survey["feedback_events"] == 1


def test_the_file_is_chosen_by_the_day_of_the_event(tmp_path):
    """Сутки берутся из `received_at`, а не из файла, который их отдал."""
    directory = tmp_path / "events"
    offered = [
        query_line("2026-10-22T23:59:59.000Z"),
        query_line("2026-10-23T00:00:01.000Z"),
    ]
    assert drain_events.merge(offered, directory)["files_touched"] == [
        "queries-2026-10-22.jsonl",
        "queries-2026-10-23.jsonl",
    ]
    assert drain_events.survey(directory)["days"] == 2


def test_loss_window_is_seen_when_the_process_is_younger_than_the_gap():
    """Перезапуск между выгрузками значит, что события до него пропали."""
    previous = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3600))
    assert drain_events.restart_window({"uptime_seconds": 60.0}, previous) is True
    assert drain_events.restart_window({"uptime_seconds": 7200.0}, previous) is False
    # Первой выгрузке сравнивать не с чем, и это не «потери не было».
    assert drain_events.restart_window({"uptime_seconds": 60.0}, None) is None
    assert drain_events.restart_window({}, previous) is None


def test_an_unreachable_service_does_not_break_the_run(tmp_path):
    """Спящий сервис это не поломка: красный main останавливает приемку."""
    state = drain_events.drain(
        "http://127.0.0.1:9", TOKEN, tmp_path / "events", {}, timeout=1, attempts=1
    )
    assert state["last_drain"]["status"] == "unreachable"
    assert state["failed_drains_in_a_row"] == 1
    assert state["drains"] == 1
    assert state["repo"]["lines"] == 0
