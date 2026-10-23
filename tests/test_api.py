"""Контракты API и поведение на грязном вводе (QUALITY, раздел 3).

Сервис поднимается лексическим: плотная часть тянет torch и весы модели, и
проверять на ней контракты эндпоинтов незачем.
"""

import os
import re

import pytest
from fastapi.testclient import TestClient

from rufinrag.api import app
from rufinrag.corpus import corpus_path


@pytest.fixture(scope="module")
def client():
    if not corpus_path().exists():
        pytest.skip("нет корпуса, собрать: python scripts/import_corpus.py")
    os.environ["RUFINRAG_DISABLE_DENSE"] = "1"
    # Контракты эндпоинтов не о журнале запросов: без этого каждый тестовый
    # прогон дописывал бы строки в data/events репозитория.
    os.environ["RUFINRAG_EVENT_LOG"] = "off"
    with TestClient(app) as test_client:
        yield test_client
    os.environ.pop("RUFINRAG_DISABLE_DENSE", None)
    os.environ.pop("RUFINRAG_EVENT_LOG", None)


def test_health_reports_versions(client):
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    assert re.fullmatch(r"chunks\d+-acts\d+", body["corpus_version"])
    assert body["chunks"] > 60000
    assert body["dense"] == "off"  # фикстура поднимает сервис лексическим
    assert body["default_config"] == "hybrid"


def test_search_returns_hits_with_versions(client):
    body = client.post("/search", json={"query": "норматив достаточности капитала H1"}).json()
    assert len(body["results"]) == 10
    assert re.fullmatch(r"chunks\d+-acts\d+", body["corpus_version"])
    assert body["config_version"].startswith("hybrid")
    assert body["timings_ms"]["total"] > 0
    first = body["results"][0]
    assert first["rank"] == 1
    assert first["url"].startswith("https://www.cbr.ru/")


def test_empty_query_is_rejected(client):
    assert client.post("/search", json={"query": "   "}).status_code == 422
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_longest_real_question_is_accepted(client):
    """Предел длины запроса не должен отрезать живые вопросы практиков."""
    import json

    from rufinrag.corpus import corpus_path

    longest = max(
        (
            json.loads(line)["text"]
            for line in (corpus_path().parents[1] / "benchmark/queries.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ),
        key=len,
    )
    assert len(longest) > 3000
    assert client.post("/search", json={"query": longest}).status_code == 200


def test_megabyte_query_is_rejected_and_service_survives(client):
    response = client.post("/search", json={"query": "а" * 1_000_000})
    assert response.status_code == 422
    assert client.get("/health").json()["status"] in ("ok", "degraded")


def test_unknown_field_is_rejected(client):
    response = client.post("/search", json={"query": "норматив H1", "depth": 5})
    assert response.status_code == 422


def test_unknown_config_is_rejected(client):
    response = client.post("/search", json={"query": "норматив H1", "config": "../secrets"})
    assert response.status_code == 400
    assert "нет связки" in response.json()["detail"]


def test_markup_and_control_characters_are_data(client):
    """Текст запроса возвращается как данные, в разметку он не превращается."""
    query = "<script>alert(1)</script> норматив H1\x00"
    response = client.post("/search", json={"query": query})
    assert response.status_code == 200
    assert response.json()["query"] == query
    assert response.headers["content-type"].startswith("application/json")


def test_top_is_bounded(client):
    assert client.post("/search", json={"query": "норматив H1", "top": 0}).status_code == 422
    assert client.post("/search", json={"query": "норматив H1", "top": 500}).status_code == 422
    body = client.post("/search", json={"query": "норматив H1", "top": 3}).json()
    assert len(body["results"]) == 3


def test_lexical_only_install_is_healthy_not_broken(client):
    """Базовая установка без плотной части это рабочий режим, а не поломка.

    Иначе проверка здоровья на хостинге считает рабочий сервис сломанным.
    """
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["dense"] == "off"
    assert body["dense_off_reason"]
    assert body["degraded"] == []


def test_root_example_survives_the_f_string(client):
    """Обратный слеш в примере curl один раз уже съелся переносом строки."""
    page = client.get("/").text
    example = page[page.index("<pre>") : page.index("</pre>")]
    assert example.count("\\\n") == 2, "пример curl собрался в одну строку"
    assert '"query"' in example
