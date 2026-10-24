"""Ручка `/answer` в двух режимах (SPEC, раздел 5; PLAN, задачи 7.1 и 7.2).

Сервис поднимается лексическим: плотная часть тянет torch и весы модели, а
контракт ручки от них не зависит. Генеративный режим получает подставную
модель: платных вызовов в тестах нет.
"""

import dataclasses
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rufinrag.answer import EXCERPTS, GENERATE
from rufinrag.api import app, state
from rufinrag.corpus import corpus_path
from rufinrag.generation import GenerationSettings

pytestmark = pytest.mark.answer

ROOT = Path(__file__).resolve().parents[1]

IN_CORPUS = "порядок формирования резервов на возможные потери по ссудам"
# Недостижимый порог опоры вместо вопроса не по теме: набор таких вопросов
# решением автора из проекта убран, и путь отказа проверяется механически
# (пояснение в tests/test_answer.py).
UNREACHABLE = 1.01


@pytest.fixture(scope="module")
def client():
    if not corpus_path().exists():
        pytest.skip("нет корпуса, собрать: python scripts/import_corpus.py")
    os.environ["RUFINRAG_DISABLE_DENSE"] = "1"
    os.environ["RUFINRAG_EVENT_LOG"] = "off"
    with TestClient(app) as test_client:
        yield test_client
    os.environ.pop("RUFINRAG_DISABLE_DENSE", None)
    os.environ.pop("RUFINRAG_EVENT_LOG", None)


@pytest.fixture
def model(client):
    """Подставная модель на время одного теста, настоящий клиент не собирается."""
    answer_service = state["answer"]
    replies: list[str] = []
    answer_service.generation = GenerationSettings(
        base_url="http://stub",
        model="stub-7b",
        key="stub",
        max_tokens=256,
        temperature=0.0,
        timeout_seconds=1.0,
    )
    answer_service.completer = lambda system, user: replies[-1]
    yield replies
    answer_service.generation = None
    answer_service.completer = None


@pytest.fixture
def no_support(client):
    """Порог опоры на время одного теста поднят выше достижимого.

    Так проверяется путь отказа у ручки: сервис поднят как в бою, и правка
    снимается после теста, иначе следующий тест получил бы отказ на все.
    """
    answer_service = state["answer"]
    было = answer_service.settings
    answer_service.settings = dataclasses.replace(было, threshold=UNREACHABLE)
    yield
    answer_service.settings = было


def test_health_shows_the_answer_layer_and_its_modes(client):
    body = client.get("/health").json()["answer"]
    assert body["status"] == "ok"
    assert body["modes"] == [EXCERPTS]
    assert body["generation"] == "off"
    assert body["generation_off_reason"]
    assert body["support_feature"]
    assert body["measured_in"] == "results/answer.json"
    measured = json.loads((ROOT / "results/answer.json").read_text(encoding="utf-8"))
    assert body["support_threshold"] == measured["support_threshold"]


def test_excerpts_mode_answers_with_citations(client):
    """Режим без ключа отвечает выдержками, и ссылка стоит у каждой."""
    body = client.post("/answer", json={"query": IN_CORPUS, "top": 5}).json()
    assert body["answered"] is True
    assert body["mode"] == EXCERPTS
    assert body["answer_source"] == "выдержки из найденных фрагментов"
    assert body["citations"]
    assert "refusal" not in body
    for citation in body["citations"]:
        assert citation["url"].startswith("https://www.cbr.ru/")
        assert citation["reference"].startswith(citation["act"])
        assert citation["quote"]
        assert citation["chunk_chars"][1] > citation["chunk_chars"][0]
    assert body["corpus_version"].startswith("chunks")
    assert body["config_version"]
    assert body["answer_version"]
    assert body["timings_ms"]["total"] > 0
    assert body["support"]["passed"] is True


def test_query_without_support_gets_a_refusal_not_an_answer(client, no_support):
    """Выход задачи 7.1: отказ приходит ответом 200 с числами, а не выдумкой."""
    response = client.post("/answer", json={"query": IN_CORPUS})
    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is False
    assert body["citations"] == []
    assert body["answer"] == ""
    assert "опоры в корпусе нет" in body["refusal"]
    assert body["support"]["passed"] is False
    assert body["support"]["threshold"] == body["support"]["threshold"]


def test_generate_mode_is_a_honest_503_without_a_key(client):
    """Режима нет, значит сервис так и говорит, а не отвечает выдумкой."""
    response = client.post("/answer", json={"query": IN_CORPUS, "mode": GENERATE})
    assert response.status_code == 503
    assert "RUFINRAG_LLM_API_KEY" in response.json()["detail"]


def test_generate_mode_answers_from_the_found_fragments(client, model):
    """Выход задачи 7.2: сервис отвечает в обоих режимах."""
    model.append(json.dumps({"answer": "Резерв формируется по Положению 590-П.", "used": [1, 2]}))
    body = client.post("/answer", json={"query": IN_CORPUS, "mode": GENERATE, "top": 5}).json()
    assert body["answered"] is True
    assert body["mode"] == GENERATE
    assert body["answer"] == "Резерв формируется по Положению 590-П."
    assert body["answer_source"] == "модель stub-7b по найденным фрагментам"
    assert len(body["citations"]) == 2
    assert body["timings_ms"]["generation"] >= 0
    assert client.get("/health").json()["answer"]["modes"] == [EXCERPTS, GENERATE]


def test_generated_answer_about_an_act_outside_the_hits_is_refused(client, model):
    model.append(json.dumps({"answer": "Смотрите Указание № 9999-У.", "used": [1]}))
    body = client.post("/answer", json={"query": IN_CORPUS, "mode": GENERATE}).json()
    assert body["answered"] is False
    assert "вне выдачи" in body["refusal"]


def test_generate_mode_refuses_without_support_and_does_not_call_the_model(
    client, model, no_support
):
    body = client.post("/answer", json={"query": IN_CORPUS, "mode": GENERATE}).json()
    assert body["answered"] is False
    # Подставная модель отдала бы последний ответ из списка, а список пуст:
    # обращение к ней уронило бы тест по IndexError.
    assert model == []


def test_dirty_input_is_rejected_the_same_way_as_in_search(client):
    assert client.post("/answer", json={"query": "   "}).status_code == 422
    assert client.post("/answer", json={"query": "а" * 1_000_000}).status_code == 422
    assert client.post("/answer", json={"query": "норматив H1", "depth": 3}).status_code == 422
    assert client.post("/answer", json={"query": "норматив H1", "mode": "дай"}).status_code == 422
    assert client.post("/answer", json={"query": "норматив H1", "top": 0}).status_code == 422
    unknown = client.post("/answer", json={"query": "норматив H1", "config": "../secrets"})
    assert unknown.status_code == 400
    assert "нет связки" in unknown.json()["detail"]
    assert client.get("/health").json()["status"] in ("ok", "degraded")


def test_root_page_names_the_answer_endpoint(client):
    assert "/answer" in client.get("/").text
