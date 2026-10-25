"""Роутер на инструментах: маршрут на каждый инструмент и то, что ломается тихо.

Выход задачи 8.2 это тест на каждый из четырех маршрутов, включая отказ, и он
здесь есть дважды. Сначала правилом, без поиска: правило S25 проверяется на
поддельной выдаче, потому что порядок проверок в нем и есть вся
маршрутизация, и перестановка двух условий меняет поведение сервиса молча.
Потом живьем, на поднятом корпусе, где каждый маршрут обязан сработать на
настоящем запросе.

Главное, что проверяется помимо маршрутов: роутер не заводит своего второго
определения отказа и не делает лишнего поиска на выбор ветки.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.search import Hit, SearchResult  # noqa: E402

router_module = pytest.importorskip(
    "rufinrag.router", reason="роутер требует langgraph: pip install -e '.[router]'"
)
pytest.importorskip("langgraph", reason="нет langgraph")

from rufinrag.router import ACTS, CLARIFY, GRAPH, REFUSE, Router  # noqa: E402


def hit(rank: int, act_id: str, number: str = "1-У") -> Hit:
    return Hit(
        chunk_id=f"{act_id}#{rank:04d}",
        score=1.0 / rank,
        rank=rank,
        act_id=act_id,
        act=f"Указание Банка России № {number} от 01.01.2020",
        title="Акт для теста",
        section="1.1",
        url=f"https://www.cbr.ru/{act_id}",
        text="текст фрагмента",
    )


def result(query: str, acts: list[str]) -> SearchResult:
    return SearchResult(
        query=query,
        config="hybrid",
        config_version="hybrid+hybrid+cand50+rufin",
        corpus_version="chunks62606-acts1555",
        hits=[hit(i + 1, act_id) for i, act_id in enumerate(acts)],
        timings_ms={"total": 1.0},
        degraded=[],
    )


class FakeSettings:
    citations = 5


class FakeAnswer:
    """Слой ответа, из которого правилу нужны только настройки и словарь."""

    def __init__(self, known: int, named: str = ""):
        self.settings = FakeSettings()
        self.known = known
        self.named = named


def choose(known: int, acts: list[str], query: str = "запрос", supported: bool = True, named=""):
    plain = Router(answer=FakeAnswer(known))
    plain._known_terms = lambda _: known
    plain._named_act = lambda _: named
    return plain._choose(query, result(query, acts), supported, "опоры в корпусе нет")


def test_no_support_goes_to_refusal_and_keeps_its_reason():
    route, reason = choose(known=5, acts=["a1"], supported=False)
    assert route == REFUSE
    # Причина отказа приходит из слоя ответа: своего текста у роутера тут быть
    # не должно, иначе в проекте окажется два разных «нет ответа».
    assert reason == "опоры в корпусе нет"


def test_short_query_over_many_acts_goes_to_clarification():
    route, reason = choose(known=2, acts=["a1", "a2", "a3", "a4", "a5"])
    assert route == CLARIFY
    assert "2 известных слов" in reason and "5 разных актов" in reason


def test_short_query_over_few_acts_is_a_plain_search():
    assert choose(known=2, acts=["a1", "a2", "a3"])[0] == ACTS


def test_long_query_over_many_acts_is_a_plain_search():
    """Длинный запрос не уточняют: человек уже сказал, чего хочет."""
    assert choose(known=9, acts=["a1", "a2", "a3", "a4", "a5"])[0] == ACTS


def test_named_act_goes_to_the_graph():
    route, reason = choose(known=9, acts=["a1"], named="590-П")
    assert route == GRAPH
    assert "590-П" in reason


def test_refusal_wins_over_the_named_act():
    """Порядок проверок из S25: отказ раньше графа, иначе вопрос без опоры, в
    котором назван акт, получил бы расширенную выдачу вместо отказа."""
    assert choose(known=9, acts=["a1"], supported=False, named="590-П")[0] == REFUSE


def test_clarification_wins_over_the_named_act():
    assert choose(known=2, acts=["a1", "a2", "a3", "a4"], named="590-П")[0] == CLARIFY


@pytest.mark.answer
@pytest.mark.slow
class TestOnTheCorpus:
    """Те же четыре маршрута на поднятом корпусе, настоящими запросами."""

    @pytest.fixture(scope="class")
    def live(self):
        from rufinrag.answer import AnswerService
        from rufinrag.search import SearchService

        service = SearchService.build(with_dense=False)
        # Инструмент графа включен явно: по умолчанию он выключен замером
        # (ничего не меняет в ответе, стоит второго поиска), но маршрут обязан
        # работать, и проверяется он здесь.
        return Router.build(AnswerService.build(service), graph=True)

    def test_graph_tool_is_off_unless_asked(self):
        from rufinrag.answer import AnswerService
        from rufinrag.search import SearchService

        plain = Router.build(AnswerService.build(SearchService.build(with_dense=False)))
        assert plain.graph_config_available is False
        run = plain.run(
            "какие требования к отчетности по Положению 590-П о резервах "
            "на возможные потери по ссудам",
            top=10,
        )
        assert run.route == ACTS

    def test_plain_search_answers_with_citations(self, live):
        run = live.run("резерв по ссуде", top=10)
        assert run.route == ACTS
        assert run.answer is not None and run.answer.answered
        assert len(run.answer.citations) == live.answer.settings.citations
        assert run.clarify is None
        assert run.trace[-1] == "инструмент поиска по актам"

    def test_question_without_support_is_refused(self, live):
        run = live.run("борщ с пампушками и чесноком", top=10)
        assert run.route == REFUSE
        assert run.answer is not None and not run.answer.answered
        assert run.answer.refusal
        assert run.answer.citations == ()

    def test_short_query_is_clarified_and_still_answered(self, live):
        run = live.run("норматив", top=10)
        assert run.route == CLARIFY
        assert run.clarify is not None
        assert len(run.clarify["акты"]) >= live.settings["clarify_min_acts"]
        # Уточнение не оставляет пустой экран: выдержки приходят вместе с
        # вопросом, иначе уточнение дороже для человека, чем слабый ответ.
        assert run.answer is not None and run.answer.answered

    def test_named_act_takes_the_graph_tool(self, live):
        run = live.run(
            "какие требования к отчетности по Положению 590-П о резервах "
            "на возможные потери по ссудам",
            top=10,
        )
        assert run.route == GRAPH
        assert "590-П" in run.reason
        assert run.answer is not None and run.answer.answered
        # Выдача у этого маршрута своя, со расширением по цитированию.
        assert run.answer.search.config == "hybrid-graph"

    def test_every_route_is_reachable(self, live):
        routes = {
            live.run(query, top=10).route
            for query in (
                "резерв по ссуде",
                "борщ с пампушками и чесноком",
                "норматив",
                "какие требования к отчетности по Положению 590-П о резервах "
                "на возможные потери по ссудам",
            )
        }
        assert routes == {ACTS, REFUSE, CLARIFY, GRAPH}


@pytest.mark.answer
@pytest.mark.slow
class TestThroughTheService:
    """Ручка `/route`: выключена по умолчанию, включается переменной."""

    def test_route_is_off_without_the_flag(self, monkeypatch):
        from fastapi.testclient import TestClient

        from rufinrag.api import app

        monkeypatch.delenv("RUFINRAG_ROUTER", raising=False)
        with TestClient(app) as client:
            health = client.get("/health").json()
            assert health["router"]["status"] == "off"
            assert "RUFINRAG_ROUTER" in health["router"]["off_reason"]
            # 503, а не 404: ручка существует, выключено поведение, и причина
            # называется в ответе, как у генеративного режима.
            assert client.post("/route", json={"query": "норматив"}).status_code == 503

    def test_route_answers_and_names_the_route(self, monkeypatch):
        from fastapi.testclient import TestClient

        from rufinrag.api import app

        monkeypatch.setenv("RUFINRAG_ROUTER", "1")
        with TestClient(app) as client:
            assert client.get("/health").json()["router"]["status"] == "on"
            body = client.post("/route", json={"query": "резерв по ссуде", "top": 10}).json()
            assert body["route"] == ACTS
            assert body["answered"] and body["citations"]
            assert body["registry_version"]
            refused = client.post("/route", json={"query": "борщ с пампушками"}).json()
            assert refused["route"] == REFUSE and refused["refusal"]

    def test_unknown_field_is_an_error(self, monkeypatch):
        from fastapi.testclient import TestClient

        from rufinrag.api import app

        monkeypatch.setenv("RUFINRAG_ROUTER", "1")
        with TestClient(app) as client:
            answer = client.post("/route", json={"query": "норматив", "config": "bm25"})
            assert answer.status_code == 422
