"""Роутер на инструментах: поиск по актам, поиск по графу, уточнение, отказ.

Задача 8.2. Граф состояний собран на LangGraph, как требует постановка, а
маршрут выбирается правилом, записанным до кода (`orchestration/EXPERIMENTS.md`,
S25). Правилом, а не вызовом модели, по двум причинам: ключа у проекта нет и по
требованию автономности не будет, а признаки, на которых решение принимается,
сервис уже считает для отказа и для монитора. Такой маршрут объясним строкой в
журнале и повторяется на той же выдаче.

Библиотека не идет в основные зависимости: `langgraph` это 70 МБ в окружении и
66,3 МБ памяти на импорт при 215 МБ пика сервиса на лексической связке и 512 МБ
бесплатного тарифа. Поэтому модуль импортирует ее внутри сборки, а сервис
поднимает роутер только при `RUFINRAG_ROUTER=1` (группа зависимостей `router`).

Порядок узлов и состояние держит граф, а не цепочка if в обработчике запроса:
маршрут один, его имя и причина попадают в ответ и в журнал, и добавить пятый
инструмент значит добавить узел и ветку, а не править разбор условий.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict

from .answer import Answer, AnswerService
from .config import load_catalog
from .references import act_numbers, normalize
from .search import SearchResult

REFUSE = "отказ"
CLARIFY = "уточнение"
GRAPH = "граф"
ACTS = "акты"
ROUTES = (REFUSE, CLARIFY, GRAPH, ACTS)

# Правило уточнения из S25: запрос короткий (столько его слов нашлось в словаре
# корпуса) и первые цитаты собраны из такого числа разных актов.
CLARIFY_MAX_TERMS = 3
CLARIFY_MIN_ACTS = 4
# Связка с расширением по графу. Она есть в каталоге и по умолчанию выключена:
# на всех 991 запросах расширение не дало ничего (S20), а здесь оно работает как
# инструмент для вопроса, в котором акт назван номером.
GRAPH_CONFIG = "hybrid-graph"
# Инструмент графа по умолчанию выключен, и это результат замера, а не
# осторожность. На 84 запросах, где акт назван номером, расширение по
# цитированию не изменило ответ ни разу: в режиме резерва оно занимает
# последнее место выдачи, а в ответ идут первые пять цитат. Посчитанный
# отдельно вариант, где добавленный фрагмент попадает в пятерку, дал 59
# корректных ссылок из 84 против 60 у прямого поиска, то есть прибавки нет и
# там. Цена маршрута при этом реальная: второй поиск на запрос и 7,8 с на
# первое обращение, пока собирается граф ссылок. Включается переменной
# RUFINRAG_ROUTER_GRAPH=1 (замер: results/router.json, EXPERIMENTS S25).
GRAPH_ENV = "RUFINRAG_ROUTER_GRAPH"


class State(TypedDict, total=False):
    """Состояние графа. Узлы дописывают поля, а не передают кортежи."""

    query: str
    top: int
    route: str
    reason: str
    search: SearchResult
    answer: Answer
    clarify: dict[str, Any]
    trace: list[str]


@dataclass(frozen=True, slots=True)
class RouterRun:
    query: str
    route: str
    reason: str
    answer: Answer | None
    clarify: dict[str, Any] | None
    trace: tuple[str, ...]
    timings_ms: dict[str, float]


@dataclass
class Router:
    """Граф инструментов над слоем ответа. Собирается `build`."""

    answer: AnswerService
    graph: Any = None
    graph_config_available: bool = True
    settings: dict[str, int] = field(
        default_factory=lambda: {
            "clarify_max_terms": CLARIFY_MAX_TERMS,
            "clarify_min_acts": CLARIFY_MIN_ACTS,
        }
    )

    @classmethod
    def build(cls, answer: AnswerService, graph: bool | None = None):
        # Импорт внутри сборки, а не в начале модуля: без роутера сервис не
        # должен платить за библиотеку ни памятью, ни секундой старта.
        from langgraph.graph import END, START, StateGraph

        enabled = os.environ.get(GRAPH_ENV, "") == "1" if graph is None else graph
        router = cls(
            answer=answer,
            graph_config_available=enabled and GRAPH_CONFIG in load_catalog().configs,
        )
        builder = StateGraph(State)
        builder.add_node("triage", router._triage)
        builder.add_node(REFUSE, router._refuse)
        builder.add_node(CLARIFY, router._clarify)
        builder.add_node(GRAPH, router._graph)
        builder.add_node(ACTS, router._acts)
        builder.add_edge(START, "triage")
        builder.add_conditional_edges(
            "triage", lambda state: state["route"], {name: name for name in ROUTES}
        )
        for name in ROUTES:
            builder.add_edge(name, END)
        router.graph = builder.compile()
        return router

    def run(self, query: str, top: int = 10) -> RouterRun:
        started = time.monotonic()
        state: State = self.graph.invoke({"query": query, "top": top, "trace": []})
        answer = state.get("answer")
        timings = dict(answer.timings_ms) if answer is not None else {}
        timings["router"] = round((time.monotonic() - started) * 1000, 3)
        return RouterRun(
            query=query,
            route=state["route"],
            reason=state.get("reason", ""),
            answer=answer,
            clarify=state.get("clarify"),
            trace=tuple(state.get("trace", ())),
            timings_ms=timings,
        )

    def _triage(self, state: State) -> State:
        """Один поиск и одна опора на запрос: по ним и выбирается маршрут.

        Выдача кладется в состояние, потому что три из четырех инструментов
        работают ровно с ней. Второй такой же поиск был бы заплачен за выбор
        ветки, а не за ответ.
        """
        query, top = state["query"], state.get("top", 10)
        result = self.answer.search.search(query, top=top)
        support, _ = self.answer.support_of(result, query)
        route, reason = self._choose(query, result, support.passed, support.reason)
        return {
            "search": result,
            "route": route,
            "reason": reason,
            "trace": [*state.get("trace", []), f"triage: {route}"],
        }

    def _choose(
        self, query: str, result: SearchResult, supported: bool, refusal: str
    ) -> tuple[str, str]:
        """Правило S25 целиком, в том же порядке. Первое сработавшее побеждает."""
        if not supported:
            return REFUSE, refusal
        limit = self.answer.settings.citations
        acts = {hit.act_id for hit in result.hits[:limit]}
        terms = self._known_terms(query)
        if (
            terms <= self.settings["clarify_max_terms"]
            and len(acts) >= self.settings["clarify_min_acts"]
        ):
            return (
                CLARIFY,
                f"в запросе {terms} известных слов, а первые {limit} фрагментов "
                f"собраны из {len(acts)} разных актов",
            )
        named = self._named_act(result)
        if named:
            return GRAPH, f"в запросе назван акт {named}, у него есть ссылки внутрь корпуса"
        return ACTS, "обычный запрос, прямой поиск по актам"

    def _known_terms(self, query: str) -> int:
        """Сколько слов запроса есть в словаре корпуса.

        Считается тем же кодом, что у монитора и у отказа, иначе «короткий
        запрос» в роутере значил бы не то, что в метрике.
        """
        from .monitoring import shape_of_text

        return shape_of_text(query, self.answer.search.bm25.vocab).found

    def _named_act(self, result: SearchResult) -> str:
        """Номер акта из запроса, который есть в корпусе и на кого-то ссылается.

        Номер берется из текста запроса тем же выражением, которым часть А
        читает ссылки между актами. Акта может не быть в корпусе (вопрос про
        федеральный закон), и тогда маршрут графа не нужен: расширять нечего.
        """
        if not self.graph_config_available:
            return ""
        numbers = act_numbers(result.query)
        if not numbers:
            return ""
        wanted = {normalize(number) for number in numbers}
        corpus = self.answer.search.corpus
        seen: dict[str, str] = {}
        for hit in result.hits:
            if hit.act_id not in seen:
                seen[hit.act_id] = normalize(corpus.get(hit.chunk_id).act.number)
        named = {act_id: number for act_id, number in seen.items() if number in wanted}
        if not named:
            return ""
        graph = self.answer.search.act_graph()
        for act_id, number in named.items():
            if graph.neighbors([act_id], "forward"):
                return number
        return ""

    def _refuse(self, state: State) -> State:
        answer = self.answer.answer_from(state["search"], state["query"])
        return {
            "answer": answer,
            "trace": [*state.get("trace", []), "инструмент отказа"],
        }

    def _clarify(self, state: State) -> State:
        """Уточняющий вопрос: назвать найденные акты и попросить выбрать один.

        Ответ при этом тоже собирается: оставить человека с одним вопросом и
        пустым экраном хуже, чем дать выдержки и спросить рядом.
        """
        result = state["search"]
        limit = self.answer.settings.citations
        acts: dict[str, str] = {}
        for hit in result.hits[:limit]:
            acts.setdefault(hit.act_id, hit.act)
        answer = self.answer.answer_from(result, state["query"])
        return {
            "answer": answer,
            "clarify": {
                "вопрос": "Вопрос можно уточнить: найденное лежит в разных документах. "
                "О каком из них речь?",
                "акты": [{"act_id": act_id, "act": name} for act_id, name in acts.items()],
            },
            "trace": [*state.get("trace", []), "инструмент уточнения"],
        }

    def _graph(self, state: State) -> State:
        """Поиск с расширением по цитированию.

        Это второй поиск на запрос, и он единственный такой: граф ссылок
        собирается лениво при первом обращении (15,7 с на маке), поэтому цена
        первого запроса по этому маршруту называется в замере 8.3, а не
        прячется.
        """
        result = self.answer.search.search(
            state["query"], top=state.get("top", 10), config=GRAPH_CONFIG
        )
        answer = self.answer.answer_from(result, state["query"])
        return {
            "search": result,
            "answer": answer,
            "trace": [*state.get("trace", []), "инструмент поиска по графу"],
        }

    def _acts(self, state: State) -> State:
        answer = self.answer.answer_from(state["search"], state["query"])
        return {
            "answer": answer,
            "trace": [*state.get("trace", []), "инструмент поиска по актам"],
        }
