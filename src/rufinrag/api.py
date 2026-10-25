"""HTTP-интерфейс сервиса.

В каждом ответе стоят версия корпуса и версия связки поиска: без них сравнение
конфигураций и регрессия качества не читаются, а на хостинге не видно, что
именно развернуто (SPEC, раздел 3).

Индексы читаются один раз при старте процесса, а не на запрос: корпус это
62 594 фрагмента, и его чтение занимает секунду.
"""

from __future__ import annotations

import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from . import __version__, events, registry
from .answer import EXCERPTS, GENERATE, AnswerError, AnswerService
from .config import ConfigError, env, live_mode, load_catalog, project_root
from .corpus import CorpusError
from .events import EventLog
from .router import ROUTES
from .search import SearchError, SearchService

# Предел длины запроса взят по эталонному набору: самый длинный живой вопрос
# практика в нем 3067 символов, а предел 1000 отрезал 22 запроса из 1182. Это
# нашел нагрузочный замер: сервис отвечал 422 на настоящие вопросы.
MAX_QUERY_CHARS = 4000
MAX_TOP = 50

# Предел одной выдачи журнала. Ответ целиком собирается в памяти, а памяти на
# бесплатном тарифе 512 МБ вместе с индексом. Отдаются первые строки в порядке
# суток, а не последние: что не уехало, уедет следующей выгрузкой, потому что
# сервис ничего не удаляет.
MAX_DRAIN_LINES = 50_000

state: dict[str, object] = {}


def _running_versions() -> dict[str, str]:
    """Версии того, что реально поднялось. Пустой словарь, если не поднялось."""
    current = state.get("service")
    if current is None:
        return {}
    running = {
        "service_version": __version__,
        "corpus_version": current.corpus_version,
        "config_version": load_catalog().get().version,
    }
    answer = state.get("answer")
    if answer is not None:
        running["answer_version"] = answer.settings.version
    return running


def _registry_version() -> str:
    """Версия выпуска для ответа. Расхождение называется прямо в строке.

    Пустая строка не годится: ответ без версии нельзя связать с числами
    README, и читающий лог не узнает, что сервис отвечал не тем, что
    зарегистрировано.
    """
    entry = state.get("registry")
    if entry is None:
        return "нет реестра"
    version = str(entry.get("version"))
    return version if not state.get("registry_mismatch") else f"{version}+расхождение"


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.monotonic()
    # Плотную часть можно выключить переменной окружения: на бесплатном
    # хостинге весов модели запроса может не быть, и тогда сервис обязан
    # подняться лексическим, а не упасть.
    with_dense = os.environ.get("RUFINRAG_DISABLE_DENSE", "") != "1"
    try:
        state["service"] = SearchService.build(with_dense=with_dense)
        state["error"] = None
    except (CorpusError, SearchError, ConfigError) as exc:
        # Сервис поднимается, но говорит о поломке в `/health`: иначе хостинг
        # перезапускает контейнер по кругу и причина не видна.
        state["service"] = None
        state["error"] = str(exc)
    # Слой ответа поднимается своей попыткой: опечатка в переменных
    # генеративного режима не должна гасить поиск, который от него не зависит.
    # Громкой она при этом остается: причина стоит в `/health` и в отказе
    # ручки `/answer`.
    state["answer"] = None
    state["answer_error"] = None
    if state["service"] is not None:
        try:
            state["answer"] = AnswerService.build(state["service"])
        except ConfigError as exc:
            state["answer_error"] = str(exc)
    # Реестр версий читается с диска, а не собирается: собирать его значит
    # считать отпечаток 448,8 МБ весов на каждом холодном старте, а на
    # бесплатном тарифе старт идет после каждого простоя (задача 6.3).
    # Роутер на инструментах поднимается только там, где его включили: его
    # библиотека стоит 66,3 МБ памяти на импорт при 215 МБ пика сервиса и
    # 512 МБ бесплатного тарифа (задача 8.2, pyproject, группа router).
    state["router"] = None
    state["router_off"] = "роутер выключен: RUFINRAG_ROUTER не равен 1"
    if os.environ.get("RUFINRAG_ROUTER", "") == "1" and state["answer"] is not None:
        try:
            from .router import Router

            state["router"] = Router.build(state["answer"])
            state["router_off"] = ""
        except ImportError as exc:
            state["router_off"] = f"нет langgraph, нужна группа router: {exc}"
    state["registry"] = registry.load(project_root())
    state["registry_mismatch"] = registry.compare(state["registry"], _running_versions())
    state["events"] = EventLog.from_env()
    state["started_at"] = time.time()
    state["startup_seconds"] = round(time.monotonic() - t0, 2)
    yield
    event_log = state.get("events")
    if event_log is not None:
        event_log.close()
    state.clear()


app = FastAPI(
    title="ru-fin-rag",
    version=__version__,
    description="Поиск по нормативным актам Банка России",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    # Лишние поля это ошибка, а не молчаливое игнорирование: иначе опечатка в
    # имени поля на клиенте выглядит как работающий запрос.
    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    top: int = Field(default=10, ge=1, le=MAX_TOP)
    config: str | None = None

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("запрос состоит из пробелов")
        return value


class AnswerRequest(BaseModel):
    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    top: int = Field(default=10, ge=1, le=MAX_TOP)
    config: str | None = None
    # Режим по умолчанию это выдержки: ключа модели у проекта нет и может не
    # появиться вовсе (PLAN, этап 7), поэтому генерация запрашивается явно.
    mode: Literal["excerpts", "generate"] = EXCERPTS

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("запрос состоит из пробелов")
        return value


class FeedbackRequest(BaseModel):
    model_config = {"extra": "forbid"}

    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    rating: Literal["up", "down"]


def record(event: dict) -> None:
    event_log = state.get("events")
    if event_log is not None:
        event_log.append(event)


# Грязный ввод до обработчика не доходит: его отвергает схема запроса. Без
# этого перехвата такой запрос не попал бы в журнал, и доля отказов считалась
# бы только по тем ошибкам, которые сервис заметил сам (QUALITY, раздел 3).
@app.exception_handler(RequestValidationError)
async def log_rejected(request: Request, exc: RequestValidationError):
    body = exc.body if isinstance(exc.body, dict) else {}
    if request.url.path in ("/search", "/answer"):
        query = body.get("query")
        top = body.get("top")
        record(
            events.rejected(
                request_id=events.new_request_id(),
                received_at=events.utc_now(),
                query=query if isinstance(query, str) else "",
                top=top if isinstance(top, int) else None,
                status=422,
                reason="; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
                ),
                service_version=__version__,
            )
        )
    return JSONResponse({"detail": jsonable_encoder(exc.errors())}, status_code=422)


# Корень отдает страницу, а не 404. Ссылка на сервис стоит в резюме, и человек,
# который по ней пришел, должен за десять секунд понять, что это и как
# попробовать, а не увидеть «Not Found» (так и случилось при первой проверке).
@app.get("/", response_class=HTMLResponse)
def index():
    current = state.get("service")
    catalog = load_catalog()
    if current is None:
        body = f"<p>Сервис не готов: {state.get('error')}</p>"
    else:
        chunks = f"{len(current.corpus):,}".replace(",", " ")
        body = (
            f"<p>Корпус: {chunks} фрагмента, версия "
            f"<code>{current.corpus_version}</code>. Связка по умолчанию: "
            f"<code>{catalog.default}</code>.</p>"
        )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ru-fin-rag</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 42rem;
       margin: 2rem auto; padding: 0 1rem; }}
code, pre {{ background: #f4f4f5; border-radius: 4px; }}
pre {{ padding: 0.75rem; overflow-x: auto; }}
code {{ padding: 0.1rem 0.3rem; }}
</style></head>
<body>
<h1>ru-fin-rag</h1>
<p>Поиск по нормативным актам Банка России: «Вестник Банка России» с 2013 года, фрагменты
со ссылкой на источник в каждом ответе.</p>
{body}
<h2>Попробовать</h2>
<p><a href="/docs">/docs</a> это интерактивная страница, где запрос можно отправить из
браузера. <a href="/health">/health</a> отдает версию корпуса и состояние ступеней.</p>
<pre>curl -s -X POST /search \\
  -H 'Content-Type: application/json' \\
  -d '{{"query": "порядок формирования резервов по ссудам", "top": 3}}'</pre>
<p>Ручка <code>/answer</code> отдает то же самое ответом с цитатами: выдержки из найденных
фрагментов со ссылкой на акт, раздел и пункты. Если опоры в корпусе под вопрос нет, она
отказывается отвечать и печатает число, по которому так решено.</p>
<pre>curl -s -X POST /answer \\
  -H 'Content-Type: application/json' \\
  -d '{{"query": "порядок формирования резервов по ссудам"}}'</pre>
<h2>Что важно знать</h2>
<p>Это бесплатный тариф: 512 МБ памяти, общий процессор, сон после 15 минут без трафика.
После сна первый запрос ждет пробуждения. Плотная ступень поиска в онлайне выключена, она
требует 1,1 ГБ памяти, и ответ помечает это полем <code>degraded</code>.</p>
<p>Код и отчет с числами: <a href="https://github.com/nata2627/ru-fin-rag">github.com/nata2627/ru-fin-rag</a></p>
</body></html>"""


@app.get("/health")
def health():
    current = state.get("service")
    catalog = load_catalog()
    body = {
        "status": "ok" if current else "broken",
        "version": __version__,
        "live": live_mode(),
        "startup_seconds": state.get("startup_seconds"),
        # Сколько живет этот процесс. По нему ночная выгрузка узнает, был ли
        # между двумя выгрузками перезапуск контейнера: если был, события,
        # записанные до него, пропали вместе с эфемерным диском, и это окно
        # потери считается числом (scripts/drain_events.py).
        "uptime_seconds": round(time.time() - float(state.get("started_at") or time.time()), 1),
        "configs": sorted(catalog.configs),
        "default_config": catalog.default,
        "registry": {
            "version": (state.get("registry") or {}).get("version", "нет реестра"),
            "built_at": (state.get("registry") or {}).get("built_at"),
            "matches_running": not state.get("registry_mismatch"),
            "mismatch": [
                {"field": m.field, "registered": m.registered, "running": m.running}
                for m in (state.get("registry_mismatch") or [])
            ],
        },
    }
    if current is None:
        body["error"] = state.get("error")
        return JSONResponse(body, status_code=503)
    body.update(
        {
            "corpus_version": current.corpus_version,
            "chunks": len(current.corpus),
            "bm25_index_mb": round(current.bm25.size_mb, 1),
            "dense": current.dense.backend if current.dense is not None else "off",
            "degraded": current.degraded,
        }
    )
    body["answer"] = _answer_health()
    # Состояние роутера видно рядом с состоянием ответа: выключенный роутер это
    # замысел, а не поломка, и причина выключения печатается строкой.
    body["router"] = (
        {"status": "on", "routes": list(ROUTES)}
        if state.get("router") is not None
        else {"status": "off", "off_reason": state.get("router_off")}
    )
    event_log = state.get("events")
    body["event_log"] = event_log.stats() if event_log is not None else "off"
    if current.dense_off:
        body["dense_off_reason"] = current.dense_off
    if current.degraded:
        body["status"] = "degraded"
    return body


def _answer_health() -> dict:
    """Состояние слоя ответа: порог отказа и доступность генерации.

    Порог стоит в `/health` затем, что по нему читается поведение сервиса:
    отказ это не ошибка, и человек, который видит отказ, должен иметь
    возможность увидеть число, по которому он случился.
    """
    answer = state.get("answer")
    if answer is None:
        return {"status": "off", "error": state.get("answer_error")}
    return {
        "status": "ok",
        "answer_version": answer.settings.version,
        "support_feature": answer.settings.feature,
        "support_threshold": answer.settings.threshold,
        "measured_in": answer.settings.measured_in,
        "modes": [EXCERPTS, GENERATE] if answer.generation_ready else [EXCERPTS],
        "generation": answer.generation_model if answer.generation_ready else "off",
        "generation_off_reason": answer.generation_off,
    }


@app.post("/search")
def search(request: SearchRequest):
    request_id = events.new_request_id()
    received_at = events.utc_now()
    current = state.get("service")
    if current is None:
        detail = f"сервис не готов: {state.get('error')}"
        record(
            events.rejected(
                request_id, received_at, request.query, request.top, 503, detail, __version__
            )
        )
        raise HTTPException(status_code=503, detail=detail)
    try:
        result = current.search(request.query, top=request.top, config=request.config)
    except (ConfigError, SearchError) as exc:
        record(
            events.rejected(
                request_id, received_at, request.query, request.top, 400, str(exc), __version__
            )
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    record(events.answered(request_id, received_at, request.top, result, __version__))
    return {
        "request_id": request_id,
        "query": result.query,
        "config": result.config,
        "config_version": result.config_version,
        "corpus_version": result.corpus_version,
        "registry_version": _registry_version(),
        "timings_ms": result.timings_ms,
        "degraded": result.degraded,
        "results": [
            {
                "rank": hit.rank,
                "chunk_id": hit.chunk_id,
                "score": round(hit.score, 6),
                "act": hit.act,
                "act_id": hit.act_id,
                "title": hit.title,
                "section": hit.section,
                "url": hit.url,
                "text": hit.text,
            }
            for hit in result.hits
        ],
    }


@app.post("/answer")
def answer(request: AnswerRequest):
    """Ответ с цитатами или явный отказ (SPEC, раздел 5).

    Отказ отдается кодом 200 и полем `answered`, а не ошибкой: запрос
    обслужен, сервис не сломался, и ответ на него это «опоры в корпусе нет»
    вместе с числом, по которому так решено. Ошибкой отвечают только
    неготовый сервис, плохой запрос и недоступный режим.
    """
    request_id = events.new_request_id()
    received_at = events.utc_now()
    service = state.get("answer")
    if service is None:
        detail = state.get("answer_error") or f"сервис не готов: {state.get('error')}"
        record(
            events.rejected(
                request_id, received_at, request.query, request.top, 503, str(detail), __version__
            )
        )
        raise HTTPException(status_code=503, detail=detail)
    try:
        result = service.answer(
            request.query, top=request.top, config=request.config, mode=request.mode
        )
    except AnswerError as exc:
        # Режим, которого нет на этом развертывании, это 503: сам запрос
        # правильный, и повторять его с ключом имеет смысл.
        status = 503 if request.mode == GENERATE and not service.generation_ready else 400
        record(
            events.rejected(
                request_id, received_at, request.query, request.top, status, str(exc), __version__
            )
        )
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except (ConfigError, SearchError) as exc:
        record(
            events.rejected(
                request_id, received_at, request.query, request.top, 400, str(exc), __version__
            )
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record(events.answer_event(request_id, received_at, request.top, result, __version__))
    body = {
        "request_id": request_id,
        "query": result.query,
        "mode": result.mode,
        "answered": result.answered,
        "answer": result.text,
        "answer_source": result.text_source,
        "answer_version": result.answer_version,
        "support": result.support.as_dict(),
        "citations": [citation.as_dict() for citation in result.citations],
        "config": result.search.config,
        "config_version": result.search.config_version,
        "corpus_version": result.search.corpus_version,
        "registry_version": _registry_version(),
        "timings_ms": result.timings_ms,
        "degraded": result.search.degraded,
    }
    if not result.answered:
        body["refusal"] = result.refusal
    return body


class RouteRequest(BaseModel):
    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    top: int = Field(default=10, ge=1, le=MAX_TOP)

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("пустой запрос")
        return value


@app.post("/route")
def route(request: RouteRequest) -> dict:
    """Ответ через роутер на инструментах: маршрут называется в ответе.

    Ручка отдельная, а не флаг у `/answer`: маршрут меняет и выдачу, и состав
    ответа, и человеку, который сравнивает роутер с прямым поиском (задача
    8.3), нужны оба поведения рядом, а не одно вместо другого.
    """
    router_service = state.get("router")
    if router_service is None:
        raise HTTPException(status_code=503, detail=state.get("router_off") or "роутер не поднят")
    try:
        run = router_service.run(request.query, top=request.top)
    except (ConfigError, SearchError, AnswerError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    answer = run.answer
    body: dict = {
        "query": run.query,
        "route": run.route,
        "reason": run.reason,
        "trace": list(run.trace),
        "timings_ms": run.timings_ms,
        "registry_version": _registry_version(),
    }
    if run.clarify is not None:
        body["clarify"] = run.clarify
    if answer is not None:
        body.update(
            {
                "answered": answer.answered,
                "answer": answer.text,
                "answer_version": answer.answer_version,
                "support": answer.support.as_dict(),
                "citations": [citation.as_dict() for citation in answer.citations],
                "config": answer.search.config,
                "config_version": answer.search.config_version,
                "corpus_version": answer.search.corpus_version,
            }
        )
        if not answer.answered:
            body["refusal"] = answer.refusal
    return body


# Оценка приходит отдельным событием: она появляется после ответа, и поле
# «оценка» в строке запроса всегда стояло бы пустым (events.py). Существование
# идентификатора здесь не проверяется, процесс держит в памяти только текущий
# файл: оценку без своего запроса витрина отбросит на соединении и посчитает
# числом.
@app.post("/feedback", status_code=202)
def feedback(request: FeedbackRequest):
    if state.get("events") is None:
        raise HTTPException(status_code=503, detail="журнал запросов выключен")
    record(events.feedback(request.request_id, request.rating))
    return {"accepted": True}


# Выдача журнала ночной выгрузке. Диск бесплатного тарифа эфемерный, и события
# живут только до перезапуска контейнера, а сам сервис в репозиторий коммитить
# не может: коммит делает прогон (DECISIONS, решение 12). Значит нужен путь, по
# которому прогон забирает строки у живого процесса.
#
# Это ручка над текстами чужих запросов, поэтому она закрыта секретом из
# окружения и выключена, пока секрета нет: без переменной маршрут отвечает 404,
# то есть снаружи его не существует. На странице `/docs` он не показывается не
# ради скрытности (в README он описан), а потому что человеку с улицы делать
# тут нечего.
#
# Строки отдаются как есть и не удаляются: повторная выгрузка безопасна, ее
# разбирает прогон по `request_id`, а удаление теряло бы события при неудачном
# коммите.
@app.get("/events", include_in_schema=False, response_class=PlainTextResponse)
def events_dump(token: str = Header(default="", alias="X-Events-Token")):
    expected = env("RUFINRAG_EVENTS_TOKEN")
    if not expected:
        raise HTTPException(status_code=404, detail="Not Found")
    if not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="неверный токен выгрузки")
    event_log = state.get("events")
    if event_log is None:
        raise HTTPException(status_code=503, detail="журнал запросов выключен")

    files = sorted(event_log.directory.glob("queries-*.jsonl"))
    lines: list[str] = []
    truncated = False
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if len(lines) >= MAX_DRAIN_LINES:
                truncated = True
                break
            lines.append(line)
        if truncated:
            break
    return PlainTextResponse(
        "".join(line + "\n" for line in lines),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "X-Events-Lines": str(len(lines)),
            "X-Events-Files": str(len(files)),
            "X-Events-Truncated": "1" if truncated else "0",
        },
    )
