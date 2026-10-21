"""HTTP-интерфейс сервиса.

В каждом ответе стоят версия корпуса и версия связки поиска: без них сравнение
конфигураций и регрессия качества не читаются, а на хостинге не видно, что
именно развернуто (SPEC, раздел 3).

Индексы читаются один раз при старте процесса, а не на запрос: корпус это
62 594 фрагмента, и его чтение занимает секунду.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import ConfigError, live_mode, load_catalog
from .corpus import CorpusError
from .search import SearchError, SearchService

MAX_QUERY_CHARS = 1000
MAX_TOP = 50

state: dict[str, object] = {}


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
    state["started_at"] = time.time()
    state["startup_seconds"] = round(time.monotonic() - t0, 2)
    yield
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


def service() -> SearchService:
    current = state.get("service")
    if current is None:
        raise HTTPException(status_code=503, detail=f"сервис не готов: {state.get('error')}")
    return current


@app.get("/health")
def health():
    current = state.get("service")
    catalog = load_catalog()
    body = {
        "status": "ok" if current else "broken",
        "version": __version__,
        "live": live_mode(),
        "startup_seconds": state.get("startup_seconds"),
        "configs": sorted(catalog.configs),
        "default_config": catalog.default,
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
    if current.degraded:
        body["status"] = "degraded"
    return body


@app.post("/search")
def search(request: SearchRequest):
    current = service()
    try:
        result = current.search(request.query, top=request.top, config=request.config)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SearchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "query": result.query,
        "config": result.config,
        "config_version": result.config_version,
        "corpus_version": result.corpus_version,
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
