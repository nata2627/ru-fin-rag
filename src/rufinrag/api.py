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
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import ConfigError, live_mode, load_catalog
from .corpus import CorpusError
from .search import SearchError, SearchService

# Предел длины запроса взят по эталонному набору: самый длинный живой вопрос
# практика в нем 3067 символов, а предел 1000 отрезал 22 запроса из 1182. Это
# нашел нагрузочный замер: сервис отвечал 422 на настоящие вопросы.
MAX_QUERY_CHARS = 4000
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
        body = (
            f"<p>Корпус: {len(current.corpus)} фрагментов, версия "
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
<pre>curl -s -X POST {{"query": "порядок формирования резервов по ссудам", "top": 3}} \
  -H 'Content-Type: application/json' /search</pre>
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
    if current.dense_off:
        body["dense_off_reason"] = current.dense_off
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
