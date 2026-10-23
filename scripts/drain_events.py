"""Выгрузка журнала запросов с хостинга в репозиторий.

Зачем так. Диск бесплатного тарифа эфемерный, и процесс засыпает после 15
минут без трафика, то есть события, записанные сервисом, живут только до
перезапуска контейнера. Сам сервис в репозиторий коммитить не может и не
должен: это потребовало бы токена со сроком, а срок токена это самая вероятная
смерть автономности (DECISIONS, решение 12). Поэтому события забирает ночной
прогон у живого процесса и коммитит встроенным `GITHUB_TOKEN`, у которого
срока нет.

Что при этом теряется. Выгрузка видит только то, что лежит на диске в момент
запроса. События контейнера, который заснул и перезапустился между двумя
выгрузками, пропали вместе с его диском, и посчитать их нельзя: счетчик умер
там же. Что считается: число выгрузок, у которых процесс сервиса оказался
моложе промежутка с прошлой выгрузки, то есть окно потери было. Числитель и
знаменатель лежат в `results/events.json`.

Идемпотентность. Повторная выгрузка предлагает те же строки (сервис ничего не
удаляет), и дубли опознаются по ключу события, а не по позиции в файле. Ключ
это `kind`, `request_id` и `received_at`: одного `request_id` мало, потому что
запрос и оценка на него это два события с одним идентификатором.

Шаг не роняет прогон. Спящий сервис, незаданный токен, пустой журнал и
недоступная сеть это нормальные состояния, а не поломка: красный main
останавливает часы приемки автономности. Состояние последней выгрузки видно в
`results/events.json`, там же число неудач подряд.

Запуск: python scripts/drain_events.py [--url https://ru-fin-rag.onrender.com]
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_URL = "https://ru-fin-rag.onrender.com"
# Бесплатный тариф просыпается около минуты, и первый запрос ждет пробуждения.
TIMEOUT = 180
ATTEMPTS = 2


class DrainError(Exception):
    """Ожидаемая неудача выгрузки: сеть, токен, выключенный журнал."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def event_key(event: dict) -> tuple[str, str, str]:
    return event["kind"], event["request_id"], event["received_at"]


def get(url: str, token: str = "", timeout: int = TIMEOUT) -> tuple[bytes, dict[str, str]]:
    """Тело и заголовки ответа. Имена заголовков приводятся к нижнему регистру:
    сервер ASGI отдает их так, а не так, как они написаны в коде сервиса."""
    headers = {"X-Events-Token": token} if token else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), {name.lower(): value for name, value in response.headers.items()}


def fetch(url: str, token: str, timeout: int, attempts: int) -> tuple[dict, list[str], dict]:
    """Состояние сервиса и строки журнала. Повтор только на отказе транспорта."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            raw, _ = get(f"{url}/health", timeout=timeout)
            health = json.loads(raw.decode("utf-8"))
            body, headers = get(f"{url}/events", token=token, timeout=timeout)
            break
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise DrainError(
                    "drain off",
                    f"{url}/events отвечает 404: на сервисе не задан RUFINRAG_EVENTS_TOKEN",
                ) from error
            if error.code == 403:
                raise DrainError(
                    "forbidden", f"{url}/events отвечает 403: токен выгрузки не совпал"
                ) from error
            if error.code == 503:
                raise DrainError("logging off", f"{url}: журнал запросов выключен") from error
            raise DrainError("http error", f"{url} ответил {error.code}") from error
        except (OSError, json.JSONDecodeError) as error:
            last = error
            if attempt + 1 < attempts:
                time.sleep(10)
    else:
        raise DrainError("unreachable", f"{url} не ответил за {attempts} попытки: {last}")

    lines = [line for line in body.decode("utf-8").splitlines() if line.strip()]
    meta = {
        "offered_lines": len(lines),
        "files": int(headers.get("x-events-files", 0)),
        "truncated": headers.get("x-events-truncated") == "1",
    }
    if meta["offered_lines"] != int(headers.get("x-events-lines", -1)):
        raise DrainError(
            "short read",
            f"сервис объявил {headers.get('x-events-lines')} строк, пришло {len(lines)}",
        )
    return health, lines, meta


def merge(lines: list[str], directory: Path) -> dict:
    """Дописать новые события в файлы суток репозитория.

    Сутки берутся из `received_at` самого события, а не из имени файла, который
    их отдал: нарезка по суткам UTC живет в одном месте (`events.py`).

    Строки дописываются в конец, а не переписываются отсортированными: сервис
    отдает их в порядке записи, а новые события всегда позже уже лежащих,
    поэтому порядок во файле остается временным, а дифф остается дописыванием.
    """
    directory.mkdir(parents=True, exist_ok=True)
    by_day: dict[str, list[tuple[tuple[str, str, str], str]]] = {}
    for line in lines:
        event = json.loads(line)
        by_day.setdefault(event["received_at"][:10], []).append((event_key(event), line))

    added = 0
    duplicates = 0
    touched: list[str] = []
    for day, items in sorted(by_day.items()):
        path = directory / f"queries-{day}.jsonl"
        known = set()
        if path.exists():
            known = {
                event_key(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        fresh = []
        for key, line in items:
            if key in known:
                duplicates += 1
                continue
            known.add(key)
            fresh.append(line)
        if not fresh:
            continue
        with path.open("a", encoding="utf-8") as handle:
            handle.write("".join(line + "\n" for line in fresh))
        added += len(fresh)
        touched.append(path.name)
    return {"added": added, "duplicates": duplicates, "files_touched": touched}


def survey(directory: Path) -> dict:
    """Что лежит в репозитории после выгрузки."""
    files = sorted(directory.glob("queries-*.jsonl")) if directory.exists() else []
    lines = 0
    kinds: dict[str, int] = {}
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            lines += 1
            kind = json.loads(line)["kind"]
            kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "files": [path.name for path in files],
        "days": len(files),
        "lines": lines,
        "query_events": kinds.get("query", 0),
        "feedback_events": kinds.get("feedback", 0),
    }


def seconds_since(moment: str | None) -> float | None:
    """Сколько прошло с момента в UTC. `timegm`, а не `mktime`: второй читает
    разобранное время как местное, и промежуток уезжал бы на часовой пояс
    машины, где идет прогон."""
    if not moment:
        return None
    return time.time() - calendar.timegm(time.strptime(moment, "%Y-%m-%dT%H:%M:%SZ"))


def restart_window(health: dict, previous_at: str | None) -> bool | None:
    """Был ли между этой и прошлой выгрузкой перезапуск контейнера.

    Возвращает `None`, когда ответить нечем: первая выгрузка или сервис старой
    версии, который не отдает `uptime_seconds`.
    """
    uptime = health.get("uptime_seconds")
    gap = seconds_since(previous_at)
    if uptime is None or gap is None:
        return None
    return float(uptime) < gap


def drain(url: str, token: str, directory: Path, state: dict, timeout: int, attempts: int) -> dict:
    """Одна выгрузка. Состояние прошлых выгрузок приходит и уходит словарем."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    previous = state.get("last_drain") or {}
    last_drain: dict = {"at": now, "url": url}
    try:
        health, lines, meta = fetch(url, token, timeout, attempts)
    except DrainError as error:
        last_drain.update({"status": error.status, "error": str(error)})
        state["failed_drains_in_a_row"] = int(state.get("failed_drains_in_a_row", 0)) + 1
    else:
        window = restart_window(health, previous.get("at"))
        event_log = health.get("event_log")
        last_drain.update(
            {
                "status": "ok",
                "service_version": health.get("version"),
                "service_uptime_seconds": health.get("uptime_seconds"),
                # Сколько событий записал именно этот процесс. Равенство с
                # числом предложенных строк означает, что на диске лежит все,
                # что процесс написал, и ничего лишнего.
                "written_by_process": event_log.get("written")
                if isinstance(event_log, dict)
                else None,
                "restart_window": window,
                **meta,
                **merge(lines, directory),
            }
        )
        state["failed_drains_in_a_row"] = 0
        state["drains_with_restart_window"] = int(
            state.get("drains_with_restart_window", 0)
        ) + bool(window)
    state["drains"] = int(state.get("drains", 0)) + 1
    state["last_drain"] = last_drain
    state["repo"] = survey(directory)
    state["note"] = (
        "выгрузка журнала запросов с хостинга в репозиторий, "
        "числа отсюда в публичные тексты не берутся: файл меняется каждым прогоном"
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("RUFINRAG_SERVICE_URL") or DEFAULT_URL)
    parser.add_argument("--events-dir", type=Path, default=ROOT / "data/events")
    parser.add_argument("--out", type=Path, default=ROOT / "results/events.json")
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    parser.add_argument("--attempts", type=int, default=ATTEMPTS)
    args = parser.parse_args()

    token = os.environ.get("RUFINRAG_EVENTS_TOKEN", "").strip()
    state = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    # Каталог и отчет создаются при любом исходе: шаг коммита в прогоне
    # перечисляет их путями, а `git add` на отсутствующем пути падает.
    args.events_dir.mkdir(parents=True, exist_ok=True)

    if not token:
        # Без секрета выгрузка не настроена, и это не поломка: сервис работает,
        # журнал пишется, а строки остаются на хостинге до перезапуска.
        print("RUFINRAG_EVENTS_TOKEN не задан, выгрузка пропущена", file=sys.stderr)
        state.setdefault("drains", 0)
        state["last_drain"] = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "url": args.url.rstrip("/"),
            "status": "not configured",
        }
        state["repo"] = survey(args.events_dir)
        args.out.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    state = drain(
        args.url.rstrip("/"), token, args.events_dir, state, args.timeout, args.attempts
    )
    args.out.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False))
    if state["last_drain"]["status"] != "ok":
        print(
            f"выгрузка не прошла: {state['last_drain']['error']}. "
            f"Неудач подряд: {state['failed_drains_in_a_row']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
