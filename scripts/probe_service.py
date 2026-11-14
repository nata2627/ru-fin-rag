#!/usr/bin/env python3
"""Проверка живой ссылки из регламента: ответил ли сервис и за сколько.

14.11.2026 ссылка молчала на пяти запросах подряд, а регламент шел зеленым:
деплой на хостинге числился живым, и ни один шаг в сервис не ходил. Узнать об
этом можно было только из браузера. Этот шаг ходит в `/health` в каждом
плановом прогоне и ведет счет в `results/uptime.json`: сколько проверок было
и на скольких ссылка ответила.

Сервис на бесплатном тарифе спит, поэтому первый ответ ждет пробуждения, и
время до ответа пишется рядом: это то, что видит человек, открывший ссылку.
Молчание прогон не роняет, оно уходит в issue тем же путем, что отказ cbr.ru.
Счет идет с первой проверки, прошлых суток в нем нет.

Запуск: python scripts/probe_service.py [--url URL]
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "uptime.json"
DEFAULT_URL = "https://ru-fin-rag.onrender.com"
# Пробуждение занимало 43-53 с в трех замерах. Одна попытка длиннее этого с
# запасом, вторая нужна на случай, когда первую оборвала сеть до пробуждения.
TIMEOUT = 120
ATTEMPTS = 2


def ask(url: str, timeout: int) -> dict:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as response:
            body = json.loads(response.read())
        status = body.get("status") if isinstance(body, dict) else None
        ok, note = status == "ok", f"status {status}"
    # Сеть, TLS, обрыв посреди ответа и не JSON в ответе: все это молчание
    # ссылки, а упавший шаг покрасил бы плановый прогон.
    except (OSError, http.client.HTTPException, ValueError) as error:
        ok, note = False, f"{type(error).__name__}: {error}"
    return {"ok": ok, "seconds": round(time.monotonic() - started, 1), "note": note}


def probe(url: str, timeout: int, attempts: int) -> dict:
    tries = []
    for _ in range(attempts):
        tries.append(ask(url, timeout))
        if tries[-1]["ok"]:
            break
    return {
        "at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": url,
        "ok": tries[-1]["ok"],
        "attempts": len(tries),
        "seconds_to_answer": round(sum(t["seconds"] for t in tries), 1),
        "note": tries[-1]["note"],
    }


def record(state: dict, last: dict) -> dict:
    """Новое состояние счета: проверка прибавляется всегда, ответ только при ответе."""
    return {
        "since": state.get("since") or last["at"],
        "checks_total": state.get("checks_total", 0) + 1,
        "checks_ok": state.get("checks_ok", 0) + int(last["ok"]),
        "silent_in_a_row": 0 if last["ok"] else state.get("silent_in_a_row", 0) + 1,
        "last": last,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("RUFINRAG_SERVICE_URL", DEFAULT_URL))
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    parser.add_argument("--attempts", type=int, default=ATTEMPTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    state = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    state = record(state, probe(args.url.rstrip("/"), args.timeout, args.attempts))
    args.out.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"status={'ok' if state['last']['ok'] else 'silent'}\n")
    last = state["last"]
    print(
        f"ссылка {'ответила' if last['ok'] else 'молчит'} за {last['seconds_to_answer']} с, "
        f"попыток {last['attempts']}; "
        f"ответов {state['checks_ok']} из {state['checks_total']} проверок"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
