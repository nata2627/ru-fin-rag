"""Репетиция выгрузки журнала: сервис процессом, выгрузка по сети, файлы в репозитории.

Зачем репетиция, а не замер на хостинге. Ручка `/events` появляется на
хостинге только после пуша, а пуш уходит наружу и ждет «да» автора. Поэтому
путь проверяется целиком, но на локальном процессе: тот же `uvicorn`, тот же
HTTP, тот же `scripts/drain_events.py`. Числа здесь это числа репетиции, что в
отчете и написано.

Что проверяется тремя выгрузками: первая кладет события в репозиторий, вторая
на том же журнале не добавляет ничего и опознает дубли по `request_id`, третья
после новых запросов добавляет только новые. Это и есть рост без дублей.

Запуск: python scripts/rehearse_drain.py [--queries 20] [--feedback 5] [--extra 5]
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import drain_events  # noqa: E402

from rufinrag.benchmark import load_queries  # noqa: E402

STARTUP_TIMEOUT = 120


def wait_for_health(url: str, process: subprocess.Popen) -> dict:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"сервис упал на старте, код {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
                return json.loads(response.read())
        except (OSError, json.JSONDecodeError):
            time.sleep(1)
    raise SystemExit(f"сервис не ответил на {url}/health за {STARTUP_TIMEOUT} с")


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def traffic(url: str, texts: list[str], rate: int = 0) -> tuple[int, int]:
    """Запросы и оценки через HTTP. Возвращает число запросов и число оценок."""
    rated = 0
    for text in texts:
        body = post(f"{url}/search", {"query": text, "top": 3})
        if rated < rate:
            post(f"{url}/feedback", {"request_id": body["request_id"], "rating": "up"})
            rated += 1
    return len(texts), rated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--feedback", type=int, default=5)
    parser.add_argument("--extra", type=int, default=5)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--out", type=Path, default=ROOT / "results/events-rehearsal.json")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="rufinrag-drain-"))
    service_dir = work / "service-disk"
    repo_dir = work / "repo-events"
    token = secrets.token_hex(16)
    url = f"http://127.0.0.1:{args.port}"

    texts = [question.text for question in load_queries().values()]
    if len(texts) < args.queries + args.extra:
        raise SystemExit("в эталонном наборе меньше вопросов, чем просит репетиция")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "rufinrag.api:app",
            "--app-dir",
            str(ROOT / "src"),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--log-level",
            "warning",
        ],
        env=dict(
            os.environ,
            RUFINRAG_DISABLE_DENSE="1",
            RUFINRAG_EVENT_DIR=str(service_dir),
            RUFINRAG_EVENTS_TOKEN=token,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        health = wait_for_health(url, process)
        sent, rated = traffic(url, texts[: args.queries], rate=args.feedback)

        state: dict = {}
        rounds = []
        for extra in (0, 0, args.extra):
            if extra:
                sent_more, _ = traffic(url, texts[args.queries : args.queries + extra])
                sent += sent_more
            state = drain_events.drain(url, token, repo_dir, state, timeout=60, attempts=1)
            last = state["last_drain"]
            if last["status"] != "ok":
                raise SystemExit(f"выгрузка не прошла: {last.get('error')}")
            rounds.append(
                {
                    "new_events_before": extra,
                    "offered_lines": last["offered_lines"],
                    "added": last["added"],
                    "duplicates": last["duplicates"],
                    "lines_in_repo": state["repo"]["lines"],
                }
            )

        final = drain_events.survey(repo_dir)
        report = {
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": (
                "репетиция на локальном процессе: ручка выдачи появится на хостинге "
                "после пуша, числа здесь это числа репетиции"
            ),
            "service": {
                "version": health["version"],
                "queries_sent": sent,
                "feedback_sent": rated,
                "written_by_process": state["last_drain"]["written_by_process"],
            },
            "drains": rounds,
            "repo": final,
            "grew": rounds[0]["added"] > 0 and rounds[2]["added"] == args.extra,
            "no_duplicates": rounds[1]["added"] == 0
            and rounds[1]["duplicates"] == rounds[0]["added"]
            and final["lines"] == rounds[2]["lines_in_repo"],
        }
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["grew"] and report["no_duplicates"] else 1
    finally:
        process.terminate()
        process.wait(timeout=30)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
