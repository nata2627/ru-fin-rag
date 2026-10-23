"""Сервис поднимается отдельным процессом и отвечает на запрос.

Клиент в тестах ходит в приложение напрямую, минуя сервер. Этот тест
проверяет то, чего там нет: что `uvicorn` поднимается той же командой, что в
Makefile и в README, и что сервис отвечает по сети.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from rufinrag.corpus import corpus_path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765
TIMEOUT_SECONDS = 90


def wait_for_health(url: str, deadline: float, process: subprocess.Popen) -> dict:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"сервис упал на старте, код {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1)
    raise AssertionError(f"сервис не ответил на {url} за {TIMEOUT_SECONDS} с")


@pytest.mark.slow
def test_service_answers_over_http(event_dir_outside_the_repo):
    if not corpus_path().exists():
        pytest.skip("нет корпуса, собрать: python scripts/import_corpus.py")

    env = dict(os.environ, RUFINRAG_DISABLE_DENSE="1")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "rufinrag.api:app",
            # Та же команда, что в Makefile и в README: без `--app-dir` uvicorn
            # берет пакет из site-packages, и тест проверял бы установленную
            # копию вместо рабочего дерева. Один раз уже проверял.
            "--app-dir",
            str(ROOT / "src"),
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        health = wait_for_health(
            f"http://127.0.0.1:{PORT}/health",
            time.monotonic() + TIMEOUT_SECONDS,
            process,
        )
        assert health["status"] in ("ok", "degraded")
        assert health["chunks"] > 60000

        payload = json.dumps({"query": "норматив достаточности базового капитала"}).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/search",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read())
        assert len(body["results"]) == 10
        assert body["corpus_version"] == health["corpus_version"]
        assert body["results"][0]["url"].startswith("https://www.cbr.ru/")

        # Подпроцесс наследует RUFINRAG_EVENT_DIR из tests/conftest.py, поэтому
        # событие прогона лежит в каталоге pytest, а не в data/events
        # репозитория, куда едут события сервиса (DECISIONS, решение 12).
        logged = "".join(
            path.read_text(encoding="utf-8")
            for path in event_dir_outside_the_repo.glob("queries-*.jsonl")
        )
        assert body["request_id"] in logged
        in_repo = "".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "data/events").glob("queries-*.jsonl")
        )
        assert body["request_id"] not in in_repo
    finally:
        process.terminate()
        process.wait(timeout=30)
