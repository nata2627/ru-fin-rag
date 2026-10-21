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

import pytest

from rufinrag.corpus import corpus_path

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
def test_service_answers_over_http():
    if not corpus_path().exists():
        pytest.skip("нет корпуса, собрать: python scripts/import_corpus.py")

    env = dict(os.environ, RUFINRAG_DISABLE_DENSE="1")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "rufinrag.api:app",
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
        assert health["chunks"] == 62594

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
    finally:
        process.terminate()
        process.wait(timeout=30)
