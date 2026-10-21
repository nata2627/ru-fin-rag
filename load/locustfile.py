"""Нагрузочный профиль сервиса поиска.

Запросы берутся из эталонного набора, а не генерируются: длина запроса влияет
на задержку BM25 напрямую (число токенов это число проходов по матрице), и на
выдуманных запросах профиль был бы не тот.

Запуск:
    .venv/bin/locust -f load/locustfile.py --headless -u 10 -r 2 -t 60s \
        --host http://127.0.0.1:8000 --csv results/load
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
QUERIES = [
    json.loads(line)["text"]
    for line in (ROOT / "data/benchmark/queries.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


class SearchUser(HttpUser):
    # Пауза между запросами, иначе замер показывает поведение генератора, а не
    # сервиса: один пользователь без паузы это один непрерывный поток.
    wait_time = between(0.1, 0.5)

    @task(10)
    def search(self):
        self.client.post(
            "/search",
            json={"query": random.choice(QUERIES), "top": 10},
            name="/search",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
