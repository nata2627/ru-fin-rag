"""Замер сервиса на хостинге: пробуждение, задержка, что именно отвечает.

Числа с бесплатного тарифа несравнимы с числами мака и подписываются отдельно:
там общий процессор и сон после 15 минут простоя. Поэтому первым меряется
именно пробуждение (холодный запрос после сна), а уже потом задержка на
прогретом сервисе.

Запуск: python scripts/measure_hosted.py --url https://ru-fin-rag.onrender.com
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_queries  # noqa: E402

TIMEOUT = 180


def get(url: str) -> tuple[int, dict | None, float]:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status, body, (time.monotonic() - started) * 1000
    except urllib.error.HTTPError as error:
        return error.code, None, (time.monotonic() - started) * 1000


def search(url: str, query: str, top: int = 10) -> tuple[int, dict | None, float]:
    payload = json.dumps({"query": query, "top": top}).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/search", data=payload, headers={"Content-Type": "application/json"}
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status, body, (time.monotonic() - started) * 1000
    except urllib.error.HTTPError as error:
        return error.code, None, (time.monotonic() - started) * 1000


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "results/latency-hosted.json")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    status, health, wake_ms = get(f"{url}/health")
    if health is None:
        print(f"{url}/health ответил {status}", file=sys.stderr)
        return 1

    queries = [q.text for q in load_queries().values()][: args.queries]
    times: list[float] = []
    degraded: list[str] = []
    configs: set[str] = set()
    for text in queries:
        code, body, ms = search(url, text)
        if body is None:
            print(f"/search ответил {code}", file=sys.stderr)
            return 1
        times.append(ms)
        configs.add(body["config"])
        degraded.extend(body.get("degraded", []))

    report = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": url,
        "machine": "бесплатный тариф Render, общий процессор, числа несравнимы с маком",
        "first_request_ms": round(wake_ms, 1),
        "health": health,
        "queries": len(times),
        "config": sorted(configs),
        "p50_ms": round(statistics.median(times), 1),
        "p95_ms": round(percentile(times, 0.95), 1),
        "max_ms": round(max(times), 1),
        "degraded": sorted(set(degraded)),
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
