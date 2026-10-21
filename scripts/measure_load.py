"""Нагрузочный замер: поднять сервис, прогнать locust, собрать числа.

Один скрипт вместо инструкции из трех команд: иначе замер повторяется руками и
расходится по параметрам между прогонами.

Узкое место ищется сравнением двух связок. Лексическая упирается в питон и в
обработку запроса, гибридная в кодирование запроса моделью, и разница между
ними и есть ответ, что оптимизировать.

Запуск: python scripts/measure_load.py [--users 10] [--seconds 60] [--config bm25]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8871


def wait_for_health(url: str, deadline: float, process: subprocess.Popen) -> dict:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"сервис упал на старте, код {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1)
    raise RuntimeError(f"сервис не ответил на {url}")


def read_locust_stats(prefix: Path) -> list[dict]:
    path = prefix.with_name(prefix.name + "_stats.csv")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--spawn-rate", type=float, default=2)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--config", default="bm25")
    parser.add_argument("--out", type=Path, default=ROOT / "results/load.json")
    args = parser.parse_args()

    env = dict(os.environ)
    # Нагружается связка по умолчанию из файла конфигурации, поэтому связка
    # замера задается переменной, а не правкой файла на время прогона.
    env["RUFINRAG_DEFAULT_CONFIG"] = args.config
    if args.config in ("bm25", "bm25-rerank"):
        env["RUFINRAG_DISABLE_DENSE"] = "1"

    service = subprocess.Popen(
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
        health = wait_for_health(f"http://127.0.0.1:{PORT}/health", time.monotonic() + 120, service)
        prefix = ROOT / "results/load"
        locust = subprocess.run(
            [
                str(ROOT / ".venv/bin/locust"),
                "-f",
                str(ROOT / "load/locustfile.py"),
                "--headless",
                "-u",
                str(args.users),
                "-r",
                str(args.spawn_rate),
                "-t",
                f"{args.seconds}s",
                "--host",
                f"http://127.0.0.1:{PORT}",
                "--csv",
                str(prefix),
                "--only-summary",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if locust.returncode != 0:
            print(locust.stdout[-2000:], file=sys.stderr)
            print(locust.stderr[-2000:], file=sys.stderr)
            return 1

        rows = {row["Name"]: row for row in read_locust_stats(prefix)}
        report = {
            "measured_at": time.strftime("%Y-%m-%d"),
            "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
            "config": args.config,
            "config_in_health": health.get("default_config"),
            "corpus_version": health.get("corpus_version"),
            "dense": health.get("dense"),
            "users": args.users,
            "seconds": args.seconds,
            "endpoints": {
                name: {
                    "requests": int(row["Request Count"]),
                    "failures": int(row["Failure Count"]),
                    "rps": float(row["Requests/s"]),
                    "p50_ms": float(row["50%"]),
                    "p95_ms": float(row["95%"]),
                    "p99_ms": float(row["99%"]),
                    "max_ms": float(row["Max Response Time"]),
                }
                for name, row in rows.items()
                if name != "Aggregated"
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for name, row in report["endpoints"].items():
            print(name, row, flush=True)
        print(f"записано в {args.out}", flush=True)
        return 0
    finally:
        service.terminate()
        service.wait(timeout=30)


if __name__ == "__main__":
    raise SystemExit(main())
