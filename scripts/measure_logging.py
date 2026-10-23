"""Чего стоит журнал запросов: скорость записи, цена для запроса, место на диске.

Четыре вопроса одним прогоном, потому что числа читаются только вместе: запись
сама по себе быстрая, но смысл имеет лишь ее доля в задержке запроса и объем
файлов за месяц.

События для замера скорости не выдумываются: прогон сначала отправляет сервису
настоящие запросы с включенным журналом, а потом гоняет писателя по тем самым
строкам, которые сервис записал. Поэтому и размер строки в отчете это размер
настоящего события, а не оценка по схеме.

Каталог журнала у прогона свой, временный: события замеров в рабочее дерево не
едут, туда едут события сервиса (DECISIONS, решение 12).

Запуск: python scripts/measure_logging.py [--repeats 8] [--writes 20000]
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import http.client
import itertools
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag import events  # noqa: E402
from rufinrag.benchmark import load_queries  # noqa: E402
from rufinrag.retrieval.text import tokenize  # noqa: E402

PORT = 8873
WARMUP = 3
# Порядок блоков встречный: если бы журнал мерился первым, а выключенный
# журнал последним, разницу дал бы прогрев машины, а не запись. Пар берется
# несколько, потому что между блоками машина плывет сильнее, чем стоит запись.
BLOCK_PAIR = ("on", "off", "off", "on")
# Уровни трафика для экстраполяции. Живого трафика у сервиса нет, это
# предположения, и в отчете они названы предположениями.
TRAFFIC_PER_DAY = (100, 1_000, 10_000)
BASE_TRAFFIC_PER_DAY = 1_000
DAYS_IN_MONTH = 30


class Client:
    """Клиент с живым соединением.

    Запись события стоит десятки микросекунд, а установка соединения на каждый
    запрос стоит миллисекунды: на `urllib` разница между включенным и
    выключенным журналом утонула бы в накладных расходах самого замера.
    """

    def __init__(self, port: int, timeout: int = 180) -> None:
        self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)

    def post(self, path: str, payload: dict) -> tuple[int, dict | None, float]:
        data = json.dumps(payload).encode("utf-8")
        started = time.monotonic()
        self.connection.request(
            "POST", path, body=data, headers={"Content-Type": "application/json"}
        )
        response = self.connection.getresponse()
        raw = response.read()
        elapsed = (time.monotonic() - started) * 1000
        body = json.loads(raw) if response.status < 300 else None
        return response.status, body, elapsed

    def get(self, path: str) -> dict:
        self.connection.request("GET", path)
        response = self.connection.getresponse()
        return json.loads(response.read())

    def close(self) -> None:
        self.connection.close()


def fetch(url: str, timeout: int = 180) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values), q))


def difference(on: list[float], off: list[float], q: float, draws: int, seed: int) -> dict:
    """Разница квантилей с интервалом: перевыборка каждой выборки отдельно.

    Выборки независимые (два разных процесса сервиса), поэтому парного
    бутстрэпа здесь нет, а интервал шире, чем был бы у парного.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(on), np.asarray(off)
    diffs = np.array(
        [
            np.quantile(rng.choice(a, a.size, replace=True), q)
            - np.quantile(rng.choice(b, b.size, replace=True), q)
            for _ in range(draws)
        ]
    )
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "point_ms": round(float(np.quantile(a, q) - np.quantile(b, q)), 3),
        "lo_ms": round(float(lo), 3),
        "hi_ms": round(float(hi), 3),
        "draws": draws,
    }


def base_env() -> dict[str, str]:
    env = dict(os.environ)
    # Связка замера задается переменной, а не правкой файла конфигурации.
    # Плотная ступень выключена: на хостинге она выключена тоже, и мерить
    # долю записи нужно на той связке, которая отвечает людям.
    env["RUFINRAG_DEFAULT_CONFIG"] = "bm25"
    env["RUFINRAG_DISABLE_DENSE"] = "1"
    env.pop("RUFINRAG_EVENT_LOG", None)
    env.pop("RUFINRAG_EVENT_DIR", None)
    return env


@contextlib.contextmanager
def service(env: dict[str, str]):
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "rufinrag.api:app",
            "--app-dir",
            "src",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 180
        health = None
        while health is None and time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"сервис упал на старте, код {process.returncode}")
            try:
                health = fetch(f"http://127.0.0.1:{PORT}/health", timeout=5)
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(0.5)
        if health is None:
            raise RuntimeError("сервис не ответил на /health за 180 с")
        yield health
    finally:
        process.terminate()
        process.wait(timeout=30)


def request_block(mode: str, event_dir: Path, queries: list[str], repeats: int) -> dict:
    env = base_env()
    if mode == "off":
        env["RUFINRAG_EVENT_LOG"] = "off"
    else:
        env["RUFINRAG_EVENT_DIR"] = str(event_dir)
    client: list[float] = []
    server: list[float] = []
    with service(env) as health:
        http_client = Client(PORT)
        for text in queries[:WARMUP]:
            http_client.post("/search", {"query": text, "top": 10})
        for _ in range(repeats):
            for text in queries:
                code, body, ms = http_client.post("/search", {"query": text, "top": 10})
                if code != 200:
                    raise RuntimeError(f"/search ответил {code}")
                client.append(ms)
                server.append(body["timings_ms"]["total"])
        stats = http_client.get("/health")["event_log"]
        http_client.close()
    return {
        "mode": mode,
        "requests": len(client),
        "warmup_requests": WARMUP,
        "client_ms": client,
        "server_total_ms": server,
        "event_log": stats,
        "corpus_version": health.get("corpus_version"),
        "config": health.get("default_config"),
        "dense": health.get("dense"),
    }


def extra_shapes(event_dir: Path, queries: list[str]) -> dict:
    """Края схемы: длинный запрос с выдачей на 50, запрос на мегабайт, оценка.

    Это те же настоящие события сервиса, просто редких форм: худший размер
    строки должен быть измерен, а не посчитан по схеме.
    """
    env = base_env()
    env["RUFINRAG_EVENT_DIR"] = str(event_dir)
    long_query = " ".join(queries) + " "
    long_query = (long_query * (4000 // len(long_query) + 1))[:4000]
    huge_query = long_query * 260
    out = {
        "repeats": 20,
        "long_query_chars": len(long_query),
        "huge_query_chars": len(huge_query),
    }
    with service(env):
        client = Client(PORT)
        for _ in range(out["repeats"]):
            code, body, _ = client.post("/search", {"query": long_query, "top": 50})
            if code != 200:
                raise RuntimeError(f"запрос на {len(long_query)} знаков получил {code}")
            rated = {"request_id": body["request_id"], "rating": "up"}
            code, _, _ = client.post("/feedback", rated)
            if code != 202:
                raise RuntimeError(f"/feedback ответил {code}")
            code, _, _ = client.post("/search", {"query": huge_query, "top": 10})
            if code != 422:
                raise RuntimeError(f"запрос на {len(huge_query)} знаков получил {code}")
            code, _, _ = client.post("/search", {"query": "   ", "top": 10})
            if code != 422:
                raise RuntimeError(f"запрос из пробелов получил {code}")
        out["event_log"] = client.get("/health")["event_log"]
        client.close()
    return out


def classify(event: dict) -> str:
    """Форма события так, как ее видно в строке, а не как она задумана.

    Отвергнутый запрос бывает двух очень разных размеров: короткий грязный
    ввод и обрезанный до предела текст на мегабайт.
    """
    if event["kind"] == "feedback":
        return "feedback"
    if event["outcome"] != "ok":
        truncated = "-truncated" if "query_truncated_to" in event else ""
        return f"rejected-{event['status']}{truncated}"
    return f"answered-top{event['top']}"


def line_sizes(directory: Path) -> tuple[dict, list[dict]]:
    """Размер строки по формам события и строки типичной формы для замера записи."""
    groups: dict[str, list[int]] = {}
    pool: list[dict] = []
    distinct: dict[str, bytes] = {}
    raw = b""
    for path in sorted(directory.glob("queries-*.jsonl")):
        content = path.read_bytes()
        raw += content
        for line in content.decode("utf-8").splitlines():
            event = json.loads(line)
            name = classify(event)
            groups.setdefault(name, []).append(len(line.encode("utf-8")) + 1)
            if name == "answered-top10":
                pool.append(event)
                distinct.setdefault(event["query"], (line + "\n").encode("utf-8"))
    # Сжатие по файлу замера врет в лучшую сторону: там 20 запросов набора
    # повторяются сотни раз, а живой трафик так не повторяется. Поэтому рядом
    # стоит сжатие по строкам с разными запросами, одна строка на запрос.
    unique = b"".join(distinct.values())
    report = {
        "groups": {
            name: {
                "events": len(sizes),
                "mean_bytes": round(sum(sizes) / len(sizes), 1),
                "min_bytes": min(sizes),
                "max_bytes": max(sizes),
            }
            for name, sizes in sorted(groups.items())
        },
        "file_bytes": len(raw),
        "gzip_bytes": len(gzip.compress(raw, 9)),
        "distinct_queries": len(distinct),
        "distinct_bytes": len(unique),
        "distinct_gzip_bytes": len(gzip.compress(unique, 9)),
    }
    report["gzip_ratio_repeating"] = round(report["file_bytes"] / report["gzip_bytes"], 2)
    report["gzip_ratio_distinct"] = round(
        report["distinct_bytes"] / report["distinct_gzip_bytes"], 2
    )
    return report, pool


def write_throughput(pool: list[dict], directory: Path, threads: int, total: int) -> dict:
    log = events.EventLog(directory)
    per_thread = total // threads
    batches = [
        list(itertools.islice(itertools.cycle(pool), index * per_thread, (index + 1) * per_thread))
        for index in range(threads)
    ]

    def write(batch: list[dict]) -> None:
        for event in batch:
            log.append(event)

    workers = [threading.Thread(target=write, args=(batch,)) for batch in batches]
    started = time.monotonic()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    seconds = time.monotonic() - started
    log.close()
    if log.failed:
        raise RuntimeError(f"не записано {log.failed} событий: {log.last_error}")
    return {
        "threads": threads,
        "events": log.written,
        "seconds": round(seconds, 3),
        "events_per_second": round(log.written / seconds, 1),
        "mean_us_per_event": round(seconds / log.written * 1e6, 2),
        # Загрузка машины рядом с числом: мак общий, и на нем в это время шла
        # вторая сессия. Без этой колонки разброс между повторами необъясним.
        "load_average_1m": round(os.getloadavg()[0], 2),
    }


def normalize_cost(queries: list[str], repeats: int) -> dict:
    """Цена сборки события, которой нет в замере писателя.

    Писатель гоняет уже собранные строки, а сервис на каждый запрос еще и
    токенизирует его для поля `query_normalized`. Разницу между ценой записи и
    ценой журнала на запрос объяснять нечем, если эту работу не измерить.
    """
    started = time.monotonic()
    calls = 0
    for _ in range(repeats):
        for text in queries:
            tokenize(text)
            calls += 1
    seconds = time.monotonic() - started
    return {"calls": calls, "mean_us": round(seconds / calls * 1e6, 2)}


def write_failures(pool: list[dict], blocked: Path, total: int) -> dict:
    """Цена отказа записи: каталога нет, потому что на его месте обычный файл."""
    blocked.write_text("на месте каталога журнала обычный файл", encoding="utf-8")
    log = events.EventLog(blocked)
    batch = list(itertools.islice(itertools.cycle(pool), total))
    started = time.monotonic()
    for event in batch:
        log.append(event)
    seconds = time.monotonic() - started
    return {
        "attempts": total,
        "failed": log.failed,
        "written": log.written,
        "seconds": round(seconds, 3),
        "events_per_second": round(total / seconds, 1),
        "mean_us_per_event": round(seconds / total * 1e6, 2),
        "last_error": log.last_error,
    }


def failing_service(blocked: Path, queries: list[str], requests: int) -> dict:
    """Сервис с недоступным каталогом журнала обязан продолжать отвечать."""
    blocked.write_text("на месте каталога журнала обычный файл", encoding="utf-8")
    env = base_env()
    env["RUFINRAG_EVENT_DIR"] = str(blocked)
    codes: list[int] = []
    times: list[float] = []
    with service(env):
        client = Client(PORT)
        for text in queries[:WARMUP]:
            client.post("/search", {"query": text, "top": 10})
        for index in range(requests):
            text = queries[index % len(queries)]
            code, _, ms = client.post("/search", {"query": text, "top": 10})
            codes.append(code)
            times.append(ms)
        health = client.get("/health")
        client.close()
    return {
        "requests": len(codes),
        "answered_200": sum(1 for code in codes if code == 200),
        "write_attempts": len(codes) + WARMUP,
        "p50_ms": round(quantile(times, 0.5), 3),
        "p95_ms": round(quantile(times, 0.95), 3),
        "health_status": health["status"],
        "event_log": health["event_log"],
    }


def month_table(mean_bytes: float, gzip_ratio: float, ceiling: dict) -> dict:
    """Экстраполяция, а не замер: трафика у сервиса нет, уровни взяты руками.

    Единственный уровень не из воздуха это потолок: найденный нагрузочным
    замером предел одного процесса, сутки под полной нагрузкой.
    """
    ceiling_rps = ceiling["endpoints"]["/search"]["rps"]
    levels = {}
    for per_day in (*TRAFFIC_PER_DAY, round(ceiling_rps * 86_400)):
        levels[str(per_day)] = {
            "requests_per_month": per_day * DAYS_IN_MONTH,
            "raw_mb": round(mean_bytes * per_day * DAYS_IN_MONTH / 1e6, 2),
            "gzip_mb": round(mean_bytes * per_day * DAYS_IN_MONTH / gzip_ratio / 1e6, 2),
        }
    return {
        "kind": "экстраполяция по среднему размеру события, живого трафика нет",
        "days": DAYS_IN_MONTH,
        "gzip_ratio": gzip_ratio,
        "mean_bytes_per_event": round(mean_bytes, 1),
        "base_per_day": BASE_TRAFFIC_PER_DAY,
        "base_reason": (
            "предположение: ссылка в резюме, десятки открытий в сутки, "
            f"{BASE_TRAFFIC_PER_DAY} взято с запасом вверх"
        ),
        "ceiling_per_day": round(ceiling_rps * 86_400),
        "ceiling_reason": (
            f"сутки на найденном пределе одного процесса {round(ceiling_rps, 1)} запроса в "
            f"секунду (results/load.json, связка {ceiling['config']} на {ceiling['users']} "
            "пользователях)"
        ),
        "levels": levels,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=8, help="проходов набора запросов в блоке")
    parser.add_argument("--block-pairs", type=int, default=1, help="повторов встречной четверки")
    parser.add_argument("--writes", type=int, default=20_000, help="событий на один замер записи")
    parser.add_argument("--writes-failed", type=int, default=2_000)
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--writer-reps", type=int, default=3, help="повторов замера записи")
    parser.add_argument("--failure-requests", type=int, default=20)
    parser.add_argument("--draws", type=int, default=2000, help="перевыборок бутстрэпа")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hosted-url", default="", help="только GET /health, запросов не шлет")
    parser.add_argument("--out", type=Path, default=ROOT / "results/logging.json")
    args = parser.parse_args()

    queries = [q.text for q in load_queries().values()][: args.queries]
    if len(queries) < args.queries:
        raise RuntimeError(f"в наборе {len(queries)} запросов")

    # Код сервиса в этот момент правила вторая сессия (задача 4.3), поэтому в
    # отчете стоит коммит рабочего дерева: без него непонятно, какой именно
    # обработчик запроса мерился.
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    report = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": head,
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "load_average_at_start": [round(v, 2) for v in os.getloadavg()],
        "queries": len(queries),
    }

    # Хостинг только опрашивается: /search туда не уходит, иначе события замера
    # попали бы в живой журнал сервиса, который едет в репозиторий.
    if args.hosted_url:
        url = args.hosted_url.rstrip("/")
        health = fetch(f"{url}/health")
        # Поля `event_log` в ответе может не быть вовсе: тогда на хостинге
        # развернут код старше журнала, и это факт замера, а не сбой прогона.
        report["hosted"] = {
            "url": url,
            "note": "только GET /health, нагрузка на бесплатный тариф не подавалась",
            "status": health["status"],
            "version": health["version"],
            "startup_seconds": health["startup_seconds"],
            "event_log": health.get("event_log", "поля нет: развернут код без журнала"),
        }
        print(f"хостинг: {report['hosted']}", flush=True)

    with tempfile.TemporaryDirectory(prefix="rufinrag-logging-") as workspace:
        work = Path(workspace)
        event_dir = work / "events"

        blocks = []
        for index, mode in enumerate(BLOCK_PAIR * args.block_pairs):
            started = time.monotonic()
            block = request_block(mode, event_dir, queries, args.repeats)
            block["block"] = index
            blocks.append(block)
            print(
                f"блок {index} журнал {mode}: {block['requests']} запросов, "
                f"p50 {round(quantile(block['client_ms'], 0.5), 2)} мс, "
                f"{round(time.monotonic() - started, 1)} с",
                flush=True,
            )

        shapes = extra_shapes(event_dir, queries)
        print(f"редкие формы события: {shapes}", flush=True)

        sizes, pool = line_sizes(event_dir)
        print(f"размеры строк: {sizes['groups']}", flush=True)

        # Повторы, а не один прогон: разброс от чужой нагрузки на маке больше,
        # чем разница между числом потоков, и его надо видеть в отчете.
        runs = []
        for threads in [int(t) for t in args.threads.split(",")]:
            reps = [
                write_throughput(pool, work / f"writer-{threads}-{rep}", threads, args.writes)
                for rep in range(args.writer_reps)
            ]
            rates = sorted(r["events_per_second"] for r in reps)
            runs.append(
                {
                    "threads": threads,
                    "repeats": len(reps),
                    "median_events_per_second": float(np.median(rates)),
                    "min_events_per_second": rates[0],
                    "max_events_per_second": rates[-1],
                    "median_us_per_event": round(1e6 / float(np.median(rates)), 2),
                    "reps": reps,
                }
            )
            print(
                f"запись в {threads} поток: медиана {runs[-1]['median_events_per_second']} "
                f"событий в секунду, разброс {rates[0]}-{rates[-1]}",
                flush=True,
            )

        normalize = normalize_cost(queries, args.writes // len(queries))
        print(f"токенизация запроса: {normalize['mean_us']} мкс на запрос", flush=True)

        failed_writes = write_failures(pool, work / "blocked-writer", args.writes_failed)
        print(f"отказ записи: {failed_writes['events_per_second']} попыток в секунду", flush=True)

        failure = failing_service(work / "blocked-service", queries, args.failure_requests)
        print(f"сервис при отказе журнала: {failure}", flush=True)

    on = [ms for b in blocks if b["mode"] == "on" for ms in b["client_ms"]]
    off = [ms for b in blocks if b["mode"] == "off" for ms in b["client_ms"]]
    server_total = {
        mode: [ms for b in blocks if b["mode"] == mode for ms in b["server_total_ms"]]
        for mode in ("on", "off")
    }
    hosted_p50 = json.loads((ROOT / "results/latency-hosted.json").read_text(encoding="utf-8"))[
        "p50_ms"
    ]
    # Потолок трафика берется из нагрузочного замера, а не придумывается здесь.
    load_runs = json.loads((ROOT / "results/load.json").read_text(encoding="utf-8"))["runs"]
    ceiling = max(
        (r for r in load_runs if r["config"] == "bm25"),
        key=lambda r: r["endpoints"]["/search"]["rps"],
    )
    typical = sizes["groups"]["answered-top10"]

    report.update(
        {
            "load_average_at_end": [round(v, 2) for v in os.getloadavg()],
            "service": {
                "config": blocks[0]["config"],
                "dense": blocks[0]["dense"],
                "corpus_version": blocks[0]["corpus_version"],
                "top": 10,
            },
            "writer": {
                "pool_events": len(pool),
                "events_per_run": args.writes,
                "runs": runs,
                "lock_cost": {
                    "single_thread_eps": runs[0]["median_events_per_second"],
                    "best_multi_thread_eps": max(r["median_events_per_second"] for r in runs[1:]),
                    "threads_of_best": max(runs[1:], key=lambda r: r["median_events_per_second"])[
                        "threads"
                    ],
                },
                "failed_writes": failed_writes,
                "query_normalized_cost": normalize,
            },
            "request": {
                "order": list(BLOCK_PAIR * args.block_pairs),
                "requests_per_mode": len(on),
                "on": {
                    "p50_ms": round(quantile(on, 0.5), 3),
                    "p95_ms": round(quantile(on, 0.95), 3),
                    "events_written": sum(
                        b["event_log"]["written"] for b in blocks if b["mode"] == "on"
                    ),
                },
                "off": {
                    "p50_ms": round(quantile(off, 0.5), 3),
                    "p95_ms": round(quantile(off, 0.95), 3),
                    "event_log": "off",
                },
                "difference_p50": difference(on, off, 0.5, args.draws, args.seed),
                "difference_p95": difference(on, off, 0.95, args.draws, args.seed),
                "server_total_p50_ms": {
                    mode: round(quantile(server_total[mode], 0.5), 3) for mode in ("on", "off")
                },
                "hosted_p50_ms": hosted_p50,
            },
            "size": sizes,
            "month": month_table(typical["mean_bytes"], sizes["gzip_ratio_distinct"], ceiling),
            "failure": failure,
            "shapes": shapes,
        }
    )
    # Выборки задержек целиком в отчет не идут: это тысячи чисел, а квантили и
    # интервал посчитаны здесь же.
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записано в {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
