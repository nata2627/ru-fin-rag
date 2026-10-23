"""Задержка связок поиска на этой машине.

Финальный замер снимается на маке потому, что пользователь работает не на
видеокарте (TASK, раздел «Ограничения»). Машина подписывается в отчете: замеры
времени без указания железа сравнивать нельзя.

Прогрев отделен от замера: первый запрос тянет за собой ленивую загрузку модели
и прогрев кэшей, и если смешать его с остальными, медиана перестанет описывать
обычный запрос.

Запуск: python scripts/measure_latency.py [--queries 20] [--configs bm25,hybrid]
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_queries  # noqa: E402
from rufinrag.retrieval.text import split_tokens, stem, tokenize  # noqa: E402
from rufinrag.search import SearchService  # noqa: E402

DEFAULT_WARMUP = 3


def rss_mb() -> float:
    # На darwin ru_maxrss возвращает байты, на linux килобайты.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--configs", default="bm25,hybrid")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--out", type=Path, default=ROOT / "results/latency-local.json")
    args = parser.parse_args()

    service = SearchService.build()
    queries = [q.text for q in load_queries().values()][: args.queries]
    if len(queries) < args.queries:
        print(f"в наборе {len(queries)} запросов", file=sys.stderr)

    # Дорогие связки меряются отдельным прогоном на меньшем числе запросов,
    # поэтому отчет дополняется, а число запросов стоит у каждой связки.
    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    # Дата прежнего отчета нужна до того, как шапка перезаписана: у строк,
    # снятых до появления даты у каждой связки, она единственный источник.
    prior_date = report.get("measured_at")
    report.update(
        {
            "measured_at": time.strftime("%Y-%m-%d"),
            "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "device": "cpu",
            "corpus_version": service.corpus_version,
            "load_seconds": round(service.load_seconds, 2),
            "rss_after_load_mb": round(rss_mb(), 1),
        }
    )
    report.setdefault("configs", {})

    for name in args.configs.split(","):
        name = name.strip()
        # Прежний замер этой связки остается в строке «replaced» со своей датой:
        # иначе «было столько, стало столько» в README пришлось бы держать
        # руками, а проверка чисел такое сверить не может.
        previous = report["configs"].get(name)
        for text in queries[: args.warmup]:
            service.search(text, args.top, name)
        totals: list[float] = []
        stages: dict[str, list[float]] = {}
        for text in queries:
            result = service.search(text, args.top, name)
            totals.append(result.timings_ms["total"])
            for stage, value in result.timings_ms.items():
                if stage != "total":
                    stages.setdefault(stage, []).append(value)
        report["configs"][name] = {
            "config_version": service.search(queries[0], args.top, name).config_version,
            # Дата у каждой связки своя: дорогие связки меряются отдельным
            # прогоном, и общая дата в шапке отнесла бы прежний замер к сегодня
            "measured_at": time.strftime("%Y-%m-%d"),
            "queries": len(queries),
            "warmup_queries": args.warmup,
            "p50_ms": round(statistics.median(totals), 2),
            "p95_ms": round(percentile(totals, 0.95), 2),
            "max_ms": round(max(totals), 2),
            "stages_p50_ms": {k: round(statistics.median(v), 2) for k, v in stages.items()},
            "rss_peak_mb": round(rss_mb(), 1),
        }
        if previous:
            report["configs"][name]["replaced"] = {
                "measured_at": previous.get("measured_at", prior_date or report["measured_at"]),
                "p50_ms": previous["p50_ms"],
                "p95_ms": previous["p95_ms"],
            }
        row = report["configs"][name]
        print(
            f"{name}: p50 {row['p50_ms']} мс, p95 {row['p95_ms']} мс, "
            f"ступени {row['stages_p50_ms']}, RSS {row['rss_peak_mb']} МБ",
            flush=True,
        )

    # Цена стеммера в онлайне. Запрос это единственное место, где сервис
    # токенизирует текст: индекс приходит собранным, и документы здесь никто
    # не разбирает. Поэтому прибавка меряется прямо на тех же запросах и в
    # двух видах. С пустым кэшем это худший случай, первый запрос после
    # старта; с теплым так, как сервис работает дальше.
    def tokenization_us(fn, cold: bool) -> float:
        samples = []
        if not cold:
            # Теплый случай это повторный запрос: основы его слов уже в кэше.
            # Без этого прохода «теплый» замер мерил бы тот же холодный кэш,
            # потому что каждый запрос в наборе встречается один раз
            for text in queries:
                fn(text)
        for text in queries:
            if cold:
                stem.cache_clear()
            t0 = time.perf_counter()
            fn(text)
            samples.append((time.perf_counter() - t0) * 1e6)
        return round(statistics.median(samples), 1)

    report["lexical_tokenization"] = {
        "note": "медиана по тем же запросам, микросекунды на запрос",
        "queries": len(queries),
        "tokens_p50": int(statistics.median(len(tokenize(t)) for t in queries)),
        "forms_us_p50": tokenization_us(split_tokens, False),
        "stemmed_cold_us_p50": tokenization_us(tokenize, True),
        "stemmed_warm_us_p50": tokenization_us(tokenize, False),
    }
    row = report["lexical_tokenization"]
    print(
        f"токенизация запроса: {row['tokens_p50']} токенов, по формам "
        f"{row['forms_us_p50']} мкс, со стеммером {row['stemmed_cold_us_p50']} мкс "
        f"на пустом кэше и {row['stemmed_warm_us_p50']} мкс на теплом",
        flush=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записано в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
