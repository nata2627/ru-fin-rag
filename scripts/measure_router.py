"""Замер роутера против прямого поиска: что он меняет по качеству и по задержке.

Задача 8.3. Правила маршрутизации и предсказания записаны до кода
(`orchestration/EXPERIMENTS.md`, S25), здесь они только применяются.

Качество меряется той же мерой, что в задаче 7.4: доля ответов, у которых хотя
бы одна цитата взята из фрагмента с оценкой эталона больше нуля. Разница между
роутером и прямым поиском считается парным бутстрэпом на одних и тех же
запросах, потому что ошибки у них скоррелированы: на маршруте прямого поиска
это буквально одно и то же поведение.

Задержка меряется на том же прогоне, медиана и p95 на запрос, и у роутера она
включает выбор маршрута. Первый запрос по маршруту графа платит за сборку графа
ссылок (15,7 с на маке), поэтому он считается отдельно и в медиану не идет: это
цена первого обращения, а не цена маршрута.

Запуск: python scripts/measure_router.py [--sample N]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.answer import AnswerService, load_answer_settings  # noqa: E402
from rufinrag.benchmark import load_qrels, load_queries  # noqa: E402
from rufinrag.metrics import paired_diff_ci  # noqa: E402
from rufinrag.router import ROUTES, Router  # noqa: E402
from rufinrag.search import SearchService  # noqa: E402

OUT_FILE = ROOT / "results/router.json"
DEPTH = 10


def wilson(wins: int, total: int) -> list[float]:
    from statistics import NormalDist

    if total == 0:
        return [0.0, 0.0]
    z = NormalDist().inv_cdf(0.975)
    share = wins / total
    denominator = 1 + z**2 / total
    center = (share + z**2 / (2 * total)) / denominator
    half = z * ((share * (1 - share) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    return [round(center - half, 4), round(center + half, 4)]


def correct(citations, relevant: set[str]) -> float:
    """Та же мера, что в задаче 7.4: хотя бы одна цитата релевантна по эталону."""
    return float(any(citation.chunk_id in relevant for citation in citations))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    args = parser.parse_args()

    search = SearchService.build()
    answer = AnswerService.build(search, settings=load_answer_settings())
    router = Router.build(answer)

    qrels = load_qrels()
    queries = {
        query_id: query
        for query_id, query in load_queries().items()
        if any(grade > 0 for grade in qrels.get(query_id, {}).values())
    }
    if args.sample:
        queries = dict(list(queries.items())[: args.sample])

    routes: Counter[str] = Counter()
    routes_by_subsample: dict[str, Counter[str]] = {}
    router_correct: list[float] = []
    direct_correct: list[float] = []
    router_ms: list[float] = []
    direct_ms: list[float] = []
    refused = {"router": 0, "direct": 0}
    clarified = 0
    graph_first_ms = 0.0

    for query_id, query in queries.items():
        relevant = {chunk for chunk, grade in qrels[query_id].items() if grade > 0}

        t0 = time.monotonic()
        run = router.run(query.text, top=DEPTH)
        router_elapsed = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        plain = answer.answer(query.text, top=DEPTH)
        direct_elapsed = (time.monotonic() - t0) * 1000

        routes[run.route] += 1
        routes_by_subsample.setdefault(query.podvyborka, Counter())[run.route] += 1
        if run.clarify is not None:
            clarified += 1
        # Сборка графа ссылок происходит один раз, при первом запросе по этому
        # маршруту. Держать ее в медиане значит мерить не маршрут, а холодный
        # старт, поэтому она откладывается в свое поле.
        if run.route == "граф" and graph_first_ms == 0.0:
            graph_first_ms = round(router_elapsed, 1)
            continue

        router_ms.append(router_elapsed)
        direct_ms.append(direct_elapsed)
        router_correct.append(correct(run.answer.citations if run.answer else (), relevant))
        direct_correct.append(correct(plain.citations, relevant))
        refused["router"] += int(run.answer is not None and not run.answer.answered)
        refused["direct"] += int(not plain.answered)

    hits_router = int(sum(router_correct))
    hits_direct = int(sum(direct_correct))
    total = len(router_correct)
    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "corpus_version": search.corpus_version,
        "answer_version": answer.settings.version,
        "queries": len(queries),
        "queries_compared": total,
        "rule": {
            "text": "маршрут выбирается правилом S25, качество считается мерой задачи 7.4",
            "written_before": "orchestration/EXPERIMENTS.md, S25",
        },
        "routes": {name: routes.get(name, 0) for name in ROUTES},
        "routes_by_subsample": {
            name: {route: tally.get(route, 0) for route in ROUTES}
            for name, tally in sorted(routes_by_subsample.items())
        },
        "clarified": clarified,
        "quality": {
            "router": {
                "correct": hits_router,
                "of": total,
                "share": round(hits_router / total, 4) if total else None,
                "ci_wilson": wilson(hits_router, total),
            },
            "direct": {
                "correct": hits_direct,
                "of": total,
                "share": round(hits_direct / total, 4) if total else None,
                "ci_wilson": wilson(hits_direct, total),
            },
            "difference": paired_diff_ci(router_correct, direct_correct),
        },
        "latency_ms": {
            "router": {
                "median": round(statistics.median(router_ms), 1),
                "p95": round(sorted(router_ms)[int(0.95 * (len(router_ms) - 1))], 1),
            },
            "direct": {
                "median": round(statistics.median(direct_ms), 1),
                "p95": round(sorted(direct_ms)[int(0.95 * (len(direct_ms) - 1))], 1),
            },
            "difference_median": round(
                statistics.median(router_ms) - statistics.median(direct_ms), 1
            ),
            "graph_first_query": graph_first_ms,
        },
        "refused": refused,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    quality = report["quality"]
    print(
        "маршруты: "
        + ", ".join(f"{name} {routes.get(name, 0)}" for name in ROUTES)
        + f" из {len(queries)}"
    )
    print(
        f"корректная ссылка: роутер {quality['router']['correct']} из {total}, "
        f"прямой поиск {quality['direct']['correct']} из {total}, "
        f"разница {quality['difference']['mean']} "
        f"[{quality['difference']['lo']}; {quality['difference']['hi']}]"
    )
    print(
        f"задержка: роутер {report['latency_ms']['router']['median']} мс медиана, "
        f"прямой поиск {report['latency_ms']['direct']['median']} мс"
    )
    print(f"записано в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
