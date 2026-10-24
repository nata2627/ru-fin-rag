"""Регрессия качества на замороженном наборе: то же железо, те же вопросы.

Ночной прогон обновляет корпус, а значит меняет и выдачу. Проверка такая: те
же 991 размеченных вопроса прогоняются через поиск сервиса, метрики сравниваются
с эталоном из `results/regression-baseline.json`, и падение больше порога
останавливает прогон. Эталон создается первым запуском и дальше меняется руками,
коммитом, а не прогоном: иначе метрика тихо сползает по шагу за ночь.

Порог падения не стоит в коде: он читается из `results/alert-threshold.json`,
где лежит замер разброса метрики от прогона к прогону и от доливки корпуса
(`scripts/measure_alert_threshold.py`). Пробитый порог это алерт: прогон
краснеет, текст алерта с числом и знаменателем уходит в отчет и в issue
ночного прогона.

Запуск: python scripts/run_regression.py [--config bm25] [--update-baseline]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_qrels, load_queries  # noqa: E402
from rufinrag.config import load_catalog  # noqa: E402
from rufinrag.metrics import DEPTH, evaluate, queries_scored  # noqa: E402
from rufinrag.pipeline import alert_threshold  # noqa: E402
from rufinrag.search import SearchService  # noqa: E402

WATCHED = "NDCG@10"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bm25")
    parser.add_argument("--out", type=Path, default=ROOT / "results/regression.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/regression-baseline.json")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    # Плотная часть поднимается только там, где связка ее просит. Иначе
    # лексическую регрессию нельзя пересчитать на машине, где векторы отстали
    # от корпуса после ночной докачки: сборка отказывается, хотя проверяемой
    # связке векторы не нужны вовсе.
    config = load_catalog().get(args.config)
    needs_dense = config.first_stage != "bm25" or config.dense_model is not None
    service = SearchService.build(with_dense=needs_dense)
    qrels = load_qrels()
    queries = load_queries()

    t0 = time.monotonic()
    run = {
        query_id: [
            hit.chunk_id for hit in service.search(query.text, top=DEPTH, config=args.config).hits
        ]
        for query_id, query in queries.items()
        if query_id in qrels
    }
    seconds = time.monotonic() - t0

    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "config": args.config,
        "corpus_version": service.corpus_version,
        "chunks": len(service.corpus),
        "queries_scored": queries_scored(qrels),
        "queries_total": len(queries),
        "seconds": round(seconds, 1),
        "metrics": evaluate(run, qrels),
    }

    if args.update_baseline or not args.baseline.exists():
        # Замененный эталон остается в файле строкой «replaced»: иначе прежнее
        # число живет только в истории git, а в README его приходится писать
        # руками. Переписывается при этом одна строка, не цепочка: хранится тот
        # эталон, который заменили сейчас, со своей датой.
        if args.baseline.exists():
            previous = json.loads(args.baseline.read_text(encoding="utf-8"))
            report["replaced"] = {
                "measured_at": previous["measured_at"],
                "config": previous["config"],
                WATCHED: previous["metrics"][WATCHED],
            }
        args.baseline.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"эталон записан в {args.baseline}")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    tolerance = alert_threshold(ROOT)
    drop = baseline["metrics"][WATCHED] - report["metrics"][WATCHED]
    report["baseline"] = {
        "measured_at": baseline["measured_at"],
        "corpus_version": baseline["corpus_version"],
        WATCHED: baseline["metrics"][WATCHED],
        "drop": round(drop, 4),
        "tolerance": tolerance,
    }
    report["passed"] = drop <= tolerance
    # Текст алерта собирается здесь, а не в прогоне: issue ночного прогона и
    # вывод команды читают одну и ту же строку, и расходиться им негде.
    report["alert"] = {
        "fired": not report["passed"],
        "metric": WATCHED,
        "drop": round(drop, 4),
        "threshold": tolerance,
        "baseline_at": baseline["measured_at"],
        "queries_scored": report["queries_scored"],
        "text": (
            f"{WATCHED} упал на {drop:.4f} против эталона от {baseline['measured_at']} "
            f"при пороге {tolerance} на {report['queries_scored']} размеченных вопросах"
        )
        if not report["passed"]
        else "",
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["metrics"], ensure_ascii=False))
    if not report["passed"]:
        print(f"алерт: {report['alert']['text']}", file=sys.stderr)
        return 1
    print(
        f"регрессия не найдена: {WATCHED} {report['metrics'][WATCHED]}, падение {drop:.4f} "
        f"при пороге {tolerance}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
