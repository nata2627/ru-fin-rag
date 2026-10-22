"""Регрессия качества на замороженном наборе: то же железо, те же вопросы.

Ночной прогон обновляет корпус, а значит меняет и выдачу. Проверка такая: те
же 991 размеченных вопроса прогоняются через поиск сервиса, метрики сравниваются
с эталоном из `results/regression-baseline.json`, и падение больше порога
останавливает прогон. Эталон создается первым запуском и дальше меняется руками,
коммитом, а не прогоном: иначе метрика тихо сползает по шагу за ночь.

Порог 0,010 NDCG@10 взят не с потолка: на 991 вопросе это примерно половина
ширины бутстрэп-интервала части А, то есть сдвиг, который уже нельзя объяснить
шумом выборки. Обоснование числом идет в задаче 5.3, здесь порог рабочий.

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
from rufinrag.metrics import DEPTH, evaluate, queries_scored  # noqa: E402
from rufinrag.search import SearchService  # noqa: E402

TOLERANCE = 0.010
WATCHED = "NDCG@10"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bm25")
    parser.add_argument("--out", type=Path, default=ROOT / "results/regression.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/regression-baseline.json")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    service = SearchService.build()
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
        args.baseline.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"эталон записан в {args.baseline}")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    drop = baseline["metrics"][WATCHED] - report["metrics"][WATCHED]
    report["baseline"] = {
        "measured_at": baseline["measured_at"],
        "corpus_version": baseline["corpus_version"],
        WATCHED: baseline["metrics"][WATCHED],
        "drop": round(drop, 4),
        "tolerance": TOLERANCE,
    }
    report["passed"] = drop <= TOLERANCE
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["metrics"], ensure_ascii=False))
    if not report["passed"]:
        print(
            f"регрессия: {WATCHED} упал на {drop:.4f} против эталона от "
            f"{baseline['measured_at']} при пороге {TOLERANCE}",
            file=sys.stderr,
        )
        return 1
    print(f"регрессия не найдена: {WATCHED} {report['metrics'][WATCHED]}, падение {drop:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
