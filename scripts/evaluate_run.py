"""Качество выдачи и разница с другой выдачей на тех же запросах.

Нужен там, где выдача посчитана не сервисом, а отдельным прогоном: например
переранжирование считается на видеокарте (`scripts/rerank_run.py`), а метрики
по его результату должны считаться тем же кодом, что все остальные числа
проекта, и с интервалом, а не голым средним.

Запуск: python scripts/evaluate_run.py --run results/rerank.jsonl \
            --baseline data/runs/hybrid.jsonl --name hybrid-rerank
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_qrels  # noqa: E402
from rufinrag.metrics import (  # noqa: E402
    bootstrap_ci,
    evaluate,
    paired_diff_ci,
    per_query,
    queries_scored,
)


def read_run(path: Path) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        rows = (json.loads(line) for line in f if line.strip())
        return {row["query_id"]: row["ranked"] for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline-name", default="базовая выдача")
    parser.add_argument("--out", type=Path, default=ROOT / "results/rerank-quality.json")
    args = parser.parse_args()

    qrels = load_qrels()
    run = read_run(args.run)
    baseline = read_run(args.baseline)

    ids_run, values_run = per_query(run, qrels)
    ids_base, values_base = per_query(baseline, qrels)
    if ids_run != ids_base:
        raise SystemExit("выдачи посчитаны на разных запросах, парное сравнение невозможно")

    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "queries_scored": queries_scored(qrels),
        "runs": {
            args.name: {
                "file": args.run.name,
                "metrics": evaluate(run, qrels),
                "ndcg10": bootstrap_ci(values_run),
            },
            args.baseline_name: {
                "file": args.baseline.name,
                "metrics": evaluate(baseline, qrels),
                "ndcg10": bootstrap_ci(values_base),
            },
        },
        "difference_ndcg10": paired_diff_ci(values_run, values_base),
    }
    if args.out.exists():
        # Замеренное прежде остается в файле, а не только в истории git: в
        # README рядом с новым числом стоит то, которое оно заменило, и писать
        # его руками значит однажды написать не то. Хранится один шаг назад, не
        # цепочка.
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        if "runs" in previous:
            report["replaced"] = {
                "measured_at": previous["measured_at"],
                "runs": {
                    name: row["metrics"]["NDCG@10"] for name, row in previous["runs"].items()
                },
                "difference_ndcg10": {
                    key: previous["difference_ndcg10"][key] for key in ("mean", "lo", "hi")
                },
            }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
