"""Сверка выдачи сервиса с выдачей бенчмарка части А.

Зачем число, а не просто тест. Код поиска перенесен из части А, но считается он
на другом железе и в другой точности: там векторы корпуса считались на
видеокорте в половинной точности и поиск шел через FAISS, здесь умножение идет
в numpy в одинарной точности на процессоре. BM25 от этого не зависит и
совпадает точно, а плотная ступень переставляет близкие по оценке фрагменты.
Эта разница публикуется числом, потому что иначе читатель README не знает, та
же ли это связка, что в бенчмарке.

Запуск: python scripts/check_reproducibility.py [--queries 200]
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

from rufinrag.benchmark import load_queries, load_run  # noqa: E402
from rufinrag.search import SearchService  # noqa: E402

DEPTH = 10


def compare(mine: list[str], theirs: list[str]) -> dict[str, bool]:
    return {
        "order": mine == theirs,
        "composition": set(mine) == set(theirs),
        "top1": bool(mine) and bool(theirs) and mine[0] == theirs[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "results/reproducibility.json")
    args = parser.parse_args()

    service = SearchService.build()
    queries = load_queries()
    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "corpus_version": service.corpus_version,
        "depth": DEPTH,
        "note": "сверка с выдачами части А, посчитанными на видеокарте",
        "configs": {},
    }

    for name in ("bm25", "hybrid"):
        reference = load_run(name)
        ids = sorted(set(reference) & set(queries))[: args.queries]
        counts = {"order": 0, "composition": 0, "top1": 0}
        for query_id in ids:
            mine = [h.chunk_id for h in service.search(queries[query_id].text, DEPTH, name).hits]
            for key, ok in compare(mine, reference[query_id][:DEPTH]).items():
                counts[key] += ok
        report["configs"][name] = {"queries": len(ids), **counts}

    if service.dense is not None:
        reference = load_run("dense")
        ids = sorted(set(reference) & set(queries))[: args.queries]
        counts = {"order": 0, "composition": 0, "top1": 0}
        for query_id in ids:
            vector = service.encoder.encode_one(queries[query_id].text)
            mine = [chunk_id for chunk_id, _ in service.dense.search(vector, k=DEPTH)]
            for key, ok in compare(mine, reference[query_id][:DEPTH]).items():
                counts[key] += ok
        report["configs"]["dense"] = {"queries": len(ids), **counts}
    else:
        report["configs"]["dense"] = {"skipped": service.degraded}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, row in report["configs"].items():
        print(name, row)
    print(f"записано в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
