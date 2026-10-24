"""Граф ссылок между актами корпуса: числа в results/graph.json.

Запуск: python scripts/build_graph.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.config import data_dir  # noqa: E402
from rufinrag.corpus import corpus_path, load_corpus  # noqa: E402
from rufinrag.graph import build_graph  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results/graph.json")
    args = parser.parse_args()

    base = data_dir()
    corpus = load_corpus(corpus_path(base), base / "index/texts.blocks")
    graph = build_graph(corpus)
    stats = asdict(graph.stats)
    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "corpus_version": corpus.version,
        "stats": stats,
        "degree": {
            "out_median": _median([len(v) for v in graph.forward.values()]),
            "out_max": max((len(v) for v in graph.forward.values()), default=0),
            "in_max": max((len(v) for v in graph.backward.values()), default=0),
        },
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


if __name__ == "__main__":
    raise SystemExit(main())
