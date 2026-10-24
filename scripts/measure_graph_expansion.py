"""Расширение выдачи по цитированию: настройка на dev и замер на подвыборках.

Порядок повторяет правила, записанные до чисел (EXPERIMENTS, S20): сетка
перебирается только на подвыборке `dev`, победитель замораживается, числа для
отчета считаются на подвыборках, в настройке не участвовавших.

Режимов два. `fusion` сливает базовую выдачу и список по графу обратными
рангами с весом. `reserve` отдает расширению последние `r` мест первой десятки
и базовый порядок не трогает: этот режим появился после того, как первый
проиграл во всей сетке, и записан в S20 отдельным предсказанием.

База это замороженные выдачи `data/benchmark/run-*.jsonl` глубиной 50: плотная
ступень не пересчитывается, иначе числа перестанут быть сравнимыми с прежними.

Запуск: python scripts/measure_graph_expansion.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_qrels, load_queries, load_run  # noqa: E402
from rufinrag.config import data_dir  # noqa: E402
from rufinrag.corpus import corpus_path, load_corpus  # noqa: E402
from rufinrag.graph import build_graph, cited_candidates, reserve_merge  # noqa: E402
from rufinrag.metrics import evaluate, ndcg_at_k, paired_diff_ci, per_query  # noqa: E402
from rufinrag.retrieval.bm25 import BM25Index  # noqa: E402
from rufinrag.retrieval.fusion import rrf  # noqa: E402

TOP = 10
GRIDS = {
    "fusion": {
        "direction": ("forward", "backward", "both"),
        "seeds": (3, 5, 10),
        "depth": (5, 10),
        "weight": (0.3, 0.5, 1.0),
    },
    "reserve": {
        "direction": ("forward", "backward", "both"),
        "seeds": (3, 5),
        "reserve": (1, 2, 3),
    },
}


class Cached:
    """BM25 с одним запомненным вектором оценок.

    Сетка прогоняет один и тот же запрос десятками вариантов, а оценки по
    корпусу от варианта не зависят: без кэша замер считал бы их заново.
    """

    def __init__(self, index: BM25Index):
        self.index = index
        self._query: str | None = None
        self._scores: np.ndarray | None = None

    def scores(self, query: str) -> np.ndarray:
        if query != self._query:
            # Копия обязательна: BM25Index переиспользует свой буфер.
            self._scores = self.index.scores(query).copy()
            self._query = query
        return self._scores


def expanded_run(queries, base_run, corpus, graph, bm25, mode, params, ids):
    out: dict[str, list[str]] = {}
    added: list[int] = []
    for query_id in ids:
        base = [(chunk_id, 0.0) for chunk_id in base_run[query_id]]
        depth = params["depth"] if mode == "fusion" else params["reserve"]
        extra, _ = cited_candidates(
            queries[query_id].text,
            base,
            corpus,
            graph,
            bm25,
            direction=params["direction"],
            seeds=params["seeds"],
            depth=depth,
        )
        if mode == "fusion":
            merged = rrf([base, extra], weights=(1.0, params["weight"]), top=TOP)
        else:
            merged = reserve_merge(base, extra, params["reserve"], top=TOP)
        ranked = [chunk_id for chunk_id, _ in merged]
        out[query_id] = ranked
        injected = {chunk_id for chunk_id, _ in extra} - set(base_run[query_id][:TOP])
        added.append(len(injected & set(ranked)))
    return out, added


def ndcg(run, qrels, ids) -> float:
    values = [ndcg_at_k(run.get(q, ()), qrels[q]) for q in ids]
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else 0.0


def paired(run, base_run, qrels, ids) -> dict:
    sub = {q: qrels[q] for q in ids}
    new_ids, new_values = per_query({q: run[q] for q in ids}, sub)
    old_ids, old_values = per_query({q: base_run[q][:TOP] for q in ids}, sub)
    if new_ids != old_ids:
        raise SystemExit("парное сравнение на разных запросах")
    report = paired_diff_ci(new_values, old_values)
    report["queries"] = len(new_ids)
    report["base"] = round(sum(old_values) / len(old_values), 4)
    report["expanded"] = round(sum(new_values) / len(new_values), 4)
    report["won"] = sum(1 for a, b in zip(new_values, old_values, strict=True) if a > b)
    report["lost"] = sum(1 for a, b in zip(new_values, old_values, strict=True) if a < b)
    return report


def seed_acts(corpus, ranked, seeds: int) -> list[str]:
    acts: list[str] = []
    for chunk_id in ranked[:seeds]:
        act_id = corpus.get(chunk_id).act_id
        if act_id not in acts:
            acts.append(act_id)
    return acts


def ceiling(corpus, graph, qrels, act_of, base_run, ids, direction, seeds) -> dict:
    """Потолок прибавки: у скольких запросов ответ лежит в соседнем акте и вне десятки."""
    reachable = 0
    for query_id in ids:
        neighbors = set(graph.neighbors(seed_acts(corpus, base_run[query_id], seeds), direction))
        top = set(base_run[query_id][:TOP])
        if any(
            grade > 0 and act_of[chunk_id] in neighbors and chunk_id not in top
            for chunk_id, grade in qrels[query_id].items()
        ):
            reachable += 1
    return {"queries": len(ids), "reachable": reachable}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results/graph-expansion.json")
    parser.add_argument("--run-out", type=Path, default=ROOT / "results/run-hybrid-graph.jsonl")
    args = parser.parse_args()

    base = data_dir()
    corpus = load_corpus(corpus_path(base), base / "index/texts.blocks")
    graph = build_graph(corpus)
    bm25 = Cached(BM25Index.load(base / "index/bm25.npz"))
    qrels = load_qrels()
    queries = load_queries()
    runs = {name: load_run(name) for name in ("bm25", "hybrid")}

    act_of: dict[str, str] = {}
    for rel in qrels.values():
        for chunk_id in rel:
            if chunk_id not in act_of:
                act_of[chunk_id] = corpus.get(chunk_id).act_id

    # Подвыборка «связанные акты»: правило из S20, записанное до отбора.
    linked: list[str] = []
    connected: list[str] = []
    for query_id, rel in sorted(qrels.items()):
        acts = {act_of[c] for c, grade in rel.items() if grade > 0}
        if len(acts) < 2:
            continue
        linked.append(query_id)
        if any(
            b in graph.forward.get(a, ()) or b in graph.backward.get(a, ())
            for a in acts
            for b in acts
            if a != b
        ):
            connected.append(query_id)

    dev = [q for q in sorted(qrels) if queries[q].podvyborka == "dev"]
    live = [q for q in sorted(qrels) if queries[q].podvyborka == "живые"]
    everything = sorted(qrels)
    linked_no_dev = [q for q in linked if queries[q].podvyborka != "dev"]
    no_dev = [q for q in everything if queries[q].podvyborka != "dev"]
    slices = (
        ("linked", linked),
        ("linked_connected", connected),
        ("linked_without_dev", linked_no_dev),
        ("live", live),
        ("all", everything),
        ("all_without_dev", no_dev),
        ("dev", dev),
    )

    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "corpus_version": corpus.version,
        "graph": {
            "edges": graph.stats.edges,
            "acts_with_references": graph.stats.acts_with_references,
            "build_seconds": round(graph.stats.build_seconds, 2),
        },
        "subsamples": {name: len(ids) for name, ids in slices},
        "min_queries_for_a_result": 30,
        "variants": {},
    }

    for mode, grid in GRIDS.items():
        t0 = time.monotonic()
        tuning = []
        for values in product(*grid.values()):
            params = dict(zip(grid, values, strict=True))
            run, added = expanded_run(
                queries, runs["hybrid"], corpus, graph, bm25, mode, params, dev
            )
            tuning.append(
                {**params, "dev_ndcg10": ndcg(run, qrels, dev), "added_mean": _mean(added)}
            )
        key = "depth" if mode == "fusion" else "reserve"
        tuning.sort(key=lambda row: (-row["dev_ndcg10"], row[key], row["seeds"]))
        best = {name: tuning[0][name] for name in grid}
        block = {
            "tuning": {
                "on": "dev",
                "queries": len(dev),
                "grid": {k: list(v) for k, v in grid.items()},
                "seconds": round(time.monotonic() - t0, 1),
                "points": len(tuning),
                "best": best,
                "top": tuning[:3],
                "worst": tuning[-1],
                "points_above_base": sum(
                    1 for row in tuning if row["dev_ndcg10"] > ndcg(runs["hybrid"], qrels, dev)
                ),
            },
            "ceiling": {
                name: ceiling(
                    corpus,
                    graph,
                    qrels,
                    act_of,
                    runs["hybrid"],
                    ids,
                    best["direction"],
                    best["seeds"],
                )
                for name, ids in slices
            },
            "levels": {},
            "difference_ndcg10": {},
        }
        for base_name in ("hybrid", "bm25"):
            base_run = runs[base_name]
            run, added = expanded_run(
                queries, base_run, corpus, graph, bm25, mode, best, everything
            )
            block["levels"][base_name] = {
                "base": evaluate({q: v[:TOP] for q, v in base_run.items() if q in qrels}, qrels),
                "expanded": evaluate(run, qrels),
                "added_into_top10_mean": _mean(added),
            }
            block["difference_ndcg10"][base_name] = {
                name: paired(run, base_run, qrels, ids) for name, ids in slices if ids
            }
            if base_name == "hybrid" and mode == "reserve":
                _write_run(args.run_out, run)
        report["variants"][mode] = block

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _write_run(path: Path, run: dict[str, list[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for query_id in sorted(run):
            row = {"query_id": query_id, "ranked": run[query_id]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean(values: list[int]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
