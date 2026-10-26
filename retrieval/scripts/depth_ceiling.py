#!/usr/bin/env python3
"""Потолок первой ступени: сколько эталонов вообще попадает в выдачу.

Вопрос, на который отвечает замер. Переранжирование может только
переставить то, что первая ступень уже нашла. Если Recall@200 заметно выше
Recall@10, задача в ранжировании, и кросс-энкодер уместен. Если Recall@200
низкий, то переставлять нечего, и упирается всё в поиск или в корпус, а
усилия в реранкер пропадут.

Разбивка по подвыборкам обязательна: заголовочная цифра смешивает живые
вопросы практиков и синтетику, сгенерированную по фрагменту, а у них разный
потолок по построению.

Выдачи берутся глубиной 200 (`data/runs/depth200__*.jsonl`), гибрид
считается тем же RRF, что и в основных прогонах.

Запуск: python3 scripts/depth_ceiling.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.retrieval.hybrid import rrf  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
DEPTHS = (10, 50, 100, 200)


def load_run(path: str) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        return {r["query_id"]: r["ranked"] for r in (json.loads(l) for l in f if l.strip())}


def recall_values(run: dict[str, list[str]], qrels: dict, qids: list[str], k: int) -> np.ndarray:
    return np.array([M.recall_at_k(run.get(q, []), qrels[q], k) for q in qids], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data/results/depth-ceiling.json"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "docs/raw/depth_ceiling.txt"))
    args = ap.parse_args()

    qrels = read_qrels(os.path.join(ROOT, "data", "queries", "qrels.tsv"))
    podvyborki = json.load(open(os.path.join(ROOT, "data/queries/podvyborki.json"), encoding="utf-8"))

    bm25 = load_run(os.path.join(RUNDIR, "depth200__bm25.jsonl"))
    runs = {"bm25": bm25}
    # Гибрид считается из тех же выдач: RRF складывает обратные ранги, и
    # глубина 200 у слагаемых дает глубину 200 у суммы.
    for model in ("rufin", "user-bge-m3"):
        path = os.path.join(RUNDIR, f"depth200__dense-{model}.jsonl")
        if not os.path.exists(path):
            continue
        dense = load_run(path)
        runs[f"dense-{model}"] = dense
        runs[f"hybrid-{model}"] = {
            q: [doc for doc, _ in rrf([
                [(doc, 0.0) for doc in bm25.get(q, [])],
                [(doc, 0.0) for doc in dense.get(q, [])],
            ], top=200)]
            for q in set(bm25) | set(dense)
        }

    scored = [q for q in qrels if any(g > 0 for g in qrels[q].values())]
    groups = {"все размеченные": scored}
    for name, ids in podvyborki.items():
        inside = [q for q in ids if q in qrels and any(g > 0 for g in qrels[q].values())]
        if len(inside) >= 20:
            groups[name] = inside

    report = {"depths": list(DEPTHS), "queries_scored": len(scored), "groups": {},
              "live": {}, "vectors_check": {}}
    for group, qids in groups.items():
        rows = {}
        for name, run in runs.items():
            rows[name] = {}
            for k in DEPTHS:
                interval = M.bootstrap_ci(recall_values(run, qrels, qids, k))
                rows[name][f"Recall@{k}"] = {
                    "mean": round(interval.mean, 4),
                    "lo": round(interval.lo, 4),
                    "hi": round(interval.hi, 4),
                }
            top = rows[name][f"Recall@{DEPTHS[-1]}"]["mean"]
            shallow = rows[name][f"Recall@{DEPTHS[0]}"]["mean"]
            rows[name]["headroom"] = round(top - shallow, 4)
        report["groups"][group] = {"queries": len(qids), "configs": rows}

    # Числа, которые идут в выводы документа, тоже считаются здесь: иначе они
    # живут в тексте без файла, и пересчитать их нечем.
    live = groups.get("живые", [])
    if live:
        for name, run in runs.items():
            empty = sum(1 for q in live if not set(qrels[q]) & set(run.get(q, [])[:DEPTHS[-1]]))
            report["live"][name] = {"queries": len(live), "no_hit_at_200": empty}
        for a, b in (("hybrid-user-bge-m3", "bm25"),
                     ("hybrid-user-bge-m3", "hybrid-rufin"),
                     ("dense-user-bge-m3", "dense-rufin"),
                     ("hybrid-rufin", "bm25")):
            if a not in runs or b not in runs:
                continue
            for k in (10, DEPTHS[-1]):
                diff, p_value = M.paired_diff_ci(recall_values(runs[a], qrels, live, k),
                                                 recall_values(runs[b], qrels, live, k))
                report["live"][f"{a} минус {b}, Recall@{k}"] = {
                    "mean": round(diff.mean, 4), "lo": round(diff.lo, 4),
                    "hi": round(diff.hi, 4), "p": round(p_value, 4)}

    # Сверка пересчитанных векторов с выдачей, сохраненной в части А: по ней
    # сказано «те же векторы с точностью до железа», и это надо чем-то держать.
    saved = os.path.join(RUNDIR, "base__dense-user-bge-m3.jsonl")
    if "dense-user-bge-m3" in runs and os.path.exists(saved):
        before = load_run(saved)
        mine = runs["dense-user-bge-m3"]
        common = sorted(set(before) & set(mine) & set(scored))
        vals_mine = M.per_query({q: mine[q] for q in common}, {q: qrels[q] for q in common})
        vals_before = M.per_query({q: before[q] for q in common}, {q: qrels[q] for q in common})
        diff, p_value = M.paired_diff_ci(vals_mine["NDCG@10"], vals_before["NDCG@10"])
        overlap = float(np.mean([len(set(mine[q][:10]) & set(before[q][:10])) / 10
                                 for q in common]))
        report["vectors_check"] = {
            "queries": len(common),
            "ndcg_recomputed": round(float(np.nanmean(vals_mine["NDCG@10"])), 4),
            "ndcg_saved": round(float(np.nanmean(vals_before["NDCG@10"])), 4),
            "diff": {"mean": round(diff.mean, 4), "lo": round(diff.lo, 4),
                     "hi": round(diff.hi, 4), "p": round(p_value, 4)},
            "overlap_top10": round(overlap, 4),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    for group, data in report["groups"].items():
        say(f"\n{group}, запросов {data['queries']}")
        say(f"{'конфигурация':<14}" + "".join(f"{'R@' + str(k):>10}" for k in DEPTHS) + f"{'запас':>9}")
        for name, row in data["configs"].items():
            line = f"{name:<14}" + "".join(f"{row['Recall@' + str(k)]['mean']:>10.3f}" for k in DEPTHS)
            say(line + f"{row['headroom']:>9.3f}")

    if report["live"]:
        say("\nживые вопросы, выводы документа")
        for name, row in report["live"].items():
            say(f"  {name}: {row}")
    if report["vectors_check"]:
        say(f"\nсверка пересчитанных векторов с частью А: {report['vectors_check']}")

    # Выгрузка в docs/raw коммитится, в отличие от data/results: по ней видно,
    # что таблица в документе не набрана руками.
    os.makedirs(os.path.dirname(args.raw), exist_ok=True)
    with open(args.raw, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nфайлы: {args.out}, {args.raw}")


if __name__ == "__main__":
    main()
