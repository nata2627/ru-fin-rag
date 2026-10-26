#!/usr/bin/env python3
"""Чем держится метрика на живых вопросах: сколько в первой десятке размечено.

Метрика считает неразмеченный фрагмент промахом. Значит у каждой цифры есть
скрытая характеристика: сколько позиций первой десятки про эту пару вообще
что-нибудь известно. Здесь она считается, и рядом ставится верхняя граница:
метрика в предположении, что все неразмеченные пары релевантны.

Верхняя граница заведомо недостижима, и её смысл не в правдоподобии, а в
границе незнания: истинное значение лежит между нынешним и ней. Точку внутри
этого отрезка даёт только разметка неразмеченных пар, и она требует судьи,
который проходит порог согласия с человеком (`scripts/judge_candidates.py`).

Нижняя граница это нынешняя метрика: ни один неразмеченный фрагмент не
релевантен.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
QDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "pool_coverage.txt")
RESULT = os.path.join(ROOT, "data", "results", "pool-coverage.json")
TOP = 10


def load_runs(config: str) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for name in sorted(os.listdir(RUNDIR)):
        m = re.fullmatch(rf"{re.escape(config)}__(.+)\.jsonl", name)
        if not m:
            continue
        run = {}
        with open(os.path.join(RUNDIR, name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                run[rec["query_id"]] = rec["ranked"]
        out[m.group(1)] = run
    return out


def judged_pairs() -> set[tuple[str, str]]:
    """Пары, про которые известна хоть какая-то оценка, включая нулевую."""
    pairs: set[tuple[str, str]] = set()
    judge = os.path.join(QDIR, "judge.tsv")
    if os.path.exists(judge):
        with open(judge, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if (row.get("otsenka_0_1_2") or "").strip() in ("0", "1", "2"):
                    pairs.add((row["query_id"], row["chunk_id"]))
    for path, qc, cc in ((os.path.join(QDIR, "qrels.tsv"), "query-id", "corpus-id"),
                         (os.path.join(QDIR, "qrels_by_clause.tsv"),
                          "query-id", "corpus-id")):
        if os.path.exists(path):
            with open(path, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    pairs.add((row[qc], row[cc]))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--subset", default="живые")
    ap.add_argument("--metric", default="NDCG@10")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    runs = load_runs(args.config)
    qrels = read_qrels(os.path.join(QDIR, "qrels.tsv"))
    subsets = json.load(open(os.path.join(QDIR, "podvyborki.json"), encoding="utf-8"))
    covered = set.intersection(*(set(run) for run in runs.values()))
    qids = sorted(q for q in subsets[args.subset] if q in qrels and q in covered)
    known = judged_pairs()

    say(f"нарезка {args.config}, подвыборка «{args.subset}», запросов {len(qids)}")
    say(f"конфигураций {len(runs)}, первая десятка, метрика {args.metric}")
    say()

    rows = []
    for cfg, run in sorted(runs.items()):
        trimmed = {q: run[q][:TOP] for q in qids}
        unknown_per_q = [sum(1 for cid in trimmed[q] if (q, cid) not in known)
                         for q in qids]
        now = M.per_query(trimmed, {q: qrels[q] for q in qids})
        # Верхняя граница: каждая неразмеченная пара первой десятки
        # объявляется прямым ответом. Знаменатель Recall растёт вместе
        # с числителем, поэтому Recall от этого может и упасть.
        bound_qrels = {}
        for q in qids:
            rel = dict(qrels[q])
            for cid in trimmed[q]:
                if (q, cid) not in known:
                    rel[cid] = 2
            bound_qrels[q] = rel
        bound = M.per_query(trimmed, bound_qrels)
        interval_now = M.bootstrap_ci(now[args.metric])
        interval_bound = M.bootstrap_ci(bound[args.metric])
        diff, p = M.paired_diff_ci(bound[args.metric], now[args.metric])
        rows.append({
            "конфигурация": cfg,
            "неразмечено в десятке": round(sum(unknown_per_q) / len(qids), 2),
            "запросов с полностью размеченной десяткой":
                sum(1 for n in unknown_per_q if n == 0),
            args.metric: round(interval_now.mean, 3),
            f"{args.metric} верхняя граница": round(interval_bound.mean, 3),
            "запас": round(diff.mean, 3),
            "запас lo": round(diff.lo, 3), "запас hi": round(diff.hi, 3),
        })

    say(f"{'конфигурация':<20} {'нераз.':>7} {'цела':>6} "
        f"{'сейчас':>8} {'граница':>9} {'запас незнания':>24}")
    for r in rows:
        gap = f"{r['запас']:.3f} [{r['запас lo']:.3f}; {r['запас hi']:.3f}]"
        say(f"{r['конфигурация']:<20} {r['неразмечено в десятке']:>7.2f} "
            f"{r['запросов с полностью размеченной десяткой']:>6} "
            f"{r[args.metric]:>8.3f} {r[f'{args.metric} верхняя граница']:>9.3f} "
            f"{gap:>24}")

    # Частота релевантных среди размеченных пар первой десятки. Переносить её
    # на неразмеченные нельзя: размечались те пары, которые прежний пул
    # поднял наверх, то есть заведомо более похожие на ответ. Цифра стоит
    # здесь как ориентир, а не как оценка поправки.
    labelled = [(q, cid) for cfg, run in runs.items() for q in qids
                for cid in run[q][:TOP] if (q, cid) in known]
    unique = set(labelled)
    relevant = sum(1 for q, cid in unique if qrels.get(q, {}).get(cid, 0) > 0)
    say()
    say(f"размеченных пар в первых десятках: {len(unique)} (без повторов между "
        f"конфигурациями), из них релевантны {relevant} "
        f"({100 * relevant / max(1, len(unique)):.0f}%)")
    say("перенос этой доли на неразмеченные пары был бы завышением: размечались "
        "те, что прежний пул поднял наверх")
    say()
    say("нераз. — сколько позиций первой десятки не размечено, в среднем на запрос")
    say("цела — запросов, где размечены все десять позиций")
    say("граница — метрика, если каждая неразмеченная пара окажется прямым ответом")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump({"подвыборка": args.subset, "запросов": len(qids),
                   "метрика": args.metric, "глубина": TOP,
                   "конфигурации": rows}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
