#!/usr/bin/env python3
"""Пункт вместо куска: метки на пунктах и метрики до и после на тех же выдачах.

Устройство единицы «пункт» и правило счета описаны в `rufin.punkt`. Здесь
метки переводятся на пункты (`data/queries/qrels_punkt.tsv`), и каждая
сохраненная выдача `data/runs/base__*.jsonl` считается дважды: по фрагментам,
как раньше, и по пунктам. Новых прогонов поиска нет.

Отдельно печатаются четыре провала, которые `docs/16-razbor-provalov.md`
назвал провалами метрики по проверке кодом: исчезли они или нет.

    python3 scripts/punkt_effect.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin import punkt as P  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.explanations import body_of, clean_clauses, starts_clause  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
RUNDIR = os.path.join(ROOT, "data", "runs")
CHUNKS = os.path.join(ROOT, "data", "chunks", "base.jsonl")
OUT_QRELS = os.path.join(QDIR, "qrels_punkt.tsv")
RESULT = os.path.join(ROOT, "data", "results", "punkt-effect.json")
REPORT = os.path.join(ROOT, "docs", "raw", "punkt_effect.txt")
DOC = os.path.join(ROOT, "docs", "18-nabor-do-i-posle.md")
TABLE_START = "<!-- tablica: punkt_effect.py -->"
TABLE_END = "<!-- /tablica punkt -->"
# Связки и подвыборки таблицы документа. Полная таблица по всем связкам
# лежит в data/results/punkt-effect.json и в docs/raw/punkt_effect.txt.
DOC_RUNS = ("bm25", "dense-user-bge-m3", "hybrid", "hybrid-rerank")
DOC_SUBSETS = ("живые", "заголовочная", "синтетика")

ROUTE_CLAUSE = "по пунктам"
METRIC_NAMES = ("NDCG@10", "Recall@10", "MRR@10")
# Провалы метрики, подтвержденные кодом в разборе двадцати провалов.
METRIC_FAILURES = ("exp0111", "exp0113", "exp0128", "exp0072")
FAILURE_RUN = "hybrid-rerank"


def load_chunks(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            out.append({"chunk_id": c["chunk_id"], "act_id": c["act_id"], "position": c["position"],
                        "section": c["section"] or "", "units": c["units"] or "",
                        "char_start": c["char_start"], "char_end": c["char_end"], "body": body_of(c)})
    return out


def load_runs() -> dict[str, dict[str, list[str]]]:
    out = {}
    for name in sorted(os.listdir(RUNDIR)):
        m = re.fullmatch(r"base__(.+)\.jsonl", name)
        if m:
            with open(os.path.join(RUNDIR, name), encoding="utf-8") as f:
                out[m.group(1)] = {r["query_id"]: r["ranked"] for r in map(json.loads, f)}
    return out


def num(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def signed(x: float) -> str:
    return ("+" if x >= 0 else "−") + num(abs(x))


def doc_table(result: dict) -> str:
    rows = {(r["связка"], r["подвыборка"]): r for r in result["таблица"]}
    out = ["| Связка | Подвыборка | Запросов | NDCG@10 по фрагментам | NDCG@10 по пунктам | Разница [95%] |",
           "|---|---|---|---|---|---|"]
    for cfg in DOC_RUNS:
        for sub in DOC_SUBSETS:
            r = rows[(cfg, sub)]
            d = r["разница NDCG@10"]
            out.append(f"| {cfg} | {sub} | {r['запросов']} | {num(r['до']['NDCG@10'])} | "
                       f"{num(r['после']['NDCG@10'])} | {signed(d['среднее'])} "
                       f"[{signed(d['lo'])}; {signed(d['hi'])}] |")
    f = result["провалы"]
    out += ["", f"Строк «связка и подвыборка», где NDCG@10 упал и интервал не накрывает ноль: "
                f"{result['упало с интервалом ниже нуля']} из {result['строк таблицы']}. Провалов {f['связка']} "
                f"на живых (ни одного позитива в первой десятке): {f['до']} по фрагментам, {f['после']} по "
                f"пунктам. Из {len(f['четыре провала метрики'])} провалов метрики разбора исчезли "
                f"{len(f['исчезли'])}."]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default=os.path.join(QDIR, "qrels.tsv"))
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    qrels = read_qrels(args.qrels)
    chunks = load_chunks(CHUNKS)
    groups = P.piece_groups(chunks)
    act_chunks: dict[str, list[dict]] = collections.defaultdict(list)
    for c in chunks:
        act_chunks[c["act_id"]].append(c)

    live = {}
    with open(os.path.join(QDIR, "explan_live.jsonl"), encoding="utf-8") as f:
        for q in map(json.loads, f):
            live[q["query_id"]] = q
    named = {qid: clean_clauses(q["clauses"]) for qid, q in live.items()
             if q.get("route") == ROUTE_CLAUSE and qid in qrels}
    act_of = {qid: live[qid]["act_id"] for qid in named}

    units = P.build_units(qrels, named, act_chunks, act_of, groups, starts_clause)
    P.write_units(OUT_QRELS, units)
    lost = [(q, c) for q, rel in qrels.items() for c, s in rel.items()
            if s > 0 and not any(c in u.chunks for u in units[q].values())]
    if lost:
        raise SystemExit(f"прежние позитивы выпали из единиц: {lost[:5]}")

    kinds = collections.Counter(u.osnovanie.split(",")[0].split(" ")[0] for us in units.values()
                                for u in us.values())
    say(f"кусков длинных пунктов в корпусе {len(groups)} из {len(chunks)} фрагментов")
    say(f"запросов {len(units)}, единиц {sum(len(u) for u in units.values())}, "
        f"прежних позитивов {sum(1 for r in qrels.values() for s in r.values() if s > 0)}")
    say(f"единицы по виду: {dict(kinds)}")
    say(f"живых с названными пунктами {len(named)}")
    say()

    subsets = json.load(open(os.path.join(QDIR, "podvyborki.json"), encoding="utf-8"))
    runs = load_runs()
    table = []
    for cfg, run in runs.items():
        for name, ids in subsets.items():
            qids = [q for q in ids if q in qrels and q in run]
            if not qids:
                continue
            old = M.per_query({q: run[q] for q in qids}, {q: qrels[q] for q in qids})
            new = {m: np.array([P.metrics(run[q], units[q])[m] for q in qids]) for m in METRIC_NAMES}
            diff, p = M.paired_diff_ci(new["NDCG@10"], old["NDCG@10"])
            row = {"связка": cfg, "подвыборка": name, "запросов": len(qids),
                   "до": {m: round(float(np.nanmean(old[m])), 4) for m in METRIC_NAMES},
                   "после": {m: round(float(np.nanmean(new[m])), 4) for m in METRIC_NAMES},
                   "разница NDCG@10": {"среднее": round(diff.mean, 4), "lo": round(diff.lo, 4),
                                       "hi": round(diff.hi, 4), "p": p}}
            table.append(row)

    for name in subsets:
        rows = [r for r in table if r["подвыборка"] == name]
        if not rows:
            continue
        say(f"== {name}, запросов {rows[0]['запросов']}: NDCG@10 до -> после, разница [95%], Recall@10 до -> после")
        for r in rows:
            d = r["разница NDCG@10"]
            say(f"  {r['связка']:24s} {r['до']['NDCG@10']:.3f} -> {r['после']['NDCG@10']:.3f}  "
                f"{d['среднее']:+.3f} [{d['lo']:+.3f}; {d['hi']:+.3f}]  "
                f"R@10 {r['до']['Recall@10']:.3f} -> {r['после']['Recall@10']:.3f}")
        say()

    worse = [r for r in table if r["разница NDCG@10"]["hi"] < 0]
    say(f"связок и подвыборок, где NDCG@10 упал и интервал не накрывает ноль: {len(worse)} из {len(table)}")

    run = runs[FAILURE_RUN]
    fail_old = [q for q in subsets["живые"] if q in run and q in qrels
                and not set(run[q][:10]) & {c for c, s in qrels[q].items() if s > 0}]
    fail_new = [q for q in fail_old if P.metrics(run[q], units[q])["MRR@10"] == 0]
    gone = [q for q in METRIC_FAILURES if q in fail_old and q not in fail_new]
    say(f"провалов {FAILURE_RUN} на живых (ни одного позитива в десятке): "
        f"до {len(fail_old)}, после {len(fail_new)}")
    say(f"из четырех провалов метрики разбора исчезли {len(gone)}: {', '.join(gone) or 'ни один'}")

    result = {"кусков длинных пунктов": len(groups), "фрагментов": len(chunks),
              "запросов": len(units), "единиц": sum(len(u) for u in units.values()),
              "живых с названными пунктами": len(named), "таблица": table,
              "упало с интервалом ниже нуля": len(worse), "строк таблицы": len(table),
              "провалы": {"связка": FAILURE_RUN, "до": len(fail_old), "после": len(fail_new),
                          "четыре провала метрики": list(METRIC_FAILURES), "исчезли": gone}}
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    body = open(DOC, encoding="utf-8").read()
    pattern = re.compile(re.escape(TABLE_START) + ".*?" + re.escape(TABLE_END), re.S)
    if not pattern.search(body):
        raise SystemExit(f"в {DOC} нет меток таблицы {TABLE_START} ... {TABLE_END}")
    with open(DOC, "w", encoding="utf-8") as f:
        f.write(pattern.sub(lambda _: f"{TABLE_START}\n{doc_table(result)}\n{TABLE_END}", body))


if __name__ == "__main__":
    main()
