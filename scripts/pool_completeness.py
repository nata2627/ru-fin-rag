#!/usr/bin/env python3
"""Полный пул по объединённой выдаче всех конфигураций на живых вопросах.

Зачем ещё один пул, когда есть `pool_candidates.py`. Тот собирает кандидатов
для запросов, у которых эталона нет вовсе, и берёт их по четырнадцать на
запрос. Здесь задача обратная: у живых вопросов эталон есть, и надо выяснить,
насколько он полон. Поэтому берутся ВСЕ пары из первой десятки каждой
конфигурации, а не выборка из них: метрика считается на первой десятке, и
ровно эти пары её и определяют. Неразмеченная пара в первой десятке читается
метрикой как промах, то есть занижает конфигурацию, которая её нашла.

Пары, размеченные прежде, из задания не выбрасываются, а помечаются. Две
причины. Разметка двух разных моделей в одном эталоне несравнима, и если
судья сменился, пул надо пройти целиком. А на пересечении видно, насколько
новый судья согласен с прежним.

Приоритет: ядро из пяти связок, по которым проект делает выводы, размечается
первым. Если прогон придётся прервать, прерван он будет на второстепенном.

Порядок строк перемешан, и какая конфигурация нашла кандидата, в задании не
написано: иначе оценка подстроится под знакомый метод.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.annotation import RUBRIC  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
REPORT = os.path.join(ROOT, "docs", "raw", "pool_completeness.txt")

# Связки, по которым проект делает выводы: лексическая база, гибрид,
# победитель с переранжированием, лучшая готовая плотная модель и дообученная.
CORE = ["bm25", "hybrid", "hybrid-rerank", "dense-user-bge-m3", "dense-rufin"]


def load_runs(config: str) -> dict[str, dict[str, list[str]]]:
    runs: dict[str, dict[str, list[str]]] = {}
    for name in sorted(os.listdir(RUNDIR)):
        m = re.fullmatch(rf"{re.escape(config)}__(.+)\.jsonl", name)
        if not m:
            continue
        run: dict[str, list[str]] = {}
        with open(os.path.join(RUNDIR, name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                run[rec["query_id"]] = rec["ranked"]
        runs[m.group(1)] = run
    return runs


def read_pairs(path: str, qcol: str, ccol: str) -> set[tuple[str, str]]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {(r[qcol], r[ccol]) for r in csv.DictReader(f, delimiter="\t")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--subset", default="живые")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--out", default=os.path.join(QDIR, "pool_live.tsv"))
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    runs = load_runs(args.config)
    if not runs:
        raise SystemExit(f"в data/runs нет выдач для нарезки «{args.config}»")
    subsets = json.load(open(os.path.join(QDIR, "podvyborki.json"), encoding="utf-8"))
    if args.subset not in subsets:
        raise SystemExit(f"нет подвыборки «{args.subset}»: {', '.join(subsets)}")

    qrels = read_qrels(os.path.join(QDIR, "qrels.tsv"))
    queries = {}
    with open(os.path.join(QDIR, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["query_id"]] = q["text"]

    # Запросы подвыборки, у которых есть и разметка, и выдача у всех
    # конфигураций: только на них метрика и считается.
    covered = set.intersection(*(set(run) for run in runs.values()))
    qids = sorted(q for q in subsets[args.subset] if q in qrels and q in covered)
    say(f"нарезка {args.config}, подвыборка «{args.subset}»: запросов {len(qids)}")
    say(f"конфигураций в выдачах: {len(runs)}, глубина {args.depth}")
    missing = sorted(set(CORE) - set(runs))
    if missing:
        raise SystemExit(f"в выдачах нет связок ядра: {', '.join(missing)}")

    judged = read_pairs(os.path.join(QDIR, "judge.tsv"), "query_id", "chunk_id")
    by_clause = read_pairs(os.path.join(QDIR, "qrels_by_clause.tsv"),
                           "query-id", "corpus-id")
    in_qrels = read_pairs(os.path.join(QDIR, "qrels.tsv"), "query-id", "corpus-id")

    def source(pair: tuple[str, str]) -> str:
        if pair in by_clause:
            return "пункт"
        if pair in judged:
            return "судья"
        if pair in in_qrels:
            return "эталон"
        return "-"

    # Пара попадает в задание один раз, с лучшим приоритетом: 1 — её нашла
    # связка ядра, 2 — только второстепенная.
    priority: dict[tuple[str, str], int] = {}
    found_by: dict[tuple[str, str], int] = collections.Counter()
    for qid in qids:
        for cfg, run in runs.items():
            for cid in run.get(qid, [])[:args.depth]:
                pair = (qid, cid)
                p = 1 if cfg in CORE else 2
                priority[pair] = min(priority.get(pair, 9), p)
                found_by[pair] += 1

    # Выборка человека входит в задание целиком: по ней считается согласие
    # нового судьи с человеком на той же выборке, что и у прежнего.
    human = read_pairs(os.path.join(QDIR, "human_sample.tsv"), "query_id", "chunk_id")
    for pair in human:
        priority[pair] = 0

    chunks = {}
    with open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["chunk_id"] in {cid for _, cid in priority}:
                chunks[c["chunk_id"]] = c

    rows = []
    skipped_text = skipped_query = 0
    for (qid, cid), p in priority.items():
        if qid not in queries:
            skipped_query += 1
            continue
        c = chunks.get(cid)
        if c is None:
            skipped_text += 1
            continue
        rows.append({"query_id": qid, "chunk_id": cid, "prioritet": p,
                     "istochnik": source((qid, cid)),
                     "naydeno_svyazok": found_by.get((qid, cid), 0),
                     "vopros": queries[qid].replace("\t", " ").replace("\n", " "),
                     "fragment": c["text"].replace("\t", " ").replace("\n", " ")})
    # Перемешивание внутри приоритета: порядок не должен подсказывать ответ,
    # но прерванный прогон обязан успеть разметить ядро.
    mixed: list[dict] = []
    for p in sorted({r["prioritet"] for r in rows}):
        part = sorted((r for r in rows if r["prioritet"] == p),
                      key=lambda r: (r["query_id"], r["chunk_id"]))
        random.Random(f"{args.subset}:{p}").shuffle(part)
        mixed.extend(part)
    rows = mixed

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    if skipped_text or skipped_query:
        say(f"пропущено: фрагмент не найден {skipped_text}, запрос не найден {skipped_query}")
    say()
    say(f"пар в задании: {len(rows)}")
    by_p = collections.Counter(r["prioritet"] for r in rows)
    say("по приоритету: " + ", ".join(
        f"{k} ({'выборка человека' if k == 0 else 'ядро' if k == 1 else 'остальные связки'}) "
        f"{v}" for k, v in sorted(by_p.items())))
    by_src = collections.Counter(r["istochnik"] for r in rows)
    say("разметка прежде: " + ", ".join(f"{k} {v}" for k, v in by_src.most_common()))
    say(f"новых пар (никем не размечены): {by_src.get('-', 0)} из {len(rows)}")
    per_q = collections.Counter(r["query_id"] for r in rows if r["prioritet"] != 0)
    if per_q:
        say(f"пар на запрос: в среднем {sum(per_q.values()) / len(per_q):.1f}, "
            f"минимум {min(per_q.values())}, максимум {max(per_q.values())}")
    say(f"файл: {os.path.relpath(args.out, ROOT)}")

    # Рубрика уезжает к судье файлом, а не копией в коде: источник один,
    # константа RUBRIC, и расхождение инструкций судьи и человека невозможно.
    rubric_path = os.path.join(os.path.dirname(args.out), "rubric.txt")
    with open(rubric_path, "w", encoding="utf-8") as f:
        f.write(RUBRIC)
    say(f"инструкция судье: {os.path.relpath(rubric_path, ROOT)}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
