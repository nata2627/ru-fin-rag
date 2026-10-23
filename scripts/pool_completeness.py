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

Приоритет: первой размечается первая десятка пяти связок, по которым проект
делает выводы, потому что на ней стоит опубликованная метрика. Потом хвост
этих же связок до глубины 50, потом то, что нашли только второстепенные
связки. Если прогон придётся прервать, прерван он будет на последнем.

Глубина 50, а не 10. Десятки хватает, чтобы посчитать NDCG@10, и не хватает,
чтобы узнать его настоящее значение: фрагмент, стоящий у одной связки
одиннадцатым, у другой стоит третьим, и неразмеченным он занижает обеих.
Пятьдесят это вся длина сохранённых выдач, дальше брать неоткуда.

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
    ap.add_argument("--depth", type=int, default=50,
                    help="глубина выдачи, с которой берутся кандидаты. "
                         "50 это вся сохранённая выдача")
    ap.add_argument("--metric-depth", type=int, default=10,
                    help="глубина, на которой считается опубликованная "
                         "метрика: по ней назначается первый приоритет")
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

    # Пара попадает в задание один раз, с лучшим приоритетом: 1 — связка ядра
    # поставила её в первую десятку, то есть пара определяет опубликованное
    # число; 2 — связка ядра нашла её глубже десятки; 3 — её нашли только
    # второстепенные связки.
    priority: dict[tuple[str, str], int] = {}
    found_by: dict[tuple[str, str], int] = collections.Counter()
    best_rank: dict[tuple[str, str], int] = {}
    for qid in qids:
        for cfg, run in runs.items():
            for место, cid in enumerate(run.get(qid, [])[:args.depth], 1):
                pair = (qid, cid)
                if cfg in CORE:
                    p = 1 if место <= args.metric_depth else 2
                else:
                    p = 3
                priority[pair] = min(priority.get(pair, 9), p)
                best_rank[pair] = min(best_rank.get(pair, 10 ** 6), место)
                found_by[pair] += 1

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
                     "luchshee_mesto": best_rank.get((qid, cid), 0),
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
    имена = {1: f"ядро, первые {args.metric_depth}", 2: "ядро, хвост",
             3: "остальные связки"}
    say("по приоритету: " + ", ".join(
        f"{k} ({имена.get(k, '?')}) {v}" for k, v in sorted(by_p.items())))
    by_src = collections.Counter(r["istochnik"] for r in rows)
    say("разметка прежде: " + ", ".join(f"{k} {v}" for k, v in by_src.most_common()))
    say(f"новых пар (никем не размечены): {by_src.get('-', 0)} из {len(rows)}")
    per_q = collections.Counter(r["query_id"] for r in rows)
    if per_q:
        say(f"пар на запрос: в среднем {sum(per_q.values()) / len(per_q):.1f}, "
            f"минимум {min(per_q.values())}, максимум {max(per_q.values())}")

    # Охват до и после, то есть сколько позиций десятки размечено. Это число
    # говорит о надёжности метрики больше, чем сама метрика, и считается без
    # всякого судьи.
    say()
    say(f"охват: пар на глубине {args.metric_depth} и на глубине {args.depth}, "
        f"и сколько из них размечено сегодня")
    say(f"{'связка':<22}{'десятка':>9}{'размечено':>11}{'глубина ' + str(args.depth):>13}"
        f"{'размечено':>11}")
    размечено = {p for p, _ in ((k, v) for k, v in priority.items())
                 if source(p) != "-"}
    for cfg in CORE:
        run = runs[cfg]
        мелко = {(q, c) for q in qids for c in run.get(q, [])[:args.metric_depth]}
        глубоко = {(q, c) for q in qids for c in run.get(q, [])[:args.depth]}
        say(f"{cfg:<22}{len(мелко):>9}{len(мелко & размечено):>11}"
            f"{len(глубоко):>13}{len(глубоко & размечено):>11}")
    say(f"неразмеченных позиций первой десятки на запрос у hybrid-rerank: "
        f"{(len({(q, c) for q in qids for c in runs['hybrid-rerank'].get(q, [])[:args.metric_depth]} - размечено)) / len(qids):.2f} "
        f"из {args.metric_depth}")
    say(f"файл: {os.path.relpath(args.out, ROOT)}")

    # Второй файл для Kaggle: там корпус лежит рядом, тексты фрагментов
    # судья берёт из него, и тащить тридцать пять мегабайт в датасет
    # запросов незачем. Текст вопроса стоит в первой строке группы, как
    # и в прежнем листе пула: `gpu_judge.read_pool` раздаёт его группе сам.
    light = os.path.join(os.path.dirname(args.out), "pool_live_kaggle.tsv")
    seen_q: set[str] = set()
    with open(light, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "chunk_id", "prioritet",
                                          "vopros"], delimiter="\t")
        w.writeheader()
        for r in rows:
            first = r["query_id"] not in seen_q
            seen_q.add(r["query_id"])
            w.writerow({"query_id": r["query_id"], "chunk_id": r["chunk_id"],
                        "prioritet": r["prioritet"],
                        "vopros": r["vopros"] if first else ""})
    # Лист Kaggle перемешан внутри приоритета, а не целиком, поэтому первые
    # N строк пробы это случайная выборка из первого приоритета: прогон,
    # прерванный на пробе, размечает именно то, на чём стоит метрика.
    say(f"то же для Kaggle, без текстов фрагментов: "
        f"{os.path.relpath(light, ROOT)} "
        f"({os.path.getsize(light) / 1048576:.1f} МБ)")

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
