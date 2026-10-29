#!/usr/bin/env python3
"""Утечка по фрагментам: убрать из обучения вопросы, чей источник лежит в эталоне теста.

Эталон теста это все фрагменты всех позитивных единиц `qrels_punkt.tsv`
(задача 11.1): кусок длинного пункта, соседний с эталонным, тоже эталон. С
задачи 11.5 к нему добавлены позитивы уплотненной разметки живого теста
`qrels_live_core.tsv`.
Убирается обучающий вопрос, тест не трогается.

Чистится основная обучающая выборка `synthetic_train.jsonl`, результат
пишется рядом в `synthetic_train_noleak.jsonl`. Прежние файлы не
переписываются: по ним посчитаны опубликованные прогоны дообучения, и их
числа должны воспроизводиться. Для остальных обучающих файлов утечка только
считается и печатается: какой из них пойдет в будущее обучение, решается
вместе с подходом к нему.

    python3 scripts/drop_leak.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import punkt as P  # noqa: E402
from rufin import split as S  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
RESULT = os.path.join(ROOT, "data", "results", "utechka.json")
REPORT = os.path.join(ROOT, "docs", "raw", "drop_leak.txt")
DOC = os.path.join(ROOT, "docs", "18-nabor-do-i-posle.md")
BLOCK_START = "<!-- tablica: drop_leak.py -->"
BLOCK_END = "<!-- /tablica utechka -->"

# Обучающие файлы прежних прогонов: утечка в них считается, файлы не меняются.
MEASURED = ("synthetic_train_e4.jsonl", "synthetic_train_e4_filtered.jsonl",
            "synthetic_train_e5.jsonl", "synthetic_train_e5_filtered.jsonl", "pairs_e5.jsonl")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_fragments(units: dict[str, dict[str, P.Unit]]) -> set[str]:
    return {c for us in units.values() for u in us.values() if u.score > 0 for c in u.chunks}


def num(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def doc_block(result: dict) -> str:
    t = result["обучение"]
    hit = ", ".join(f"«{k}» {num(v)}" for k, v in result["задетых запросов теста"].items())
    out = [f"Из `{t['файл']}` убраны вопросы, написанные по фрагментам эталона теста, "
           f"{num(t['убрано'])} из {num(t['вопросов'])}, осталось {num(t['осталось'])} "
           f"(`{t['результат']}`). Фрагментов утечки "
           f"{num(len(result['фрагменты утечки']))} из {num(result['фрагментов эталона теста'])} "
           f"фрагментов эталона теста. Запросов теста, чей эталон задет утечкой: {hit}.",
           "", "| Прежний обучающий файл (не менялся) | Записей | С утечкой |", "|---|---|---|"]
    for name, m in result["прежние файлы"].items():
        out.append(f"| `{name}` | {num(m['записей'])} | {num(m['записей с утечкой'])} |")
    if result["негативы"]:
        n = result["негативы"]
        out += ["", f"Негативов подготовки D0 (`train_prepared.jsonl`), лежащих в эталоне теста: "
                    f"{num(n['из эталона теста'])} из {num(n['кандидатов'])}."]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default=os.path.join(QDIR, "qrels_punkt.tsv"))
    ap.add_argument("--train", default=os.path.join(QDIR, "synthetic_train.jsonl"))
    ap.add_argument("--out", default=os.path.join(QDIR, "synthetic_train_noleak.jsonl"))
    ap.add_argument("--live-qrels", default=os.path.join(QDIR, "qrels_live_core.tsv"),
                    help="уплотненная разметка живого теста (задача 11.5): ее фрагменты тоже эталон")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    units = P.read_units(args.qrels)
    test = test_fragments(units)
    live = read_qrels(args.live_qrels) if os.path.exists(args.live_qrels) else {}
    test |= {c for rel in live.values() for c, s in rel.items() if s > 0}
    train = read_jsonl(args.train)
    kept, dropped = S.drop_leaked(train, test)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    leaked = sorted({c for row in dropped for c in S.sources_of(row) & test})

    queries = {}
    with open(os.path.join(QDIR, "queries.jsonl"), encoding="utf-8") as f:
        for q in map(json.loads, f):
            queries[q["query_id"]] = q
    by_subset: collections.Counter = collections.Counter()
    leaked_set = set(leaked)
    for qid in set(units) | set(live):
        gold = {c for u in units.get(qid, {}).values() if u.score > 0 for c in u.chunks}
        gold |= {c for c, s in live.get(qid, {}).items() if s > 0}
        if gold & leaked_set:
            by_subset[queries[qid]["podvyborka"]] += 1

    say(f"фрагментов эталона теста (единицы qrels_punkt и уплотненный живой тест): {len(test)}")
    say(f"{os.path.basename(args.train)}: вопросов {len(train)}, убрано {len(dropped)}, "
        f"осталось {len(kept)} -> {os.path.relpath(args.out, ROOT)}")
    say(f"фрагментов утечки: {len(leaked)}")
    say(f"запросов теста, задетых утечкой, по подвыборкам: {dict(by_subset)}")
    say()
    say("утечка в прежних обучающих файлах (файлы не менялись):")
    measured = {}
    for name in MEASURED:
        rows = read_jsonl(os.path.join(QDIR, name))
        hits = S.fragment_leaks(rows, test)
        rows_hit = len({q for q, _ in hits})
        measured[name] = {"записей": len(rows), "записей с утечкой": rows_hit}
        say(f"   {name:36s} записей {len(rows):>6}, с утечкой {rows_hit:>4}")

    prepared = os.path.join(QDIR, "train_prepared.jsonl")
    negatives = None
    if os.path.exists(prepared):
        rows = read_jsonl(prepared)
        # Кандидаты это пары [фрагмент, оценка учителя], первым стоит сам эталон.
        neg = [c[0] for r in rows for c in (r.get("candidates") or []) if c[0] != r["gold_chunk_id"]]
        negatives = {"кандидатов": len(neg), "из эталона теста": sum(1 for c in neg if c in test)}
        say(f"   негативы train_prepared.jsonl: {negatives['из эталона теста']} из "
            f"{negatives['кандидатов']} кандидатов лежат в эталоне теста")

    result = {"фрагментов эталона теста": len(test),
              "обучение": {"файл": os.path.relpath(args.train, ROOT), "вопросов": len(train),
                           "убрано": len(dropped), "осталось": len(kept),
                           "результат": os.path.relpath(args.out, ROOT)},
              "фрагменты утечки": leaked, "задетых запросов теста": dict(sorted(by_subset.items())),
              "прежние файлы": measured, "негативы": negatives}
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    body = open(DOC, encoding="utf-8").read()
    pattern = re.compile(re.escape(BLOCK_START) + ".*?" + re.escape(BLOCK_END), re.S)
    if not pattern.search(body):
        raise SystemExit(f"в {DOC} нет меток {BLOCK_START} ... {BLOCK_END}")
    with open(DOC, "w", encoding="utf-8") as f:
        f.write(pattern.sub(lambda _: f"{BLOCK_START}\n{doc_block(result)}\n{BLOCK_END}", body))


if __name__ == "__main__":
    main()
