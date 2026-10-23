#!/usr/bin/env python3
"""Таблица кандидатов этапа 3б по подвыборкам, с интервалами и знаменателями.

Зачем отдельно от `metrics_report.py`. Тот считает всю таблицу разом: берёт
все выдачи нарезки, сводит их по общим запросам и переписывает
`data/results/base.json` с `docs/raw/metrics_base.txt`. Это правильно для
итогового отчёта и неправильно здесь: кандидаты этапа 3б не публикуются,
пока их не пропустили ворота, а опубликованная таблица в это время живёт
своей жизнью и переписывать её нечем.

Поэтому здесь считаются ровно названные выдачи, ровно по названным
подвыборкам, и пишется свой файл. Метрика та же, из `rufin.metrics`:
второй реализации в проекте быть не должно.

    python3 scripts/etap3b_table.py --candidates dense-rufin-a-1ep dense-rufin-c-oba
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
QDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "etap3b_kandidaty.txt")
RESULT = os.path.join(ROOT, "data", "results", "etap3b.json")


def load_run(config: str, имя: str) -> dict[str, list[str]]:
    путь = os.path.join(RUNDIR, f"{config}__{имя}.jsonl")
    if not os.path.exists(путь):
        raise SystemExit(f"нет выдачи {os.path.relpath(путь, ROOT)}")
    with open(путь, encoding="utf-8") as f:
        return {r["query_id"]: r["ranked"]
                for r in (json.loads(line) for line in f if line.strip())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--baseline", default="dense-e5-small",
                    help="нулевая точка: необученный ученик")
    ap.add_argument("--extra", nargs="*", default=["bm25", "hybrid", "dense-rufin"],
                    help="что ещё показать в таблице без парной разницы")
    ap.add_argument("--metric", default="NDCG@10")
    ap.add_argument("--qrels", default=os.path.join(QDIR, "qrels.tsv"))
    ap.add_argument("--subsets", default=os.path.join(QDIR, "podvyborki.json"))
    ap.add_argument("--only", nargs="*", default=["все", "заголовочная", "живые"])
    ap.add_argument("--out", default=REPORT)
    ap.add_argument("--json", default=RESULT)
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    qrels = read_qrels(args.qrels)
    подвыборки = json.load(open(args.subsets, encoding="utf-8"))
    выдачи = {имя: load_run(args.config, имя)
              for имя in [args.baseline] + list(args.candidates) + list(args.extra)}

    # Знаменатель один на всю таблицу: метрика считается по запросам, которые
    # есть у всех сравниваемых конфигураций. Разные знаменатели сделали бы
    # парную разницу бессмысленной.
    общие = set(qrels)
    for run in выдачи.values():
        общие &= set(run)
    say(f"нарезка {args.config}, разметка {os.path.relpath(args.qrels, ROOT)}, "
        f"метрика {args.metric}")
    say(f"конфигураций {len(выдачи)}, запросов с разметкой и выдачей у всех: "
        f"{len(общие)}")
    say(f"нулевая точка сравнения: {args.baseline}")
    say()

    fn = M.METRICS[args.metric]
    итог: dict = {"метрика": args.metric, "нулевая точка": args.baseline,
                  "подвыборки": {}}

    for имя_под in args.only:
        ids = sorted(общие if имя_под == "все"
                     else {q for q in подвыборки.get(имя_под, []) if q in общие})
        if not ids:
            say(f"подвыборка «{имя_под}»: ни одного запроса, пропуск")
            continue
        say(f"=== подвыборка «{имя_под}», запросов {len(ids)} ===")
        say(f"{'конфигурация':<24} {args.metric:<24} разница с "
            f"{args.baseline}")
        значения = {имя: np.array([fn(выдачи[имя][q], qrels[q]) for q in ids])
                    for имя in выдачи}
        строки: dict = {}
        for имя in [args.baseline] + list(args.candidates) + list(args.extra):
            v = значения[имя]
            ci = M.bootstrap_ci(v)
            клетка = f"{ci.mean:.3f} [{ci.lo:.3f}; {ci.hi:.3f}]"
            запись = {"mean": round(float(ci.mean), 4), "lo": round(float(ci.lo), 4),
                      "hi": round(float(ci.hi), 4), "запросов": len(ids)}
            if имя in args.candidates:
                d, p = M.paired_diff_ci(v, значения[args.baseline])
                разница = (f"{d.mean:+.3f} [{d.lo:+.3f}; {d.hi:+.3f}] p={p:.3f}"
                           + ("  установлена" if d.lo > 0 or d.hi < 0 else ""))
                запись["разница"] = {"mean": round(float(d.mean), 4),
                                     "lo": round(float(d.lo), 4),
                                     "hi": round(float(d.hi), 4),
                                     "p": round(float(p), 4)}
            else:
                разница = ""
            say(f"{имя:<24} {клетка:<24} {разница}")
            строки[имя] = запись
        say()
        итог["подвыборки"][имя_под] = {"запросов": len(ids), "строки": строки}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(итог, f, ensure_ascii=False, indent=1)
    print(f"{os.path.relpath(args.out, ROOT)} и {os.path.relpath(args.json, ROOT)}")


if __name__ == "__main__":
    main()
