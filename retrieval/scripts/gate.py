#!/usr/bin/env python3
"""Ворота по живым вопросам: пускать ли конфигурацию дальше.

Правило и его причины описаны в `rufin.gate`. Коротко: синтетический dev
ранжирует кандидатов, живые вопросы отвечают на единственный вопрос,
не стало ли хуже настоящему пользователю.

Каждое применение дописывается в `docs/raw/gate_log.txt`. Число применений
идёт в отчёт: ворота, использованные три раза, и ворота, использованные
тридцать раз, это разные инструменты, и читатель должен видеть, какой
из них перед ним.

    python3 scripts/gate.py --candidate dense-rufin
    python3 scripts/gate.py --candidate hybrid-rufin --baseline hybrid

Ни модели, ни видеокарты: считается по сохранённым выдачам.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.gate import ДОПУСК, решить  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
QDIR = os.path.join(ROOT, "data", "queries")
LOG = os.path.join(ROOT, "docs", "raw", "gate_log.txt")


def load_run(path: str) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        return {r["query_id"]: r["ranked"] for r in (json.loads(l) for l in f if l.strip())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--candidate", required=True,
                    help="имя выдачи, например dense-rufin или hybrid-rufin")
    ap.add_argument("--baseline", default="dense-e5-small",
                    help="с чем сравнивать; для гибрида обычно hybrid")
    ap.add_argument("--subset", default="живые")
    ap.add_argument("--metric", default="NDCG@10")
    ap.add_argument("--tolerance", type=float, default=ДОПУСК)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="", help="зачем проверяли, идёт в журнал")
    ap.add_argument("--no-log", action="store_true",
                    help="не записывать в журнал; для отладки, не для рецепта")
    args = ap.parse_args()

    qrels = read_qrels(os.path.join(QDIR, "qrels.tsv"))
    subsets = json.load(open(os.path.join(QDIR, "podvyborki.json"), encoding="utf-8"))
    if args.subset not in subsets:
        raise SystemExit(f"нет подвыборки «{args.subset}»; есть: {', '.join(subsets)}")

    пути = {}
    for роль, имя in (("кандидат", args.candidate), ("эталон", args.baseline)):
        p = os.path.join(RUNDIR, f"{args.config}__{имя}.jsonl")
        if not os.path.exists(p):
            raise SystemExit(f"нет выдачи {роль}а: {os.path.relpath(p, ROOT)}")
        пути[роль] = p
    кан, эт = load_run(пути["кандидат"]), load_run(пути["эталон"])

    ids = sorted(q for q in subsets[args.subset] if q in qrels and q in кан and q in эт)
    if len(ids) < 30:
        raise SystemExit(f"в подвыборке «{args.subset}» всего {len(ids)} запросов "
                         f"с выдачей у обеих конфигураций; ворота на таком не работают")

    fn = M.METRICS[args.metric]
    a = np.array([fn(кан[q], qrels[q]) for q in ids])
    b = np.array([fn(эт[q], qrels[q]) for q in ids])
    в = решить(a, b, имя=args.candidate, имя_эталона=args.baseline,
               допуск=args.tolerance, seed=args.seed)

    строки = [
        f"ворота по подвыборке «{args.subset}», метрика {args.metric}",
        f"кандидат {args.candidate}, эталон {args.baseline}, запросов {в.запросов}",
        str(в),
        f"набор различает просадку примерно от {в.различимое:.3f}; "
        f"допуск по точечной оценке {args.tolerance:.2f}",
    ]
    if args.note:
        строки.append(f"повод: {args.note}")
    print("\n".join(строки))

    if not args.no_log:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        новый = not os.path.exists(LOG)
        with open(LOG, "a", encoding="utf-8") as f:
            if новый:
                f.write("Журнал применений ворот по живым вопросам.\n"
                        "Правило зафиксировано в src/rufin/gate.py до первого "
                        "применения и после не менялось.\n"
                        "Число строк в этом файле идёт в отчёт: ворота, применённые "
                        "три раза,\nи ворота, применённые тридцать раз, это разные "
                        "инструменты.\n")
            f.write("\n" + dt.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n")
            f.write("\n".join("  " + s for s in строки) + "\n")
        print(f"\nзаписано в {os.path.relpath(LOG, ROOT)}")

    raise SystemExit(0 if в.принят else 1)


if __name__ == "__main__":
    main()
