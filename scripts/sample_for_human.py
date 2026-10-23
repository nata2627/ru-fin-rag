#!/usr/bin/env python3
"""Лист пар для ручной разметки: им мерится погрешность судьи.

Судья это прибор, и у прибора есть погрешность. Мерится она согласием
с человеком на выборке (`scripts/kappa.py`), и человек отвечает на те же
два вопроса, на которые отвечает судья: оценку 0/1/2 ни тот, ни другой
не ставит, она собирается правилом из инструкции.

Выборка стратифицирована по собранной оценке судьи. Случайная состояла бы
почти из одних нулей: в пуле на двести восемьдесят кандидатов релевантны
единицы, и каппа померила бы согласие в том, что случайный фрагмент
нерелевантен. Доли берутся равными, а не пропорциональными, иначе двоек
в выборке окажется две-три.

Оценок судьи в листе нет. Разметка, сделанная после подглядывания, мерит
внушаемость, а не согласие.

Фрагмент идёт целиком. В прежнем листе он был обрезан до трёхсот знаков ради
обозримости, а судья видел полный текст, то есть человека и судью спрашивали
о разном. Длинный текст в клетке таблицы читать неудобно, поэтому рядом
пишется та же выборка в читаемом виде, тем же порядком и с теми же номерами:
читать в ней, отвечать в таблице.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.annotation import RUBRIC, stratified_sample  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
REPORT = os.path.join(ROOT, "docs", "raw", "sample_for_human.txt")

FIELDS = ["nomer", "otvet1_celikom", "otvet2_vmeste", "tip",
          "zapros", "akt", "razdel", "punkty", "fragment", "query_id", "chunk_id"]

HEAD = """\
Лист ручной разметки. Инструкция целиком ниже, она же лежит в
docs/ANNOTATION_GUIDE.md и data/queries/rubric.txt. Инструкция одна на
человека и на модель: если размечать по другой, согласие измерит разницу
инструкций, а не согласие.

Заполняются три колонки таблицы {table}:
  otvet1_celikom   да или нет
  otvet2_vmeste    да или нет; при «да» в первой колонке ставится «-»
  tip              общая, часть или иное; при «да» в первой колонке
                   или «нет» во второй ставится «-»

Оценка 0/1/2 не ставится: она собирается из ответов кодом.
Оценок модели в листе нет намеренно.

================================ ИНСТРУКЦИЯ ================================

{rubric}
============================== КОНЕЦ ИНСТРУКЦИИ =============================
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=os.path.join(QDIR, "judge_live.tsv"),
                    help="разметка судьи: по её оценкам идёт стратификация")
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--size", type=int, default=150,
                    help="пар в выборке. На 100 интервал каппы шире примерно "
                         "на четверть (scripts/kappa.py --sizes)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=os.path.join(QDIR, "human_sample_new.tsv"))
    ap.add_argument("--readable", default=os.path.join(ROOT, "docs", "raw",
                                                      "human_sample_new.txt"))
    args = ap.parse_args()

    if not os.path.exists(args.judge):
        raise SystemExit(f"нет {os.path.relpath(args.judge, ROOT)}: сначала "
                         f"разметка судьёй на Kaggle, по её оценкам идёт "
                         f"стратификация")

    with open(args.judge, encoding="utf-8", newline="") as f:
        pairs = [r for r in csv.DictReader(f, delimiter="\t")
                 if (r.get("otsenka_0_1_2") or "").strip() in ("0", "1", "2")]
    if not pairs:
        raise SystemExit(f"в {os.path.relpath(args.judge, ROOT)} нет ни одной "
                         f"разобранной оценки")
    выбрано = stratified_sample(pairs, "otsenka_0_1_2", args.size, seed=args.seed)

    queries = {}
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries[q["query_id"]] = q["text"]
    нужны = {r["chunk_id"] for r in выбрано}
    chunks = {}
    with open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["chunk_id"] in нужны:
                chunks[c["chunk_id"]] = c

    rows = []
    for n, r in enumerate(sorted(выбрано, key=lambda x: (x["query_id"],
                                                         x["chunk_id"])), 1):
        c = chunks.get(r["chunk_id"], {})
        текст = c.get("text", "")
        rows.append({
            "nomer": n,
            "otvet1_celikom": "",
            "otvet2_vmeste": "",
            "tip": "",
            "zapros": queries.get(r["query_id"], ""),
            "akt": f"{c.get('number', '?')} от {c.get('date', '?')}",
            "razdel": (c.get("section") or "")[:60],
            "punkty": c.get("units", ""),
            # Шапка фрагмента (название акта и раздел) стоит отдельными
            # колонками, поэтому из текста она снимается: иначе человек
            # читает её дважды.
            "fragment": (текст.split("\n\n", 1)[-1] if "\n\n" in текст else текст),
            "query_id": r["query_id"],
            "chunk_id": r["chunk_id"],
        })

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    os.makedirs(os.path.dirname(args.readable), exist_ok=True)
    with open(args.readable, "w", encoding="utf-8") as f:
        f.write(HEAD.format(table=os.path.relpath(args.out, ROOT), rubric=RUBRIC))
        for r in rows:
            f.write(f"\n\n--- пара {r['nomer']} из {len(rows)} "
                    f"------------------------------------\n")
            f.write(f"ВОПРОС: {r['zapros']}\n\n")
            f.write(f"АКТ: {r['akt']}, раздел «{r['razdel']}», "
                    f"пункты {r['punkty'] or 'не выделены'}\n\n")
            f.write(f"ФРАГМЕНТ:\n{r['fragment']}\n\n")
            f.write("ответ 1 (ответ целиком, да/нет): ___\n")
            f.write("ответ 2 (помогает вместе, да/нет): ___\n")
            f.write("тип (общая/часть/иное): ___\n")

    было = collections.Counter(r["otsenka_0_1_2"] for r in pairs)
    стало = collections.Counter(r["otsenka_0_1_2"] for r in выбрано)
    lines = [
        f"пар в разметке судьи: {len(pairs)}, собранные оценки: {dict(было)}",
        f"в выборке: {len(rows)}, оценки судьи в ней: {dict(стало)}",
        f"запросов в выборке: {len({r['query_id'] for r in rows})}",
        "оценок судьи в листе нет: колонки ответов пустые",
        "инструкция: docs/ANNOTATION_GUIDE.md, она же в шапке читаемого листа",
        f"таблица для ответов: {os.path.relpath(args.out, ROOT)}",
        f"читаемый лист: {os.path.relpath(args.readable, ROOT)}",
        "дальше: make kappa",
    ]
    for s in lines:
        print(s)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
