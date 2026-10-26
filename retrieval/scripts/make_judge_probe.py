#!/usr/bin/env python3
"""Лист пробы для судьи: ровно те 150 пар, которые размечает человек.

Проба судьи на Kaggle (задача 9.6.7) должна дать четыре числа: скорость,
прогноз на весь пул, согласие двух прогонов одной модели и долю несогласий
двух судей. Три последних считаются только на тех же парах, на которых стоит
лист человека, иначе каппу не с чем сравнивать.

Ключ `--judge-limit 150` для этого не годится: он берёт первые 150 пар пула,
а в стратифицированный лист человека из них попадают 4. Поэтому лист пробы
собирается здесь: пары берутся из листа человека, а текст вопроса и приоритет
из пула, то есть судья видит ровно то, что видел бы на полном прогоне.

Запуск: python3 scripts/make_judge_probe.py
"""
from __future__ import annotations

import argparse
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
FIELDS = ["query_id", "chunk_id", "prioritet", "vopros"]


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def questions(rows: list[dict]) -> dict[str, str]:
    """Текст вопроса по каждому вопросу пула.

    В пуле он стоит только в первой строке группы, остальные пустые: так лист
    читает человек, а `gpu_judge.read_pool` раздаёт текст вниз по файлу. Лист
    пробы идёт в другом порядке, поэтому раздача вниз подставила бы соседний
    вопрос, и судья отвечал бы не на тот. Здесь текст заполняется в каждой
    строке, и порядок перестаёт на что-либо влиять.
    """
    out: dict[str, str] = {}
    for row in rows:
        if row.get("vopros"):
            out.setdefault(row["query_id"], row["vopros"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default=os.path.join(QDIR, "human_sample_new.tsv"))
    ap.add_argument("--pool", default=os.path.join(QDIR, "pool_live_kaggle.tsv"))
    ap.add_argument("--out", default=os.path.join(QDIR, "pool_probe_kaggle.tsv"))
    args = ap.parse_args()

    human = read_rows(args.human)
    pool = {(r["query_id"], r["chunk_id"]): r for r in read_rows(args.pool)}

    # Пары листа, которых нет в пуле, это не повод тихо их пропустить: лист
    # собран из пула, и расхождение означает, что один из файлов пересобран
    # отдельно от другого. Тогда каппа считалась бы по неполному пересечению,
    # а знаменатель в отчёте стоял бы прежний.
    missing = [key for key in ((r["query_id"], r["chunk_id"]) for r in human) if key not in pool]
    if missing:
        raise SystemExit(
            f"в пуле нет {len(missing)} пар из листа человека, первая {missing[0]}. "
            f"Пересоберите оба файла: make pool-live, затем make sample-for-human"
        )

    rows = read_rows(args.pool)
    text = questions(rows)
    silent = sorted({r["query_id"] for r in human} - set(text))
    if silent:
        raise SystemExit(
            f"в пуле нет текста {len(silent)} вопросов листа, первый {silent[0]}: "
            f"судья получил бы пустой запрос"
        )

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        for row in human:
            source = pool[(row["query_id"], row["chunk_id"])]
            w.writerow(
                {
                    "query_id": row["query_id"],
                    "chunk_id": row["chunk_id"],
                    "prioritet": source["prioritet"],
                    "vopros": text[row["query_id"]],
                }
            )

    queries = len({r["query_id"] for r in human})
    print(
        f"лист пробы: {len(human)} пар, вопросов {queries}, "
        f"файл {os.path.relpath(args.out, ROOT)}"
    )


if __name__ == "__main__":
    main()
