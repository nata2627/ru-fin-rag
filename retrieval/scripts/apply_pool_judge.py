#!/usr/bin/env python3
"""Расширение эталона живых вопросов разметкой пула.

Отдельно от `apply_judge.py`: тот собирает эталон с нуля для запросов, у
которых его нет, и выбрасывает запрос, где ни одному кандидату не поставлена
двойка. Здесь эталон уже есть, и вопрос в его полноте. Поэтому правила другие.

**Прежний эталон не удаляется.** Он собран по названным Банком России пунктам
и прежним пулом, и то, что новый судья ставит там ноль, означает разногласие
двух измерений, а не отсутствие релевантности. Такие пары считаются и
печатаются числом: это проверка судьи, а не повод править эталон.

**Новая оценка добавляется, если она больше нуля.** Фрагмент, который хотя бы
одна конфигурация ставит в первую десятку и который судья признал относящимся
к делу, обязан быть в эталоне: иначе метрика читает его как промах.

**Разметка по пунктам сильнее разметки судьи.** Где они спорят, остаётся
первая.

На выходе два файла эталона, старый и расширенный, с одинаковым охватом
запросов: поправка меряется на одних и тех же 142 вопросах.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.benchmark import read_qrels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "apply_pool_judge.txt")


def read_marks(path: str) -> dict[str, dict[str, int]]:
    marks: dict[str, dict[str, int]] = collections.defaultdict(dict)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            mark = (row.get("otsenka_0_1_2") or "").strip()
            if mark in ("0", "1", "2"):
                marks[row["query_id"]][row["chunk_id"]] = int(mark)
    return marks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=os.path.join(QDIR, "judge_live.tsv"),
                    help="разметка пула новым судьёй")
    ap.add_argument("--old-judge", default=os.path.join(QDIR, "judge.tsv"),
                    help="разметка прежнего судьи, для сверки на общих парах")
    ap.add_argument("--qrels", default=os.path.join(QDIR, "qrels.tsv"))
    ap.add_argument("--by-clause", default=os.path.join(QDIR, "qrels_by_clause.tsv"))
    ap.add_argument("--subset", default="живые")
    ap.add_argument("--out", default=os.path.join(QDIR, "qrels_pool_full.tsv"))
    ap.add_argument("--model", default="unsloth/Qwen3-8B-bnb-4bit",
                    help="имя судьи, идёт в отчёт рядом с числами")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    subsets = json.load(open(os.path.join(QDIR, "podvyborki.json"), encoding="utf-8"))
    old = read_qrels(args.qrels)
    by_clause = read_qrels(args.by_clause)
    marks = read_marks(args.judge)
    qids = sorted(q for q in subsets[args.subset] if q in old)

    say(f"судья: {args.model}")
    say(f"подвыборка «{args.subset}»: запросов с эталоном {len(qids)}")
    say(f"пар в новой разметке: {sum(len(v) for v in marks.values())} "
        f"по {len(marks)} запросам")

    # Сверка с прежним судьёй на общих парах: это единственная дешёвая
    # проверка того, что смена модели не сдвинула разметку целиком.
    if os.path.exists(args.old_judge):
        prev = read_marks(args.old_judge)
        common = [(q, c) for q in marks for c in marks[q] if c in prev.get(q, {})]
        if common:
            same = sum(1 for q, c in common if marks[q][c] == prev[q][c])
            rough = sum(1 for q, c in common if abs(marks[q][c] - prev[q][c]) == 2)
            say(f"общих пар с прежним судьёй: {len(common)}, совпало дословно {same} "
                f"({100 * same / len(common):.0f}%), грубых расхождений {rough}")

    # Спор судьи с разметкой по пунктам: пункт назван Банком России,
    # это внешнее свидетельство, и оно остаётся.
    спор = совпало = 0
    for qid, rel in by_clause.items():
        for cid, score in rel.items():
            if cid in marks.get(qid, {}):
                if marks[qid][cid] == score:
                    совпало += 1
                else:
                    спор += 1
    if совпало or спор:
        say(f"пар, где пункт назван и судья высказался: {совпало + спор}; "
            f"совпало {совпало}, разошлось {спор} "
            f"({100 * спор / max(1, совпало + спор):.0f}%)")

    # Спор судьи с прежним эталоном: судья ставит ноль там, где эталон
    # говорит «релевантно».
    нули_на_эталоне = sum(1 for qid in qids for cid, score in old[qid].items()
                          if score > 0 and marks.get(qid, {}).get(cid) == 0)
    всего_эталонов = sum(1 for qid in qids for score in old[qid].values() if score > 0)
    say(f"прежних эталонов, которым новый судья поставил 0: {нули_на_эталоне} "
        f"из {всего_эталонов}; эталон сохранён, это разногласие двух измерений")

    full: dict[str, dict[str, int]] = {}
    added = collections.Counter()
    for qid in qids:
        rel = dict(old[qid])
        for cid, score in marks.get(qid, {}).items():
            if score > 0 and cid not in rel:
                rel[cid] = score
                added[qid] += 1
        for cid, score in by_clause.get(qid, {}).items():
            rel[cid] = score
        full[qid] = rel

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid in sorted(full):
            for cid, score in sorted(full[qid].items()):
                f.write(f"{qid}\t{cid}\t{score}\n")

    before = sum(len([1 for s in old[q].values() if s > 0]) for q in qids)
    after = sum(len(full[q]) for q in qids)
    say()
    say(f"эталонов было {before}, стало {after}, добавлено {after - before}")
    say(f"вопросов с новым эталоном: {len(added)} из {len(qids)} "
        f"({100 * len(added) / len(qids):.0f}%)")
    say(f"эталонов на вопрос: было {before / len(qids):.1f}, "
        f"стало {after / len(qids):.1f}")
    двоек = sum(1 for q in added for cid, s in full[q].items()
                if s == 2 and cid not in old[q])
    единиц = sum(1 for q in added for cid, s in full[q].items()
                 if s == 1 and cid not in old[q])
    say(f"добавлено с оценкой 2: {двоек}, с оценкой 1: {единиц}")
    say(f"файл: {os.path.relpath(args.out, ROOT)}")
    say("дальше: scripts/pool_effect.py — поправка к метрикам")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
