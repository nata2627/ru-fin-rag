#!/usr/bin/env python3
"""Выбор судьи по согласию с человеком: кандидаты рядом, порог один.

Прежний судья проекта `Qwen2.5-14B-Instruct-AWQ` размечал пул на Kaggle, где
карта шестнадцатигигабайтная. На доступной карте 8 ГБ он не поднимается, и
перед доразметкой пула надо узнать, годится ли кто-нибудь, кто поднимается.
Годность меряется одним числом: каппа Коэна с разметкой человека на той же
выборке 150 пар, по той же инструкции. Порог 0,40 назначен проектом до всех
этих прогонов (`docs/ANNOTATION_GUIDE.md`), и прежний судья взял его едва:
0,408.

Каппа считается на пересечении с выборкой человека, поэтому кандидату
достаточно разметить её одну, а не весь пул.

**Отбор по этой же выборке смещает каппу вверх.** Кандидатов несколько,
выборка одна, и лучший из них выглядит лучше, чем был бы на новых парах.
Поэтому каппа выбранного судьи это верхняя оценка его согласия, и так она
и называется в отчёте. Честно это стоило бы мерить на отложенных парах, но
человек разметил 150, делить их на две половины по 75 бессмысленно:
интервал каппы станет шире самой каппы.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.annotation import cohen_kappa, kappa_verdict  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CANDDIR = os.path.join(QDIR, "judge_candidates")
REPORT = os.path.join(ROOT, "docs", "raw", "judge_candidates.txt")
RESULT = os.path.join(ROOT, "data", "results", "judge-candidates.json")
THRESHOLD = 0.40


def read_marks(path: str) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            mark = (row.get("otsenka_0_1_2") or "").strip()
            if mark in ("0", "1", "2"):
                out[(row["query_id"], row["chunk_id"])] = int(mark)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default=os.path.join(QDIR, "human_sample.tsv"))
    ap.add_argument("--previous", default=os.path.join(QDIR, "judge.tsv"),
                    help="разметка прежнего судьи, для строки отсчёта")
    ap.add_argument("--candidates", default=CANDDIR,
                    help="папка с разметкой кандидатов, имя файла это имя кандидата")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    human = read_marks(args.human)
    say(f"выборка человека: {len(human)} пар, порог каппы {THRESHOLD:.2f}")

    candidates = [("Qwen2.5-14B-Instruct-AWQ (прежний, Kaggle)", args.previous)]
    for name in sorted(os.listdir(args.candidates)):
        if name.endswith(".tsv"):
            candidates.append((name[:-4], os.path.join(args.candidates, name)))

    rows = []
    for name, path in candidates:
        if not os.path.exists(path):
            say(f"{name}: файла нет, пропущен")
            continue
        marks = read_marks(path)
        common = sorted(set(marks) & set(human))
        if len(common) < 100:
            say(f"{name}: общих пар с человеком {len(common)}, "
                f"для каппы мало, пропущен")
            continue
        a = [marks[k] for k in common]
        b = [human[k] for k in common]
        simple = cohen_kappa(a, b)
        weighted = cohen_kappa(a, b, weighted=True)
        same = sum(1 for x, y in zip(a, b, strict=True) if x == y)
        dist = collections.Counter(a)
        rows.append({
            "судья": name, "пар всего": len(marks), "общих с человеком": len(common),
            "совпало": same, "каппа простая": round(simple, 3),
            "каппа взвешенная": round(weighted, 3),
            "вердикт": kappa_verdict(simple),
            "порог взят": bool(simple >= THRESHOLD),
            "оценок 0": dist[0], "оценок 1": dist[1], "оценок 2": dist[2],
        })

    say()
    say(f"{'судья':<44} {'пар':>5} {'каппа':>7} {'взвеш':>7} "
        f"{'совпало':>8} {'0/1/2 судьи':>14}")
    for r in rows:
        say(f"{r['судья']:<44} {r['общих с человеком']:>5} "
            f"{r['каппа простая']:>7.3f} {r['каппа взвешенная']:>7.3f} "
            f"{100 * r['совпало'] / r['общих с человеком']:>7.0f}% "
            f"{str(r['оценок 0']) + '/' + str(r['оценок 1']) + '/' + str(r['оценок 2']):>14}")
    dist_human = collections.Counter(human.values())
    say(f"{'человек':<44} {len(human):>5} {'':>7} {'':>7} {'':>8} "
        f"{str(dist_human[0]) + '/' + str(dist_human[1]) + '/' + str(dist_human[2]):>14}")

    taken = [r for r in rows if r["порог взят"]]
    say()
    if taken:
        best = max(taken, key=lambda r: r["каппа простая"])
        say(f"порог взяли: {', '.join(r['судья'] for r in taken)}")
        say(f"лучший: {best['судья']}, каппа {best['каппа простая']:.3f} "
            f"(верхняя оценка: кандидат выбран по этой же выборке)")
    else:
        say(f"порог {THRESHOLD:.2f} не взял никто. Разметка пула на доступной карте "
            f"не делается: доразметка переносится на карту, где поднимается "
            f"прежний судья.")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump({"порог": THRESHOLD, "выборка человека": len(human),
                   "кандидаты": rows}, f, ensure_ascii=False, indent=2)
    say(f"файлы: {os.path.relpath(REPORT, ROOT)}, {os.path.relpath(RESULT, ROOT)}")


if __name__ == "__main__":
    main()
