#!/usr/bin/env python3
"""Длина вопросов по стилям в двух обучающих выборках: прежней и E4.

Предсказание S18 было записано до прогона и касалось медианы длины по стилям.
На пробе из 11 вопросов стиль «короткий поисковый запрос» тоже вырос, хотя
примеров живых вопросов ему не показывают вовсе. На пробе это шум, и проверить
надо на полном прогоне, иначе объяснение придётся придумывать постфактум.

Длина считается в знаках, как в прежних замерах этого проекта, и рядом
печатается знаменатель: сколько вопросов каждого стиля в выборке.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def stats(path: str) -> dict:
    по_стилю: dict[str, list[int]] = collections.defaultdict(list)
    всего = 0
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            стиль = q.get("style") or q.get("стиль") or "без стиля"
            по_стилю[стиль].append(len(q.get("text") or q.get("vopros") or ""))
            всего += 1
    out = {"файл": path, "вопросов": всего, "стили": {}}
    for стиль, длины in sorted(по_стилю.items()):
        длины.sort()
        out["стили"][стиль] = {
            "вопросов": len(длины),
            "медиана знаков": int(statistics.median(длины)),
            "минимум": длины[0], "максимум": длины[-1],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=[
        "data/queries/synthetic_train.jsonl",
        "data/queries/synthetic_train_e4.jsonl",
        "data/queries/synthetic_train_e4_filtered.jsonl",
    ])
    ap.add_argument("--result", default="data/results/e4_question_stats.json")
    ap.add_argument("--report", default="docs/raw/e4_question_stats.txt")
    args = ap.parse_args()

    собрано = [stats(p) for p in args.files if
               os.path.exists(os.path.join(ROOT, p))]
    with open(os.path.join(ROOT, args.result), "w", encoding="utf-8") as f:
        json.dump({"выборки": собрано}, f, ensure_ascii=False, indent=2)

    L = ["длина вопросов по стилям, в знаках", ""]
    for s in собрано:
        L.append(f"{s['файл']}: вопросов {s['вопросов']}")
        for стиль, d in s["стили"].items():
            L.append(f"   {стиль:<28} медиана {d['медиана знаков']:>4}, "
                     f"от {d['минимум']:>3} до {d['максимум']:>4}, "
                     f"вопросов {d['вопросов']}")
        L.append("")
    with open(os.path.join(ROOT, args.report), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
