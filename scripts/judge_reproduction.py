#!/usr/bin/env python3
"""Воспроизводим ли судья: та же модель, те же пары, другой стек вывода.

Согласие судьи с человеком (каппа 0,408) измерено 01.10 на 150 парах
стратифицированной выборки. 23.10 те же 150 пар размечены той же моделью
`Qwen2.5-14B-Instruct-AWQ`, но на другом стеке: свежий vLLM, образ Kaggle
на питоне 3.13, веса не из хаба, а из Kaggle Models, и включённый кэш
общего префикса.

Каппа при этом стала 0,307, то есть порог 0,40, записанный до прогона,
не взят. Прежде чем называть это свойством судьи, надо посмотреть, чем
два прогона отличаются между собой: сравнение двух разметок одной модели
не требует ни карты, ни человека и считается здесь.

Печатается и уходит в JSON три вещи:

1. согласие двух прогонов между собой;
2. направление расхождения: сколько пар понижено и сколько повышено.
   Односторонний сдвиг и случайный шум означают разные причины;
3. длина фрагмента у расходящихся пар против остальных. Это проверка
   догадки про обрезку подсказки: будь прогон 23.10 обрезан по длине,
   понижались бы длинные фрагменты.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.annotation import RUBRIC, cohen_kappa  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "judge_reproduction.txt")
RESULT = os.path.join(ROOT, "data", "results", "judge-reproduction.json")


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
    ap.add_argument("--new", default=os.path.join(QDIR, "judge_live_proba150.tsv"),
                    help="разметка прогона 23.10")
    ap.add_argument("--old", default=os.path.join(QDIR, "judge.tsv"),
                    help="разметка прогона 01.10")
    ap.add_argument("--human", default=os.path.join(QDIR, "human_sample.tsv"))
    ap.add_argument("--chunks", default=os.path.join(ROOT, "data", "chunks",
                                                    "base.jsonl"))
    ap.add_argument("--out", default=REPORT)
    ap.add_argument("--json", default=RESULT)
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    новый = read_marks(args.new)
    прежний = read_marks(args.old)
    человек = read_marks(args.human)
    общие = sorted(set(новый) & set(прежний) & set(человек))
    if not общие:
        raise SystemExit("нет пар, размеченных всеми тремя")

    итог: dict = {"пар": len(общие)}
    say(f"пар, размеченных обоими прогонами и человеком: {len(общие)}")

    for имя, a in (("прогон 01.10", прежний), ("прогон 23.10", новый)):
        к = cohen_kappa([a[p] for p in общие], [человек[p] for p in общие])
        взв = cohen_kappa([a[p] for p in общие], [человек[p] for p in общие],
                          weighted=True)
        расп = collections.Counter(a[p] for p in общие)
        say(f"{имя}: каппа с человеком {к:.3f}, взвешенная {взв:.3f}, "
            f"оценки 0/1/2 {расп[0]}/{расп[1]}/{расп[2]}")
        итог[имя] = {"каппа": round(к, 4), "взвешенная": round(взв, 4),
                     "оценки": {str(k): расп[k] for k in (0, 1, 2)}}
    расп_ч = collections.Counter(человек[p] for p in общие)
    say(f"человек: оценки 0/1/2 {расп_ч[0]}/{расп_ч[1]}/{расп_ч[2]}")
    итог["человек"] = {"оценки": {str(k): расп_ч[k] for k in (0, 1, 2)}}

    между = cohen_kappa([новый[p] for p in общие], [прежний[p] for p in общие])
    между_взв = cohen_kappa([новый[p] for p in общие],
                            [прежний[p] for p in общие], weighted=True)
    совпало = sum(1 for p in общие if новый[p] == прежний[p])
    понижено = [p for p in общие if новый[p] < прежний[p]]
    повышено = [p for p in общие if новый[p] > прежний[p]]
    say()
    say(f"два прогона одной модели между собой: каппа {между:.3f}, "
        f"взвешенная {между_взв:.3f}, дословно совпало {совпало} из {len(общие)}")
    say(f"понижено прогоном 23.10: {len(понижено)} из {len(общие)}; "
        f"повышено: {len(повышено)}")
    итог["между прогонами"] = {
        "каппа": round(между, 4), "взвешенная": round(между_взв, 4),
        "совпало": совпало, "понижено": len(понижено), "повышено": len(повышено)}

    переходы = collections.Counter((прежний[p], новый[p]) for p in общие
                                  if новый[p] != прежний[p])
    say("переходы (было -> стало): " + ", ".join(
        f"{б} -> {н}: {n}" for (б, н), n in sorted(переходы.items())))
    итог["переходы"] = {f"{б}->{н}": n for (б, н), n in sorted(переходы.items())}

    # Проверка догадки про обрезку подсказки. Длина фрагмента берётся
    # в знаках, а не в токенах: токенизатор судьи локально не поднять,
    # а для ответа «обрезка или нет» порядок величины и так виден.
    тексты: dict[str, str] = {}
    нужны = {c for _, c in общие}
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            if c["chunk_id"] in нужны:
                тексты[c["chunk_id"]] = c["text"]
    say()
    say(f"инструкция судье: {len(RUBRIC)} знаков; "
        f"фрагментов найдено {len(тексты)} из {len(нужны)}")
    итог["длина"] = {"инструкция, знаков": len(RUBRIC)}
    for имя, пары in (("понижено", понижено),
                      ("остальные", [p for p in общие if p not in set(понижено)])):
        длины = [len(тексты[c]) for _, c in пары if c in тексты]
        if not длины:
            continue
        say(f"   {имя:<10} пар {len(длины):>3}, фрагмент в знаках: медиана "
            f"{statistics.median(длины):.0f}, среднее {statistics.mean(длины):.0f}, "
            f"максимум {max(длины)}")
        итог["длина"][имя] = {
            "пар": len(длины), "медиана": round(statistics.median(длины)),
            "среднее": round(statistics.mean(длины)), "максимум": max(длины)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(итог, f, ensure_ascii=False, indent=1)
    print(f"\n{os.path.relpath(args.out, ROOT)} и "
          f"{os.path.relpath(args.json, ROOT)}")


if __name__ == "__main__":
    main()
