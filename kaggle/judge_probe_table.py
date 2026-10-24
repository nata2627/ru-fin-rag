#!/usr/bin/env python3
"""Сводка пробы судей: скорость, разбор, распределение, прогноз на весь пул.

Числа не пересчитываются и не правятся руками: берутся из метаданных прогонов
(`*_meta.json`, их пишет `gpu_judge.judge`) и из выводов `scripts/judge_vote.py`.
Машина у каждого прогона своя, и она печатается рядом со скоростью: 0,355 пары/с
на двух T4 и 4,129 на RTX 3070 это числа про разное железо, и складывать их
нельзя.

Прогноз на весь пул считается по замеренной скорости того же прогона, а не по
скорости другой модели: именно на этом предсказание S17 и разошлось с фактом.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POOL_FULL = 40134
POOL_CORE10 = 4134


def load(path: str) -> dict:
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def card_of(meta: dict) -> str:
    cards = meta.get("стек вывода", {}).get("карты", [])
    if not cards:
        return "неизвестно"
    return cards[0] if len(cards) == 1 else f"{cards[0]} x{len(cards)}"


def row(name: str, meta_path: str) -> dict:
    m = load(meta_path)
    d = m["распределение"]
    graded = m["пар размечено"]
    task = m["пар в задании"]
    rate = m["пар в секунду"]
    return {
        "судья": name,
        "модель": m["модель"],
        "машина": card_of(m),
        "пар/с": rate,
        "секунд на 150 пар": m["секунд"],
        "не разобрано": d.get("ответ не разобран", 0),
        "из": task,
        "оценка 0": d.get("оценка 0", 0),
        "оценка 1": d.get("оценка 1", 0),
        "оценка 2": d.get("оценка 2", 0),
        "размечено": graded,
        "тип общая норма": d.get("тип общая", 0),
        "тип часть ответа": d.get("тип часть", 0),
        "тип иное": d.get("тип иное", 0),
        "весь пул 40134, часов": round(POOL_FULL / rate / 3600, 2) if rate else None,
        "ядро первой десятки 4134, часов":
            round(POOL_CORE10 / rate / 3600, 2) if rate else None,
        "движок": m["стек вывода"].get("vllm"),
        "отпечаток инструкции": m["стек вывода"].get("рубрика sha256"),
        "файл": meta_path,
    }


def pair(name: str, result_path: str) -> dict:
    v = load(result_path)
    both = v["размечено обоими"]
    return {
        "сравнение": name,
        "A": v["судья A"]["имя"],
        "B": v["судья B"]["имя"],
        "размечено обоими": both,
        "совпало": v["совпало"],
        "разошлось": v["разошлось"],
        "доля несогласий": round(v["разошлось"] / both, 3) if both else None,
        "разошлось на две ступени": v["разошлось на две ступени"],
        "размечено только A": v["размечено только A"],
        "размечено только B": v["размечено только B"],
        "пометка типа: оба поставили": v["пометка типа: оба поставили"],
        "пометка типа: совпала": v["пометка типа: совпала"],
        "файл": result_path,
    }


JUDGES = [
    ("Qwen2.5-14B-Instruct-AWQ, прогон 1",
     "data/kaggle_out/ru-fin-judge-proba-14b/queries/judge_live_meta.json"),
    ("Qwen2.5-14B-Instruct-AWQ, повтор",
     "data/kaggle_out/ru-fin-judge-14b-repeat/queries/judge_live_repeat_meta.json"),
    ("Qwen2.5-32B-Instruct-AWQ, разбор починен",
     "data/kaggle_out/ru-fin-judge-32b-fix/queries/judge_live_32b_meta.json"),
    ("Qwen2.5-7B-Instruct-AWQ",
     "data/kaggle_out/ru-fin-judge-7b/queries/judge_live_7b_meta.json"),
    ("Qwen3-4B-AWQ, прогон 1",
     "data/kaggle_out/local-3070/judge_local_4b_meta.json"),
    ("Qwen3-4B-AWQ, повтор",
     "data/kaggle_out/local-3070/judge_local_4b_repeat_meta.json"),
]

PAIRS = [
    ("14B против 32B", "data/results/judge_vote_14b_32b.json"),
    ("4B против 14B", "data/results/judge_vote_4b_14b.json"),
    ("4B против 32B", "data/results/judge_vote_4b_32b.json"),
    ("7B против 14B", "data/results/judge_vote_7b_14b.json"),
    ("7B против 32B", "data/results/judge_vote_7b_32b.json"),
    ("14B против себя (повтор)", "data/results/judge_vote_14b_self.json"),
    ("4B против себя (повтор)", "data/results/judge_vote_4b_self.json"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="data/results/judge_probe_table.json")
    ap.add_argument("--report", default="docs/raw/judge_probe_table.txt")
    args = ap.parse_args()

    judges = [row(n, p) for n, p in JUDGES]
    pairs = [pair(n, p) for n, p in PAIRS]
    out = {"лист": "data/queries/pool_probe_kaggle.tsv", "пар в листе": 150,
           "пул глубины 50": POOL_FULL, "ядро первой десятки": POOL_CORE10,
           "судьи": judges, "сравнения": pairs}

    with open(os.path.join(ROOT, args.result), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lines = ["проба судей по листу data/queries/pool_probe_kaggle.tsv, 150 пар",
             f"пул глубины 50: {POOL_FULL} пар, ядро первой десятки: {POOL_CORE10} пар",
             "",
             f"{'судья':<38} {'машина':<16} {'пар/с':>7} {'не разобр':>10} "
             f"{'0/1/2':>12} {'пул, ч':>8} {'ядро, ч':>8}"]
    for r in judges:
        unparsed = f"{r['не разобрано']} из {r['из']}"
        spread = f"{r['оценка 0']}/{r['оценка 1']}/{r['оценка 2']}"
        lines.append(f"{r['судья']:<38} {r['машина']:<16} {r['пар/с']:>7.3f} "
                     f"{unparsed:>10} {spread:>12} "
                     f"{r['весь пул 40134, часов']:>8.2f} "
                     f"{r['ядро первой десятки 4134, часов']:>8.2f}")
    lines += ["", "пометка типа (ставится только при оценке 1):"]
    for r in judges:
        lines.append(f"   {r['судья']:<38} общая норма {r['тип общая норма']:>3}, "
                     f"часть ответа {r['тип часть ответа']:>3}, "
                     f"иное {r['тип иное']:>3}, всего единиц {r['оценка 1']:>3}")
    lines += ["", "несогласия, знаменатель это пары, размеченные обоими:"]
    for p in pairs:
        lines.append(f"   {p['сравнение']:<28} разошлось "
                     f"{p['разошлось']} из {p['размечено обоими']} "
                     f"({p['доля несогласий'] * 100:.1f}%), на две ступени "
                     f"{p['разошлось на две ступени']}")
    lines += ["", "файлы с числами:"]
    for r in judges:
        lines.append(f"   {r['файл']}")
    for p in pairs:
        lines.append(f"   {p['файл']}")

    with open(os.path.join(ROOT, args.report), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
