#!/usr/bin/env python3
"""Примеры живых вопросов для генератора обучающей выборки (E4).

E4 проверяет догадку: обучающие вопросы пишет модель, и просадка на вопросах
людей может быть следствием того, что модель пишет не как люди. Проверяется
это так: генератору показываются настоящие вопросы из разъяснений Банка
России, и обучающая выборка собирается заново.

**Что здесь важнее самого файла.** Примеры берутся только из тех живых
вопросов, которых нет в наборе запросов. 1621 вопрос собран с `cbr.ru/explan`,
325 из них попали в набор и на них же считается метрика. Покажи генератору
их, и обучение подстроилось бы под текст, по которому его потом проверяют;
просадка ушла бы, а причиной была бы утечка. Поэтому вопросы набора
исключаются по тексту, а не по идентификатору: в наборе они переписаны не
были, но совпадение по идентификатору сломалось бы при любой пересборке.

Запуск: python3 scripts/make_fewshot_examples.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
SPACE = re.compile(r"\s+")
MIN_CHARS = 80
MAX_CHARS = 700


def key(text: str) -> str:
    return SPACE.sub(" ", text).strip().lower()


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", default=os.path.join(QDIR, "explan_questions.jsonl"))
    ap.add_argument("--benchmark", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--out", default=os.path.join(QDIR, "live_fewshot.jsonl"))
    args = ap.parse_args()

    collected = read_jsonl(args.collected)
    taken = {key(row["text"]) for row in read_jsonl(args.benchmark)}

    kept: list[dict] = []
    seen: set[str] = set()
    counts = {"собрано": len(collected), "в наборе": 0, "повтор": 0, "по длине": 0}
    for row in collected:
        text = SPACE.sub(" ", row["text"]).strip()
        if key(text) in taken:
            counts["в наборе"] += 1
            continue
        if key(text) in seen:
            counts["повтор"] += 1
            continue
        # Слишком короткие и слишком длинные примеры в подсказке работают
        # против задачи: по трем словам манеру не видно, а письмо на полторы
        # тысячи знаков съедает окно и место под сам фрагмент.
        if not MIN_CHARS <= len(text) <= MAX_CHARS:
            counts["по длине"] += 1
            continue
        seen.add(key(text))
        kept.append(
            {
                "text": text,
                "topic": row.get("topic", ""),
                "section": row.get("section", ""),
                "url": row.get("url", ""),
                "chars": len(text),
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = [row["chars"] for row in kept]
    print(
        f"примеров {len(kept)} из {counts['собрано']} собранных: "
        f"исключено как вопросы набора {counts['в наборе']}, повторов {counts['повтор']}, "
        f"по длине {counts['по длине']}"
    )
    print(
        f"длина знаков: медиана {int(statistics.median(lengths))}, "
        f"мин {min(lengths)}, макс {max(lengths)}"
    )
    print(f"файл {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
