#!/usr/bin/env python3
"""Загрузка набора `alekseevpavel04/ru-law-retrieval` для контрольного прогона.

Набор опубликован в формате MTEB и нужен для одной проверки: воспроизводит ли
наш код чужой уровень на чужих данных. Если да, низкие числа на нормативке
Банка России это свойство предметной области, а не ошибка в коде.

Берутся только файлы, которые нужны прогону, а не весь набор. Запросы идут
через вежливый клиент проекта (`rufin.http`): последовательно, с паузой и
внятным User-Agent, как к любому чужому серверу.

Набор чужой и в наш репозиторий не попадает: папка `data/friend/` закрыта
в .gitignore, а в документе печатаются только числа и ссылка на источник.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.http import Client  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(ROOT, "data", "friend")
REPO = "alekseevpavel04/ru-law-retrieval"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

FILES = (
    "README.md",
    "chunks/chunks.jsonl",
    "corpus/corpus.jsonl",
    "corpus/sources.json",
    "queries/test.jsonl",
    "queries/golden.jsonl",
    "queries/tk_hard.jsonl",
    "qrels/test.jsonl",
    "qrels/golden.jsonl",
    "qrels/tk_hard.jsonl",
    "stats/corpus_stats.json",
    "stats/splits.json",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=1.5)
    ap.add_argument("--force", action="store_true", help="перекачать уже скачанное")
    args = ap.parse_args()

    client = Client(pause=args.pause)
    for path in FILES:
        dest = os.path.join(OUTDIR, path)
        if os.path.exists(dest) and not args.force:
            print(f"{path}: уже есть, {os.path.getsize(dest) / 1048576:.1f} МБ")
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        r = client.get(f"{BASE}/{path}")
        # запись через временный файл: обрыв на середине не должен оставить
        # обрезанный файл под правильным именем
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            f.write(r.content)
        os.replace(tmp, dest)
        print(f"{path}: {os.path.getsize(dest) / 1048576:.1f} МБ")

    print(f"\nнабор в {os.path.relpath(OUTDIR, ROOT)}, источник huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
