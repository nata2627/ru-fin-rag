#!/usr/bin/env python3
"""Входные файлы для счета векторов на чужой машине: имя и текст, сжатые.

Зачем. Плотные половины документов 13 и 14 считаются на видеокарте в чужой
песочнице, где свободно 11 ГБ диска, а нарезки ветвей весят 1,6 ГБ. В счет
векторов идут только имя фрагмента и его текст, поэтому отсюда уезжает
именно это: остальные поля остаются на маке. Сжатие дает примерно трехкратную
экономию на русском тексте и столько же на времени пересылки.

Тексты не переписываются и не чистятся. Файл, уехавший на карту, обязан
совпадать с тем, по которому считается лексическая половина, до символа,
иначе ступени окажутся посчитаны на разных данных.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
FRIEND = os.path.join(ROOT, "data", "friend")
QDIR = os.path.join(ROOT, "data", "queries")

# Ветви стенда подъема к родителю (`scripts/chunk_parent_effect.py`): имя ветви
# и файл нарезки. Имена те же, иначе векторы не склеятся с лексической частью.
BRANCHES = {
    "base": os.path.join(CHUNKDIR, "base.jsonl"),
    "size-256": os.path.join(CHUNKDIR, "size-256.jsonl"),
    "base-local": os.path.join(CHUNKDIR, "local", "base.jsonl"),
    "no-heading": os.path.join(CHUNKDIR, "local", "no-heading.jsonl"),
    "no-section": os.path.join(CHUNKDIR, "local", "no-section.jsonl"),
    "fine-local": os.path.join(CHUNKDIR, "local", "size-256.jsonl"),
    "fine-no-heading": os.path.join(CHUNKDIR, "local", "size-256-no-heading.jsonl"),
}
FRIEND_SPLITS = ("test", "golden", "tk_hard")


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write(out: str, rows, id_field: str) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    count = 0
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for key, text in rows:
            f.write(json.dumps({id_field: key, "text": text}, ensure_ascii=False) + "\n")
            count += 1
    print(f"{os.path.relpath(out, ROOT)}: {count} записей, "
          f"{os.path.getsize(out) / 1048576:.0f} МБ", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="+", required=True,
                    choices=["branches", "queries", "friend"])
    ap.add_argument("--out", required=True, help="папка под jsonl.gz")
    ap.add_argument("--only", nargs="*", help="для branches: только эти ветви")
    args = ap.parse_args()

    if "branches" in args.what:
        for name, path in BRANCHES.items():
            if args.only and name not in args.only:
                continue
            write(os.path.join(args.out, f"chunks-{name}.jsonl.gz"),
                  ((r["chunk_id"], r["text"]) for r in read_jsonl(path)), "chunk_id")

    if "queries" in args.what:
        write(os.path.join(args.out, "queries-ours.jsonl.gz"),
              ((r["query_id"], r["text"])
               for r in read_jsonl(os.path.join(QDIR, "queries.jsonl"))), "query_id")

    if "friend" in args.what:
        # Протокол chunk у него индексирует фрагменты, протокол article статьи
        # целиком с названием. Склейка названия и текста повторяет лексическую
        # половину (`scripts/friend_control.py`) дословно.
        write(os.path.join(args.out, "friend-chunks.jsonl.gz"),
              ((r["_id"], r["text"])
               for r in read_jsonl(os.path.join(FRIEND, "chunks", "chunks.jsonl"))),
              "chunk_id")
        write(os.path.join(args.out, "friend-articles.jsonl.gz"),
              ((r["_id"], f"{r['title']}\n\n{r['text']}")
               for r in read_jsonl(os.path.join(FRIEND, "corpus", "corpus.jsonl"))),
              "chunk_id")
        for split in FRIEND_SPLITS:
            write(os.path.join(args.out, f"friend-queries-{split}.jsonl.gz"),
                  ((r["_id"], r["text"])
                   for r in read_jsonl(os.path.join(FRIEND, "queries", f"{split}.jsonl"))),
                  "query_id")


if __name__ == "__main__":
    main()
