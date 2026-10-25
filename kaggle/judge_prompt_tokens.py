#!/usr/bin/env python3
"""Сколько токенов занимает подсказка судьи по новой инструкции.

Нужно затем, что `--max-model-len` задаётся явно, а не на глаз. Прежний прогон
7B жил на 1548, новая инструкция длиннее, и подсказка, которая не влезла,
обрезается молча: модель отвечает на обрубок, а в метаданных прогона об этом
ничего нет. Счёт идёт тем же сборщиком подсказки (`gpu_gen._chat`) и тем же
листом, что пойдёт в прогон, иначе число мерило бы не то.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_gen as G  # noqa: E402
import gpu_judge as J  # noqa: E402

from rufin.annotation import ANSWER_FORMAT, RUBRIC  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    pairs = J.read_pool(args.pool)
    texts = J.load_texts(args.chunks)
    out = {"лист": args.pool, "пар": len(pairs), "модели": {}}
    for name in args.models:
        tok = AutoTokenizer.from_pretrained(name)
        lens = sorted(
            len(tok(G._chat(tok, J.PROMPT.format(
                rubric=RUBRIC, query=p["vopros"], passage=texts[p["chunk_id"]],
                format=ANSWER_FORMAT), system=J.SYSTEM),
                add_special_tokens=False)["input_ids"])
            for p in pairs)
        out["модели"][name] = {
            "минимум": lens[0], "медиана": lens[len(lens) // 2],
            "95-я": lens[int(len(lens) * 0.95)], "максимум": lens[-1],
            "отпечаток инструкции": J.stack().get("рубрика sha256"),
        }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
