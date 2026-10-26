#!/usr/bin/env python3
"""Разметка пула судьей на своей карте, тем же кодом, что на Kaggle.

Зачем отдельный вход. `run_phase_c.py` рассчитан на раскладку Kaggle: входы в
`/kaggle/input`, нарезка считается или берется из датасета, модель ищется среди
подключенных. На внешнем ПК ничего этого нет, а карта есть, и квота Kaggle на
нем не тратится вовсе. Поэтому здесь путь короткий: нарезка, лист, модель,
разметка. Сама разметка идет функцией `gpu_judge.judge`, то есть тем же кодом,
инструкцией и разбором ответа, что в прогоне на Kaggle, иначе числа двух машин
нельзя было бы сравнивать.

Что важно на карте 8 ГБ (RTX 3070): модель в четырех битах вместе с
активациями оставляет под кэш ключей меньше гигабайта, поэтому графы CUDA
выключаются (`--enforce-eager`), а длина подсказки задается явно. Рассуждение
вслух выключено в `gpu_gen._chat`: с ним Qwen3 давал 0,5 пары/с против 5,6.

Запуск:
    python3 kaggle/judge_local.py --chunks ~/work-3b/chunks/base.jsonl \\
        --pool data/queries/pool_probe_kaggle.tsv --out judge_local.tsv \\
        --model Qwen/Qwen3-4B-AWQ --enforce-eager
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_gen as G  # noqa: E402
import gpu_judge as J  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="каноническая нарезка, jsonl")
    ap.add_argument("--pool", required=True, help="лист пар: query_id, chunk_id, vopros")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=J.DEFAULT_JUDGE)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--quantization", default="awq", help="пусто — без квантования")
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=2560)
    ap.add_argument("--gpu-util", type=float, default=0.86)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="размечать только первые N пар")
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()

    pairs = J.read_pool(args.pool)
    if args.limit:
        pairs = pairs[: args.limit]
    texts = J.load_texts(args.chunks)
    missing = sum(1 for p in pairs if p["chunk_id"] not in texts)
    print(f"пар {len(pairs)}, фрагментов в нарезке {len(texts)}, "
          f"пар без фрагмента {missing}", flush=True)
    if missing:
        # Пара без текста это не мелочь: она уйдет в счетчик, и знаменатель
        # разметки перестанет совпадать с листом. Чаще всего это другая нарезка.
        raise SystemExit("нарезка не та, что у листа: часть фрагментов не найдена")

    llm = G.load_llm(args.model, args.dtype, args.quantization or None,
                     args.tensor_parallel, args.max_model_len, args.gpu_util,
                     prefix_caching=True, max_num_seqs=args.max_num_seqs,
                     seed=args.seed, enforce_eager=args.enforce_eager)
    result = J.judge(llm, pairs, texts, args.out, batch=args.batch,
                     seed=args.seed, max_tokens=args.max_tokens,
                     model=args.model, weights=args.model)
    print(json.dumps({k: v for k, v in result.items() if k != "стек вывода"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
