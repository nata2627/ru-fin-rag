#!/usr/bin/env python3
"""Векторы произвольной нарезки на видеокарте, с продолжением после обрыва.

Зачем отдельный скрипт. Абляции этапа 3 (документы 13 и 14) сравнивают ветви
нарезки и чужой набор, а векторов для них на диске нет: этап B считал векторы
только базовой нарезки и только на Kaggle. Прогон идет на чужой машине, и
процесс, снятый снаружи, финальный блок не исполняет, поэтому векторы пишутся
в файл по мере счета, рядом лежит файл прогресса, а повторный запуск
продолжает с того места, где остановился.

Настройки берутся из описания модели (`rufin.retrieval.model_specs`) через тот
же `Encoder`, которым считает весь проект: префиксы, предел длины, нормировка
и половинная точность на карте. Своих настроек здесь нет вовсе. Разойдись
любая из них с этапом B, векторы перестанут быть сопоставимыми, и заметить
это по числам будет нельзя.

Матрица хранится в float16, как и векторы этапа B: для косинуса точности
хватает, а места и времени на пересылку уходит вдвое меньше.

Проба:          python scripts/embed_chunks.py --input chunks.jsonl.gz --model e5-small --out emb/x --limit 400
Полный прогон:  python scripts/embed_chunks.py --input chunks.jsonl.gz --model e5-small --out emb/x
Запросы:        python scripts/embed_chunks.py --input queries.jsonl --model e5-small --out emb/q --queries
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.retrieval.model_specs import MODELS  # noqa: E402

SHARD = 2000


def open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def read_texts(path: str, id_field: str, text_field: str) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open_maybe_gz(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ids.append(rec[id_field])
            texts.append(rec[text_field])
    return ids, texts


def progress_path(out_dir: str) -> str:
    return os.path.join(out_dir, "progress.json")


def read_done(out_dir: str, model: str, total: int) -> int:
    path = progress_path(out_dir)
    if not os.path.exists(path) or not os.path.exists(os.path.join(out_dir, "vectors.npy")):
        return 0
    state = json.load(open(path, encoding="utf-8"))
    if state.get("model") != model or state.get("total") != total:
        raise SystemExit(f"{path} остался от другого прогона: модель "
                         f"{state.get('model')}, фрагментов {state.get('total')}. "
                         "Задайте другой --out.")
    return int(state["done"])


def count_tokens(spec, texts: list[str], args) -> None:
    """Сколько токенов видит энкодер и у скольких записей он режет хвост.

    Вопрос не праздный. Нарезка считает границу фрагмента в токенах, и
    считает ее токенизатором той версии библиотеки, что стояла при нарезке.
    Энкодер считает ту же границу своей версией, и если версии разошлись,
    часть фрагментов окажется длиннее предела, то есть обрезанной. Довод
    «базовый фрагмент по построению не длиннее предела» держится на
    совпадении версий, и доля обрезанных должна быть напечатана, а не
    предположена нулевой.
    """
    import statistics

    from transformers import AutoTokenizer
    prefix = spec.query_prefix if args.queries else spec.passage_prefix
    tok = AutoTokenizer.from_pretrained(spec.path, use_fast=True)
    lengths: list[int] = []
    for start in range(0, len(texts), 1000):
        batch = [prefix + t for t in texts[start:start + 1000]]
        lengths.extend(len(x) for x in tok(batch, add_special_tokens=True)["input_ids"])
    limit = spec.max_seq_length
    over = sum(1 for n in lengths if n > limit)
    near = sum(1 for n in lengths if limit - 8 <= n <= limit)
    report = {"model": args.model, "path": spec.path,
              "input": os.path.basename(args.input), "records": len(lengths),
              "max_seq_length": limit, "with_prefix": bool(prefix),
              "median": statistics.median(lengths), "max": max(lengths),
              "over_limit": over, "within_8_of_limit": near}
    with open(os.path.join(args.out, "tokens.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl или jsonl.gz с текстами")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", required=True, help="папка под vectors.npy, ids.txt, meta.json")
    ap.add_argument("--limit", type=int, default=0, help="проба на первых N записях")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--queries", action="store_true",
                    help="это запросы: берется префикс запроса, а не фрагмента")
    ap.add_argument("--id-field", default=None)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--tokens-only", action="store_true",
                    help="не считать векторы, а измерить длины в токенах "
                         "токенизатором этой же модели")
    args = ap.parse_args()

    spec = MODELS[args.model]
    id_field = args.id_field or ("query_id" if args.queries else "chunk_id")
    ids, texts = read_texts(args.input, id_field, args.text_field)
    total = min(args.limit, len(ids)) if args.limit else len(ids)
    ids, texts = ids[:total], texts[:total]
    os.makedirs(args.out, exist_ok=True)

    if args.tokens_only:
        count_tokens(spec, texts, args)
        return

    # Имена пишутся до счета: процесс, снятый снаружи, до конца не доходит,
    # а матрица без имен не читается.
    with open(os.path.join(args.out, "ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")

    from rufin.retrieval.dense import Encoder
    enc = Encoder(spec, device=args.device, batch_size=args.batch_size)
    dim = spec.truncate_dim or spec.dim

    done = read_done(args.out, args.model, total)
    vectors = np.lib.format.open_memmap(
        os.path.join(args.out, "vectors.npy"),
        mode="r+" if done else "w+", dtype=np.float16, shape=(total, dim))

    print(f"модель {spec.name} ({spec.path}), устройство {enc.device}, "
          f"половинная точность {enc.half}, предел {spec.max_seq_length} токенов",
          flush=True)
    print(f"записей {total} из {args.input}, с места {done}, пачка {args.batch_size}",
          flush=True)

    started = time.monotonic()
    processed = 0
    for start in range(done, total, SHARD):
        end = min(start + SHARD, total)
        batch = enc.encode(texts[start:end], is_query=args.queries)
        if batch.shape[1] != dim:
            raise SystemExit(f"модель дала {batch.shape[1]} измерений, в описании {dim}")
        vectors[start:end] = batch.astype(np.float16)
        vectors.flush()
        processed += end - start
        spent = time.monotonic() - started
        speed = processed / spent if spent else 0.0
        with open(progress_path(args.out), "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "total": total, "done": end,
                       "seconds": round(spent, 1),
                       "per_second": round(speed, 1),
                       "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                      f, ensure_ascii=False, indent=1)
        left = (total - end) / speed if speed else 0
        print(f"{end}/{total}, {speed:.1f} записи в секунду, "
              f"осталось {left / 60:.1f} мин", flush=True)

    spent = time.monotonic() - started
    meta = {"model": args.model, "path": spec.path, "input": os.path.basename(args.input),
            "records": total, "dim": dim, "dtype": "float16",
            "max_seq_length": spec.max_seq_length, "batch_size": args.batch_size,
            "queries": args.queries, "device": enc.device, "half": enc.half,
            "normalized": True, "seconds": round(spent, 1),
            "per_second": round(processed / spent, 1) if processed and spent else None,
            "size_mb": round(vectors.nbytes / 1048576, 1)}
    if args.device == "cuda":
        import torch
        meta["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1048576, 1)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
