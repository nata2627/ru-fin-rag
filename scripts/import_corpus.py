"""Перенос корпуса, векторов и эталонного набора из части А в данные сервиса.

Запускается один раз на стройке. Сервис после этого ни от папки части А, ни от
машины автора не зависит: фрагменты лежат сжатыми в его собственных данных
(DECISIONS, решение 7).

Запуск: python scripts/import_corpus.py [--part-a ПУТЬ]
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.corpus import FIELDS  # noqa: E402

# Часть А лежит внутри этого же репозитория папкой `retrieval` (решение 22):
# сдается одна ссылка и один клон, поэтому путь считается от корня репозитория,
# а не от домашней папки.
DEFAULT_PART_A = Path(__file__).resolve().parents[1] / "retrieval"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-a", type=Path, default=DEFAULT_PART_A)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    chunks_src = args.part_a / "data/chunks/base.jsonl"
    vectors_src = args.part_a / "data/embeddings/base/rufin/vectors.npy"
    ids_src = args.part_a / "data/embeddings/base/rufin/ids.txt"
    for path in (chunks_src, vectors_src, ids_src):
        if not path.exists():
            print(f"нет файла {path}", file=sys.stderr)
            return 1

    corpus_dst = args.data / "corpus/chunks.jsonl.gz"
    corpus_dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    rows = 0
    with (
        open(chunks_src, encoding="utf-8") as src,
        gzip.open(corpus_dst, "wt", encoding="utf-8", compresslevel=9) as dst,
    ):
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(
                json.dumps({k: row[k] for k in FIELDS if k in row}, ensure_ascii=False) + "\n"
            )
            rows += 1
    print(
        f"фрагментов {rows}, {corpus_dst} {corpus_dst.stat().st_size / 1e6:.1f} МБ, "
        f"{time.monotonic() - t0:.1f} с"
    )

    bench_dst = args.data / "benchmark"
    bench_dst.mkdir(parents=True, exist_ok=True)
    queries_rows = 0
    with (
        open(args.part_a / "data/queries/queries.jsonl", encoding="utf-8") as src,
        open(bench_dst / "queries.jsonl", "w", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(
                json.dumps(
                    {k: row[k] for k in ("query_id", "text", "origin", "podvyborka")},
                    ensure_ascii=False,
                )
                + "\n"
            )
            queries_rows += 1
    shutil.copyfile(args.part_a / "data/queries/qrels.tsv", bench_dst / "qrels.tsv")
    # Выдача базовой линии из части А целиком: по ней сверяется, что поиск
    # сервиса воспроизводит бенчмарк, и по ней же пересчитываются его числа.
    runs = {
        "bm25": "base__bm25.jsonl",
        "dense": "base__dense-rufin.jsonl",
        "hybrid": "base__hybrid-rufin.jsonl",
    }
    for name, src_name in runs.items():
        shutil.copyfile(args.part_a / "data/runs" / src_name, bench_dst / f"run-{name}.jsonl")
    print(
        f"эталонный набор: запросов {queries_rows}, разметка и выдачи "
        f"{', '.join(runs)} в {bench_dst}"
    )

    vectors_dst = args.data / "vectors/rufin-384.npy"
    vectors_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vectors_src, vectors_dst)
    shutil.copyfile(ids_src, vectors_dst.with_suffix(".ids.txt"))
    print(f"векторы {vectors_dst} {vectors_dst.stat().st_size / 1e6:.1f} МБ")

    # Веса модели запроса. В репозиторий они не едут: 470 МБ больше предела
    # файла в git, и чем их заменить на хостинге, решается в задаче 2.4.
    model_src = args.part_a / "data/weights/d-matryoshka"
    model_dst = args.data / "models/rufin"
    if model_src.exists():
        model_dst.mkdir(parents=True, exist_ok=True)
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "obuchenie.json",
        ):
            if (model_src / name).exists():
                shutil.copyfile(model_src / name, model_dst / name)
        total = sum(f.stat().st_size for f in model_dst.iterdir())
        print(f"модель запроса {model_dst} {total / 1e6:.1f} МБ")
    else:
        print(f"нет весов {model_src}, плотный поиск останется выключенным")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
