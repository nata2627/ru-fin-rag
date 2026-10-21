"""Сборка индекса BM25 из корпуса сервиса.

Индекс сохраняется файлом, потому что сборка из 62 594 фрагментов занимает
около десяти секунд, и на холодном старте бесплатного хостинга это десять
секунд недоступности.

Запуск: python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.corpus import corpus_path, load_corpus  # noqa: E402
from rufinrag.retrieval.bm25 import BM25Index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    t0 = time.monotonic()
    corpus = load_corpus(corpus_path(args.data))
    read_seconds = time.monotonic() - t0
    index = BM25Index.build(corpus.ids, corpus.texts)
    out = args.data / "index/bm25.npz"
    index.save(out)
    print(
        f"корпус {len(corpus)} фрагментов прочитан за {read_seconds:.1f} с, "
        f"индекс собран за {index.build_seconds:.1f} с, словарь {len(index.vocab)}, "
        f"в памяти {index.size_mb:.1f} МБ, файл {out.stat().st_size / 1e6:.1f} МБ"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
