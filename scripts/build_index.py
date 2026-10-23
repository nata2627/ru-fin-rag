"""Сборка рабочих артефактов поиска из корпуса сервиса.

Собираются два файла. Индекс BM25, потому что сборка из 62 594 фрагментов
занимает около семи секунд. И блоки текстов, потому что сжатие их на старте
занимает еще пять, а холодный старт на бесплатном хостинге случается после
каждого простоя.

Запуск: python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.corpus import corpus_path, load_corpus  # noqa: E402
from rufinrag.retrieval.bm25 import BM25Index  # noqa: E402
from rufinrag.search import TEXTS_FILE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    t0 = time.monotonic()
    corpus = load_corpus(corpus_path(args.data))
    read_seconds = time.monotonic() - t0
    index = BM25Index.build(corpus.ids, list(corpus.iter_texts()))
    out = args.data / "index/bm25.npz"
    index.save(out)
    texts_out = args.data / TEXTS_FILE
    corpus.texts.save(texts_out)
    # Числа сборки идут в результат, а не только в вывод команды: словарь и
    # размер индекса стоят в README, а стеммер их заметно изменил.
    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "chunks": len(corpus),
        "vocab": len(index.vocab),
        "read_seconds": round(read_seconds, 1),
        "build_seconds": round(index.build_seconds, 1),
        "memory_mb": round(index.size_mb, 1),
        "index_file_mb": round(out.stat().st_size / 1e6, 1),
        "texts_file_mb": round(texts_out.stat().st_size / 1e6, 1),
    }
    result = ROOT / "results/index.json"
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"корпус {len(corpus)} фрагментов прочитан за {read_seconds:.1f} с, "
        f"индекс собран за {index.build_seconds:.1f} с, словарь {len(index.vocab)}, "
        f"в памяти {index.size_mb:.1f} МБ, файл {out.stat().st_size / 1e6:.1f} МБ, "
        f"блоки текстов {texts_out.stat().st_size / 1e6:.1f} МБ, "
        f"числа в {result.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
