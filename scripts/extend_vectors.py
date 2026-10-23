"""Догнать плотные векторы до корпуса после ночной докачки.

Зачем. Корпус растет каждую ночь, а матрица векторов нет: ее считали один раз
при переносе из части А. Как только числа расходятся, `SearchService.build()`
отказывается собирать плотную ступень, и сервис локально поднимается одной
лексикой, а сверки плотной части пропускаются с причиной. Пересчитывать все
62 606 фрагментов ради двенадцати новых незачем: имена уже посчитанных идут в
корпусе первыми и в том же порядке, поэтому не хватает ровно хвоста.

Чем считаем. Тем же `QueryEncoder`, которым сервис кодирует запрос, с префиксом
фрагмента вместо префикса запроса. Это не предпочтение: матрицу считали в части
А через `sentence-transformers`, и совпадение двух путей надо не предполагать, а
мерить. Поэтому перед дописыванием кодируется выборка уже посчитанных фрагментов
и сравнивается с матрицей по косинусу. Медиана ниже порога означает, что векторы
окажутся из другого пространства, и дописывать нельзя: по метрикам поиска на
двенадцати фрагментах такую беду заметить невозможно.

Запуск: python scripts/extend_vectors.py
Только проверка, без записи: python scripts/extend_vectors.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.corpus import corpus_path, load_corpus  # noqa: E402
from rufinrag.search import DENSE_MODEL_DIR, TEXTS_FILE, VECTOR_IDS_FILE, VECTORS_FILE  # noqa: E402

# Порог, ниже которого дописывать нельзя. Записан до замера в EXPERIMENTS.md
# (S14): ожидание медианы не ниже 0,999, отказ ниже 0,99.
MIN_COSINE = 0.99


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def drift(encoder, corpus, ids: list[str], vectors: np.ndarray, sample: int) -> dict:
    """Насколько наш энкодер совпадает с тем, которым посчитана матрица.

    Выборка берется с равным шагом по всей матрице, а не из начала: фрагменты
    одного акта лежат рядом, и первые двести пришлись бы на один документ.
    """
    step = max(1, len(ids) // sample)
    picked = list(range(0, len(ids), step))[:sample]
    texts = [corpus.text(ids[i]) for i in picked]
    fresh = encoder.encode(texts, is_query=False)
    stored = vectors[picked].astype(np.float32)
    cosine = np.sum(fresh * stored, axis=1) / (
        np.linalg.norm(fresh, axis=1) * np.linalg.norm(stored, axis=1)
    )
    return {
        "sample": len(picked),
        "cosine_median": round(float(np.median(cosine)), 6),
        "cosine_min": round(float(np.min(cosine)), 6),
        "cosine_mean": round(float(np.mean(cosine)), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--sample", type=int, default=200, help="сверка с матрицей: сколько пар")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="посчитать и не записывать")
    parser.add_argument("--report", type=Path, default=ROOT / "results/vectors.json")
    args = parser.parse_args()

    vectors_path = args.data / VECTORS_FILE
    ids_path = args.data / VECTOR_IDS_FILE
    model_dir = args.data / DENSE_MODEL_DIR
    for path in (vectors_path, ids_path, model_dir):
        if not path.exists():
            raise SystemExit(
                f"нет {path}: плотная часть не перенесена, см. scripts/import_corpus.py"
            )

    corpus = load_corpus(corpus_path(args.data), args.data / TEXTS_FILE)
    stored = read_ids(ids_path)
    vectors = np.load(vectors_path)
    if len(stored) != vectors.shape[0]:
        raise SystemExit(f"{vectors_path}: векторов {vectors.shape[0]}, имен {len(stored)}")

    if corpus.ids[: len(stored)] != stored:
        raise SystemExit(
            f"{ids_path}: имена матрицы не совпадают с началом корпуса. Дописать хвост нельзя, "
            "нужен полный пересчет: порядок строк матрицы обязан совпадать с порядком корпуса"
        )
    missing = corpus.ids[len(stored) :]
    print(f"корпус {len(corpus)}, матрица {len(stored)}, не хватает {len(missing)}")

    encoder_started = time.monotonic()
    from rufinrag.retrieval.dense import QueryEncoder

    encoder = QueryEncoder(model_dir, device=args.device)
    load_seconds = time.monotonic() - encoder_started

    measured = drift(encoder, corpus, stored, vectors, args.sample)
    print(json.dumps(measured, ensure_ascii=False))
    if measured["cosine_median"] < MIN_COSINE:
        raise SystemExit(
            f"сверка с матрицей дала медиану косинуса {measured['cosine_median']} при пороге "
            f"{MIN_COSINE}: наш энкодер считает не то же, что считала часть А, дописывать нельзя"
        )

    added = np.zeros((0, vectors.shape[1]), dtype=vectors.dtype)
    started = time.monotonic()
    if missing:
        texts = [corpus.text(chunk_id) for chunk_id in missing]
        pieces = [
            encoder.encode(texts[i : i + args.batch_size], is_query=False)
            for i in range(0, len(texts), args.batch_size)
        ]
        added = np.vstack(pieces).astype(vectors.dtype)
    seconds = time.monotonic() - started

    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "model": str(model_dir.relative_to(args.data.parent)),
        "device": args.device,
        "dim": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
        "corpus_version": corpus.version,
        "chunks_before": len(stored),
        "chunks_after": len(stored) + len(missing),
        "added": len(missing),
        "seconds": round(seconds, 1),
        "model_load_seconds": round(load_seconds, 1),
        "per_second": round(len(missing) / seconds, 1) if missing and seconds else None,
        "agreement_with_part_a": measured,
        "replaced": {
            "measured_at": "2026-10-21",
            "chunks": len(stored),
            "source": "перенос из части А, scripts/import_corpus.py",
        },
    }
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("--dry-run: ничего не записано")
        return 0

    if missing:
        # Матрица перезаписывается целиком, а не дописывается на месте: 48 МБ
        # это дешевле, чем рисковать половиной записи в файл, из которого
        # сервис читает на старте.
        np.save(vectors_path, np.vstack([vectors, added]))
        ids_path.write_text("\n".join(corpus.ids) + "\n", encoding="utf-8")
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
