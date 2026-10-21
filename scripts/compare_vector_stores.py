"""Qdrant против FAISS против numpy: задержка, память, качество.

Зачем три, а не два. numpy это то, чем сервис ищет сейчас: точное умножение по
62 594 векторам на 384 измерения. FAISS это стандарт для такого размера, а
Qdrant это хранилище с сетевым интерфейсом и фильтрами по метаданным, то есть
другой класс инструмента. Сравнение честно только если у всех трех одна и та
же выдача по качеству, поэтому рядом с задержкой считается совпадение с точным
поиском.

Qdrant поднимается во встроенном режиме (локальная папка), а не контейнером,
если его адрес не задан. Встроенный режим меряет структуру индекса и память,
но не сетевой слой, и это написано в отчете.

Запуск: python scripts/compare_vector_stores.py [--queries 20] [--qdrant-url ...]
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_queries  # noqa: E402
from rufinrag.search import SearchService  # noqa: E402

TOP = 10
COLLECTION = "rufin"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def overlap(mine: list[str], reference: list[str]) -> float:
    return len(set(mine) & set(reference)) / len(reference)


def measure(fn, vectors: list[np.ndarray]) -> tuple[list[list[str]], dict[str, float]]:
    results = []
    times = []
    for vector in vectors:
        t0 = time.monotonic()
        results.append(fn(vector))
        times.append((time.monotonic() - t0) * 1000)
    return results, {
        "p50_ms": round(statistics.median(times), 3),
        "p95_ms": round(percentile(times, 0.95), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--qdrant-url", default="")
    parser.add_argument("--out", type=Path, default=ROOT / "results/vector-store.json")
    args = parser.parse_args()

    service = SearchService.build()
    if service.dense is None or service.encoder is None:
        print(f"нет плотной части: {service.degraded}", file=sys.stderr)
        return 1

    queries = [q.text for q in load_queries().values()][: args.queries]
    vectors = [service.encoder.encode_one(text) for text in queries]
    corpus_vectors = service.dense.vectors
    ids = service.dense.ids

    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "vectors": int(corpus_vectors.shape[0]),
        "dim": int(corpus_vectors.shape[1]),
        "source_dtype": str(corpus_vectors.dtype),
        "queries": len(queries),
        "top": TOP,
        "stores": {},
    }

    # numpy, точный поиск. Он же эталон качества для остальных.
    exact, timings = measure(
        lambda v: [chunk_id for chunk_id, _ in service.dense.search(v, k=TOP)], vectors
    )
    report["stores"]["numpy"] = {
        **timings,
        "index_mb": round(corpus_vectors.astype(np.float32).nbytes / 1048576, 1),
        "stored_dtype": "float32 на время умножения",
        "overlap_with_exact": 1.0,
        "note": "точное умножение, им сервис ищет сейчас",
    }

    vectors32 = corpus_vectors.astype(np.float32)

    import faiss

    index = faiss.IndexFlatIP(vectors32.shape[1])
    t0 = time.monotonic()
    index.add(vectors32)
    faiss_build = time.monotonic() - t0

    def faiss_search(vector):
        _, found = index.search(vector.reshape(1, -1).astype(np.float32), TOP)
        return [ids[i] for i in found[0]]

    found, timings = measure(faiss_search, vectors)
    report["stores"]["faiss"] = {
        **timings,
        "index_mb": round(vectors32.nbytes / 1048576, 1),
        "stored_dtype": "float32",
        "build_seconds": round(faiss_build, 2),
        "overlap_with_exact": round(
            statistics.mean(overlap(a, b) for a, b in zip(found, exact, strict=True)), 4
        ),
        "note": "IndexFlatIP, точный перебор по нормированным векторам",
    }

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    storage = Path(tempfile.mkdtemp(prefix="qdrant-"))
    mode = "сервер" if args.qdrant_url else "встроенный режим, локальная папка"
    client = QdrantClient(url=args.qdrant_url) if args.qdrant_url else QdrantClient(path=storage)
    try:
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=vectors32.shape[1], distance=Distance.COSINE),
        )
        t0 = time.monotonic()
        batch = 2000
        for start in range(0, len(ids), batch):
            client.upsert(
                COLLECTION,
                points=[
                    PointStruct(id=i, vector=vectors32[i].tolist(), payload={"chunk_id": ids[i]})
                    for i in range(start, min(start + batch, len(ids)))
                ],
            )
        qdrant_build = time.monotonic() - t0

        def qdrant_search(vector):
            hits = client.query_points(
                COLLECTION, query=vector.astype(np.float32).tolist(), limit=TOP
            ).points
            return [h.payload["chunk_id"] for h in hits]

        found, timings = measure(qdrant_search, vectors)
        report["stores"]["qdrant"] = {
            **timings,
            "index_mb": round(
                sum(f.stat().st_size for f in storage.rglob("*") if f.is_file()) / 1048576, 1
            )
            if not args.qdrant_url
            else None,
            "stored_dtype": "float32",
            "build_seconds": round(qdrant_build, 2),
            "overlap_with_exact": round(
                statistics.mean(overlap(a, b) for a, b in zip(found, exact, strict=True)), 4
            ),
            "mode": mode,
        }
    finally:
        client.close()
        if not args.qdrant_url:
            shutil.rmtree(storage, ignore_errors=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, row in report["stores"].items():
        print(name, row, flush=True)
    print(f"записано в {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
