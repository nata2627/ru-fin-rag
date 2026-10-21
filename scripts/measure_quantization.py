"""Квантизация векторов: что теряем в качестве, что выигрываем в памяти.

Способ простой и объяснимый: скалярная квантизация в int8 по общей для всей
матрицы шкале. Вектора нормированы, значения лежат в узком диапазоне, поэтому
одной шкалы хватает, а по-канальная усложнила бы и хранение, и объяснение.

Качество считается не метрикой на разметке, а совпадением с точной выдачей:
вопрос здесь инженерный, «та же ли это выдача», а не «хорошая ли она».
Доверительный интервал разницы берется бутстрэпом по запросам, потому что
разброс по запросам тут и есть источник неопределенности.

Запуск: python scripts/measure_quantization.py [--queries 100]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_queries  # noqa: E402
from rufinrag.search import SearchService  # noqa: E402

TOP = 10
SEED = 13
BOOTSTRAP = 2000


def search_matrix(matrix: np.ndarray, ids: list[str], vector: np.ndarray) -> list[str]:
    scores = matrix @ vector
    part = np.argpartition(-scores, TOP)[:TOP]
    order = part[np.argsort(-scores[part])]
    return [ids[i] for i in order]


def bootstrap_ci(values: list[float], seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=np.float64)
    means = data[rng.integers(0, len(data), size=(BOOTSTRAP, len(data)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--out", type=Path, default=ROOT / "results/quantization.json")
    args = parser.parse_args()

    service = SearchService.build()
    if service.dense is None or service.encoder is None:
        print(f"нет плотной части: {service.degraded}", file=sys.stderr)
        return 1

    ids = service.dense.ids
    base = service.dense.vectors.astype(np.float32)
    queries = [q.text for q in load_queries().values()][: args.queries]
    vectors = [service.encoder.encode_one(text) for text in queries]

    scale = float(np.abs(base).max()) / 127.0
    quantized = np.clip(np.round(base / scale), -127, 127).astype(np.int8)

    exact = [search_matrix(base, ids, v) for v in vectors]

    t0 = time.monotonic()
    approx = [search_matrix(quantized.astype(np.float32) * scale, ids, v) for v in vectors]
    seconds = time.monotonic() - t0

    overlaps = [len(set(a) & set(b)) / TOP for a, b in zip(approx, exact, strict=True)]
    top1 = [float(a[0] == b[0]) for a, b in zip(approx, exact, strict=True)]
    low, high = bootstrap_ci([1.0 - o for o in overlaps])

    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "vectors": int(base.shape[0]),
        "dim": int(base.shape[1]),
        "queries": len(queries),
        "top": TOP,
        "scale": scale,
        "memory": {
            "float16_source_mb": round(service.dense.vectors.nbytes / 1048576, 1),
            "float32_mb": round(base.nbytes / 1048576, 1),
            "int8_mb": round(quantized.nbytes / 1048576, 1),
        },
        "quality": {
            "mean_overlap_top10": round(statistics.mean(overlaps), 4),
            "loss_mean": round(1.0 - statistics.mean(overlaps), 4),
            "loss_ci95": [round(low, 4), round(high, 4)],
            "same_top1_share": round(statistics.mean(top1), 4),
            "same_top1_count": f"{int(sum(top1))} из {len(top1)}",
        },
        "search_seconds_total": round(seconds, 2),
        "note": (
            "скалярная квантизация в int8 с одной шкалой на матрицу, "
            "качество это совпадение с точной выдачей, а не метрика на разметке"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["memory"], ensure_ascii=False), flush=True)
    print(json.dumps(report["quality"], ensure_ascii=False), flush=True)
    print(f"записано в {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
