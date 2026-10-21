"""Сколько памяти занимает сервис: пик и состояние после запросов.

Зачем отдельный скрипт. Бесплатные тарифы ограничивают память (у Render это
512 МБ), и решает не средняя величина, а пик на старте: процесс, который
кратко вырос выше предела, убивается. Пик берется из `ru_maxrss`, текущее
потребление из `ps`, потому что питон не возвращает освобожденную память
операционной системе и `ru_maxrss` ее не покажет.

Запуск: python scripts/measure_memory.py [--config bm25]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_queries  # noqa: E402
from rufinrag.search import SearchService  # noqa: E402


def peak_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def current_mb() -> float:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip()) / 1e3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bm25")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "results/memory.json")
    args = parser.parse_args()

    with_dense = args.config not in ("bm25", "bm25-rerank")
    t0 = time.monotonic()
    service = SearchService.build(with_dense=with_dense)
    startup = time.monotonic() - t0

    queries = [q.text for q in load_queries().values()][: args.queries]
    latencies = []
    for text in queries:
        t0 = time.monotonic()
        service.search(text, 10, args.config)
        latencies.append((time.monotonic() - t0) * 1000)

    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    report.update(
        {
            "measured_at": time.strftime("%Y-%m-%d"),
            "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
            "corpus_version": service.corpus_version,
            "text_blocks_mb": round(service.corpus.texts.size_bytes / 1048576, 1),
            "bm25_index_mb": round(service.bm25.size_mb, 1),
        }
    )
    report.setdefault("configs", {})[args.config] = {
        "startup_seconds": round(startup, 2),
        "peak_mb": round(peak_mb(), 1),
        "current_mb": round(current_mb(), 1),
        "queries": len(queries),
        "p50_ms": round(statistics.median(latencies), 2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["configs"][args.config], ensure_ascii=False), flush=True)
    print(f"записано в {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
