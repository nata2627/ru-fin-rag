"""Переранжирование готовой выдачи кросс-энкодером.

Зачем отдельно от сервиса. В онлайне кросс-энкодер выключен: на процессоре он
стоит 44 секунды на запрос. Но качество связки с ним надо знать числом, иначе в
таблице README остается пустая клетка, а соседнее число из части А относится к
другой связке (реранк поверх гибрида на bge-m3, а не на дообученной модели).

Считается на видеокарте, вход это уже посчитанная выдача первой ступени.
Тексты фрагментов берутся из корпуса сервиса, то есть ровно те, что индексирует
сам сервис.

Запуск: python scripts/rerank_run.py --run data/runs/hybrid.jsonl --depth 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_queries  # noqa: E402
from rufinrag.corpus import load_corpus  # noqa: E402

MODEL = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 512


def read_run(path: Path) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        rows = (json.loads(line) for line in f if line.strip())
        return {row["query_id"]: row["ranked"] for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "results/rerank-run.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "results/rerank.json")
    args = parser.parse_args()

    run = read_run(args.run)
    queries = load_queries()
    corpus = load_corpus()

    model_kwargs: dict = {}
    torch = None
    if args.device != "cpu":
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("карта не видна: torch.cuda.is_available() ложно")
        model_kwargs = {"torch_dtype": torch.float16}

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        MODEL, device=args.device, max_length=MAX_LENGTH, model_kwargs=model_kwargs
    )

    order = [q for q in sorted(run) if q in queries]
    print(f"{MODEL}, запросов {len(order)}, глубина {args.depth}", flush=True)
    started = time.monotonic()
    pairs_done = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for i, query_id in enumerate(order, start=1):
            candidates = run[query_id][: args.depth]
            pairs = [(queries[query_id].text, corpus.text(c)) for c in candidates]
            scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
            ranked = [
                c for c, _ in sorted(zip(candidates, scores, strict=True), key=lambda x: -x[1])
            ]
            out.write(
                json.dumps({"query_id": query_id, "ranked": ranked}, ensure_ascii=False) + "\n"
            )
            pairs_done += len(pairs)
            if i % 100 == 0 or i == len(order):
                spent = time.monotonic() - started
                speed = pairs_done / spent
                left = (len(order) - i) * args.depth / speed
                print(
                    f"{i}/{len(order)}, {speed:.0f} пар в секунду, осталось {left / 60:.0f} мин",
                    flush=True,
                )

    report = {
        "model": MODEL,
        "device": args.device,
        "source_run": args.run.name,
        "queries": len(order),
        "depth": args.depth,
        "pairs": pairs_done,
        "batch_size": args.batch_size,
        "max_length": MAX_LENGTH,
        "seconds": round(time.monotonic() - started, 1),
        "pairs_per_second": round(pairs_done / (time.monotonic() - started), 1),
    }
    if torch is not None and args.device == "cuda":
        report["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1048576, 1)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
