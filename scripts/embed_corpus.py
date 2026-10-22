"""Векторы корпуса энкодером `deepvk/USER-bge-m3` на видеокарте.

Зачем. В части А выигравшая связка дала 0,690 NDCG@10 на векторах USER-bge-m3,
но самих векторов на диске не осталось: их считали на арендованной карте и
вернули только списки выдачи. Без них ни сервис, ни диагностика качества не
могут проверить выигравшую связку своим кодом, и приходится считать на
дообученной модели, которая просела на живых вопросах.

Настройки повторяют часть А дословно (`src/rufin/retrieval/model_specs.py` и
`retrieval/dense.py`): модель `deepvk/USER-bge-m3`, префиксы не используются,
максимальная длина 512 токенов, нормировка векторов включена, половинная
точность на карте. Разойдись любая из них, векторы перестанут быть
сопоставимыми с бенчмарком, и заметить это по числам будет нельзя.

Прогон рассчитан на чужую машину, поэтому он продолжается после обрыва:
векторы пишутся в файл на диске по мере счета, рядом лежит файл прогресса,
повторный запуск начинает с того места, где остановился. Перед полным
прогоном стоит сделать пробу на малой части: она печатает скорость и прогноз.

Запуск пробы:  python scripts/embed_corpus.py --limit 400
Полный прогон: python scripts/embed_corpus.py
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

from rufinrag.corpus import load_corpus  # noqa: E402

MODEL = "deepvk/USER-bge-m3"
DIM = 1024
MAX_SEQ = 512
SHARD = 500


def progress_path(out: Path) -> Path:
    return out.with_suffix(".progress.json")


def read_progress(out: Path, total: int) -> int:
    """Сколько фрагментов уже посчитано. Файл прогресса пишется после каждого
    куска, а не в конце: процесс, снятый снаружи, финальный блок не исполняет."""
    path = progress_path(out)
    if not path.exists() or not out.exists():
        return 0
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("model") != MODEL or state.get("total") != total:
        raise SystemExit(f"файл прогресса {path} от другого прогона, удалите его или задайте --out")
    return int(state["done"])


def write_progress(out: Path, done: int, total: int, seconds: float) -> None:
    progress_path(out).write_text(
        json.dumps(
            {
                "model": MODEL,
                "total": total,
                "done": done,
                "seconds": round(seconds, 1),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data/vectors/user-bge-m3-1024.npy")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="проба на первых N фрагментах")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    corpus = load_corpus()
    texts = list(corpus.iter_texts())
    ids = corpus.ids
    total = args.limit or len(texts)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("карта не видна: torch.cuda.is_available() ложно")

    done = read_progress(args.out, total)
    vectors = np.lib.format.open_memmap(
        args.out,
        mode="r+" if done else "w+",
        dtype=np.float16,
        shape=(total, DIM),
    )

    model = SentenceTransformer(
        MODEL, device=args.device, model_kwargs={"torch_dtype": torch.float16}
    )
    model.max_seq_length = MAX_SEQ

    print(f"модель {MODEL}, устройство {args.device}, фрагментов {total}, с места {done}")
    started = time.monotonic()
    processed = 0
    for start in range(done, total, SHARD):
        end = min(start + SHARD, total)
        batch = model.encode(
            texts[start:end],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors[start:end] = batch.astype(np.float16)
        vectors.flush()
        processed += end - start
        spent = time.monotonic() - started
        speed = processed / spent
        left = (total - end) / speed if speed else 0
        write_progress(args.out, end, total, spent)
        print(
            f"{end}/{total}, {speed:.1f} фрагмента в секунду, осталось {left / 60:.0f} мин",
            flush=True,
        )

    ids_path = args.out.with_suffix(".ids.txt")
    ids_path.write_text("\n".join(ids[:total]) + "\n", encoding="utf-8")

    report = {
        "model": MODEL,
        "device": args.device,
        "chunks": total,
        "dim": DIM,
        "max_seq_length": MAX_SEQ,
        "batch_size": args.batch_size,
        "normalized": True,
        "dtype": "float16",
        "seconds": round(time.monotonic() - started, 1),
        "speed_per_second": round(processed / (time.monotonic() - started), 1) if processed else 0,
    }
    if args.device == "cuda":
        report["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1048576, 1)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (args.out.parent / "user-bge-m3-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"векторы: {args.out}\nномера: {ids_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
