"""Порог алерта по деградации метрики, выведенный из разброса на эталонном наборе.

Алерт смотрит не на уровень метрики, а на падение против эталона на одном и том
же замороженном наборе. Значит и порог должен выводиться из разброса этой
разности, а не из ширины интервала самого уровня. Прежнее обоснование («половина
бутстрэп-интервала части А») брало статистику не про ту неопределенность, и этот
замер ее заменяет.

Разброс складывается из двух источников, и считаются они по-разному.

1. Прогон к прогону при неизменном входе. Берется из истории: каждый плановый
   прогон коммитит `results/regression.json`, и скрипт читает все версии этого
   файла из git, а не со слов автора.
2. Доливка корпуса. Новый акт меняет частоты слов, длины документов и состав
   конкурентов в выдаче, поэтому метрика шевелится, даже когда ничего не
   сломалось. Реальная доливка за историю проекта одна, поэтому доливка
   повторяется кодом: из корпуса убирается случайный акт, индекс пересобирается,
   метрика считается заново. Возврат этого акта обратно это доливка ровно того
   же размера, что ходит ночью. Убираются только акты, у которых в разметке нет
   ни одного фрагмента: акт с размеченным фрагментом унес бы часть эталона, и
   это была бы настоящая потеря полноты, а не помеха от чужого текста.

Правило выбора порога записано до чисел (`orchestration/EXPERIMENTS.md`, S16) и
применяется здесь без правок: порог это наибольшее по модулю изменение метрики
среди прогонов доливки, умноженное на пять, округленное вверх до шага 0,005, но
не меньше одного шага и меньше половины наименьшего реального сдвига качества,
который в проекте случался.

Выдача берется прямо у индекса BM25, а не у собранного сервиса: связка `bm25`
это ровно первая ступень без слияния и переранжирования. Что это один и тот же
путь, скрипт не предполагает, а проверяет на полном корпусе перед прогонами и
падает при первом же расхождении.

Запуск: python scripts/measure_alert_threshold.py [--draws 30] [--group-draws 10]
Стоит около 20 с на прогон, то есть сорок прогонов это примерно 15 минут.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_qrels, load_queries  # noqa: E402
from rufinrag.corpus import corpus_path, load_corpus  # noqa: E402
from rufinrag.metrics import DEPTH, bootstrap_ci, per_query  # noqa: E402
from rufinrag.retrieval.bm25 import BM25Index  # noqa: E402
from rufinrag.search import SearchService  # noqa: E402

WATCHED = "NDCG@10"
SEED = 20261024
SAFETY = 5.0
STEP = 0.005
GROUP_ACTS = 5
HISTORY_FILES = ("results/regression.json", "results/regression-baseline.json")


def ndcg_values(index: BM25Index, queries: dict, qrels: dict) -> list[float]:
    run = {
        query_id: [chunk_id for chunk_id, _ in index.search(query.text, k=DEPTH)]
        for query_id, query in queries.items()
        if query_id in qrels
    }
    _, values = per_query(run, qrels, WATCHED)
    return values


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((v - center) ** 2 for v in values) / (len(values) - 1))


def check_same_path(queries: dict, qrels: dict, expected: list[float]) -> dict:
    """Сверка короткого пути с путем сервиса на полном корпусе.

    Без нее весь замер висел бы на предположении, что связка `bm25` это ровно
    индекс и ничего больше.
    """
    service = SearchService.build(with_dense=False)
    run = {
        query_id: [
            hit.chunk_id for hit in service.search(query.text, top=DEPTH, config="bm25").hits
        ]
        for query_id, query in queries.items()
        if query_id in qrels
    }
    _, values = per_query(run, qrels, WATCHED)
    if values != expected:
        raise SystemExit(
            "выдача индекса разошлась с выдачей сервиса на полном корпусе: "
            f"{mean(values):.6f} против {mean(expected):.6f}"
        )
    return {
        "queries": len(run),
        "service_ndcg10": round(mean(values), 6),
        "corpus_version": service.corpus_version,
    }


def git_history() -> list[dict]:
    """Все закоммиченные версии файлов с метрикой: история прогонов из git.

    Числа в этих файлах округлены до четвертого знака, поэтому история отвечает
    только на вопрос, менялась ли метрика заметнее 0,00005.
    """
    rows: list[dict] = []
    for name in HISTORY_FILES:
        out = subprocess.run(
            ["git", "log", "--format=%H %ad", "--date=format:%d.%m %H:%M", "--", name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for line in out.strip().splitlines():
            commit, when = line.split(" ", 1)
            blob = subprocess.run(
                ["git", "show", f"{commit}:{name}"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            report = json.loads(blob)
            rows.append(
                {
                    "commit": commit[:7],
                    "at": when,
                    "file": name,
                    "machine": report["machine"],
                    "corpus_version": report["corpus_version"],
                    WATCHED: report["metrics"][WATCHED],
                }
            )
    return sorted(rows, key=lambda r: (r[WATCHED], r["at"]))


def history_spread(rows: list[dict]) -> dict:
    """Что история говорит про разброс и про реальный сдвиг качества.

    Группа это одно значение метрики: внутри группы прогоны отличаются машиной и
    версией корпуса, но не числом. Наименьший реальный сдвиг берется на одной и
    той же версии корпуса, иначе в него примешалась бы доливка.
    """
    groups: dict[float, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[WATCHED], []).append(row)
    biggest = max(groups.values(), key=len)
    by_corpus: dict[str, set[float]] = {}
    for row in rows:
        by_corpus.setdefault(row["corpus_version"], set()).add(row[WATCHED])
    shifts = [
        round(max(values) - min(values), 4) for values in by_corpus.values() if len(values) > 1
    ]
    return {
        "runs": len(rows),
        "distinct_values": sorted(groups),
        "repeated_value": max(groups, key=lambda v: len(groups[v])),
        "repeated_runs": len(biggest),
        "repeated_machines": sorted({row["machine"] for row in biggest}),
        "repeated_corpus_versions": sorted({row["corpus_version"] for row in biggest}),
        "real_shift_same_corpus": min(shifts) if shifts else None,
        "stored_precision": 0.0001,
    }


def draw_acts(acts: list[str], free: set[str], size: int, count: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    pool = [act for act in acts if act in free]
    return [rng.sample(pool, size) for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--group-draws", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "results/alert-threshold.json")
    args = parser.parse_args()

    started = time.monotonic()
    queries = load_queries()
    qrels = load_qrels()
    judged = {chunk_id for rel in qrels.values() for chunk_id in rel}

    corpus = load_corpus(corpus_path())
    texts = list(corpus.iter_texts())
    ids = corpus.ids
    act_of = [chunk.act.act_id for chunk in corpus.chunks]
    acts = sorted(set(act_of))
    busy = {act for act, chunk_id in zip(act_of, ids, strict=True) if chunk_id in judged}
    free = set(acts) - busy
    print(
        f"корпус {len(ids)} фрагментов, актов {len(acts)}, "
        f"с размеченными фрагментами {len(busy)}, свободных {len(free)}",
        flush=True,
    )

    full_index = BM25Index.build(ids, texts)
    full_values = ndcg_values(full_index, queries, qrels)
    full = mean(full_values)
    print(f"полный корпус: {WATCHED} {full:.6f} на {len(full_values)} вопросах", flush=True)
    same_path = check_same_path(queries, qrels, full_values)

    rounds: list[dict] = []
    for size, count in ((1, args.draws), (GROUP_ACTS, args.group_draws)):
        for dropped in draw_acts(acts, free, size, count, SEED + size):
            t0 = time.monotonic()
            gone = set(dropped)
            kept = [i for i, act in enumerate(act_of) if act not in gone]
            index = BM25Index.build([ids[i] for i in kept], [texts[i] for i in kept])
            value = mean(ndcg_values(index, queries, qrels))
            rounds.append(
                {
                    "acts_removed": size,
                    "chunks_removed": len(ids) - len(kept),
                    "acts": dropped,
                    WATCHED: round(value, 6),
                    # Доливка это возврат убранного акта обратно, поэтому падение
                    # метрики при доливке равно «без акта минус с актом».
                    "drop": round(value - full, 6),
                }
            )
            print(
                f"прогон {len(rounds)}: убрано актов {size}, фрагментов "
                f"{len(ids) - len(kept)}, {WATCHED} {value:.6f}, сдвиг "
                f"{value - full:+.6f}, {time.monotonic() - t0:.1f} с",
                flush=True,
            )

    previous = measure_previous_corpus(queries, qrels, full, len(ids))
    noise = [row["drop"] for row in rounds]
    ceiling = max(abs(value) for value in noise)
    threshold = max(STEP, math.ceil(SAFETY * ceiling / STEP) * STEP)

    history = git_history()
    spread = history_spread(history)
    real_shift = spread["real_shift_same_corpus"]
    cap = round(real_shift / 2, 4) if real_shift else None

    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "metric": WATCHED,
        "config": "bm25",
        "queries_scored": len(full_values),
        "corpus_version": same_path["corpus_version"],
        "full_corpus_ndcg10": round(full, 6),
        "same_path_as_service": same_path,
        "rule": {
            "safety": SAFETY,
            "step": STEP,
            "seed": SEED,
            "text": (
                "порог это наибольшее по модулю изменение метрики среди прогонов доливки, "
                "умноженное на пять, округленное вверх до шага 0,005, но не меньше одного "
                "шага и меньше половины наименьшего реального сдвига качества"
            ),
            "written_before": "orchestration/EXPERIMENTS.md, S16",
        },
        "history": {**spread, "runs_listed": history},
        "delivery_noise": {
            "runs": len(rounds),
            "single_act_runs": args.draws,
            "group_act_runs": args.group_draws,
            "acts_in_group": GROUP_ACTS,
            "chunks_removed_min": min(row["chunks_removed"] for row in rounds),
            "chunks_removed_max": max(row["chunks_removed"] for row in rounds),
            "mean_drop": round(mean(noise), 6),
            "stdev_drop": round(stdev(noise), 6),
            "max_drop": round(max(noise), 6),
            "min_drop": round(min(noise), 6),
            "ceiling": round(ceiling, 6),
            "runs_listed": rounds,
        },
        "real_delivery": previous,
        "level_bootstrap": {
            **bootstrap_ci(full_values),
            "note": (
                "интервал уровня метрики, а не разности. Порог из него не выводится: "
                "набор заморожен, и оба замера идут на одних и тех же вопросах"
            ),
        },
        "real_shift_same_corpus": real_shift,
        "cap": cap,
        "threshold": round(threshold, 4),
        "seconds": round(time.monotonic() - started, 1),
    }
    if cap is not None and threshold >= cap:
        report["passed"] = False
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"порог {threshold:.4f} уперся в потолок {cap:.4f}: набор слишком шумный "
            "для такого мониторинга",
            file=sys.stderr,
        )
        return 1

    report["passed"] = True
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"разброс доливки по {len(rounds)} прогонам: наибольший сдвиг {ceiling:.6f}, "
        f"порог {threshold:.4f} при потолке {cap}, числа в {args.out}"
    )
    return 0


def measure_previous_corpus(
    queries: dict, qrels: dict, full: float, chunks_now: int
) -> dict | None:
    """Та же метрика на прошлой версии корпуса, взятой из git.

    Это единственная настоящая доливка за историю проекта, измеренная тем же
    кодом, что и прогоны шума. Если она попадает в их разброс, значит доливка
    кодом повторяет то, что делает ночной прогон, а не что-то свое.
    """
    out = subprocess.run(
        ["git", "log", "--format=%H", "--", "data/corpus/chunks.jsonl.gz"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if len(out) < 2:
        return None
    commit = out[1]
    blob = ROOT / "logs" / "corpus-previous.jsonl.gz"
    blob.parent.mkdir(parents=True, exist_ok=True)
    with open(blob, "wb") as f:
        subprocess.run(
            ["git", "show", f"{commit}:data/corpus/chunks.jsonl.gz"],
            cwd=ROOT,
            stdout=f,
            check=True,
        )
    corpus = load_corpus(blob)
    index = BM25Index.build(corpus.ids, list(corpus.iter_texts()))
    value = mean(ndcg_values(index, queries, qrels))
    print(
        f"прошлая версия корпуса ({corpus.version}): {WATCHED} {value:.6f}, "
        f"сдвиг {value - full:+.6f}",
        flush=True,
    )
    return {
        "commit": commit[:7],
        "corpus_version": corpus.version,
        "chunks_added_by_delivery": chunks_now - len(corpus),
        WATCHED: round(value, 6),
        "drop": round(value - full, 6),
    }


if __name__ == "__main__":
    raise SystemExit(main())
