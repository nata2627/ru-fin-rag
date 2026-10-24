"""Проверка повторного прогона DAG: работа сделана один раз, дублей нет.

Скрипт вызывается вокруг двух подряд прогонов DAG в контейнере
(`scripts/run_dag.sh`) и отвечает на три вопроса числами, а не на глаз по логу.

1. Первый прогон действительно сделал работу, а не пропустил все шаги. Иначе
   зеленый прогон CI ничего не доказывал бы: DAG, который все пропустил, тоже
   зеленый.
2. Второй прогон работу не повторил: датчики свежести увидели, что вход не
   менялся, и шаги пропущены.
3. Корпус после двух прогонов тот же, и в нем нет ни одного повторяющегося
   идентификатора фрагмента.

Запуск идет в два этапа, потому что состояние корпуса надо снять до первого
прогона:
    python scripts/check_dag_rerun.py --stage before --state <файл>
    python scripts/check_dag_rerun.py --stage after --state <файл> \
        --states states-1.json states-2.json --out results/dag-rerun.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.corpus import corpus_path, read_jsonl  # noqa: E402
from rufinrag.pipeline import digest  # noqa: E402

DAG_ID = "corpus_refresh"
# Чего ждем от каждого прогона. Первый прогон обязан собрать индекс и посчитать
# регрессию, второй обязан их пропустить. Докачка в контейнере пропускается
# всегда: боевой режим там выключен, и cbr.ru прогон CI не трогает.
EXPECTED = (
    {
        "fetch_acts": "skipped",
        "build_index": "success",
        "run_regression": "success",
        "record_heartbeat": "success",
    },
    {
        "fetch_acts": "skipped",
        "build_index": "skipped",
        "run_regression": "skipped",
        "record_heartbeat": "success",
    },
)


def corpus_state() -> dict:
    path = corpus_path()
    rows = 0
    chunk_ids = set()
    acts = set()
    for row in read_jsonl(path):
        rows += 1
        chunk_ids.add(row["chunk_id"])
        acts.add(row["act_id"])
    return {
        "digest": digest(path),
        "chunks": rows,
        "acts": len(acts),
        "unique_chunk_ids": len(chunk_ids),
    }


def task_states(path: Path) -> dict[str, str]:
    """Состояния задач одного прогона из вывода `airflow tasks states-for-dag-run`."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["task_id"]: row["state"] for row in rows}


def problems(before: dict, after: dict, runs: list[dict]) -> list[str]:
    found = []
    if after["chunks"] != after["unique_chunk_ids"]:
        found.append(
            f"в корпусе повторяются идентификаторы фрагментов: строк {after['chunks']}, "
            f"различных {after['unique_chunk_ids']}"
        )
    if after["digest"] != before["digest"]:
        found.append(
            f"корпус изменился за два прогона: было фрагментов {before['chunks']}, "
            f"стало {after['chunks']}"
        )
    if len(runs) != len(EXPECTED):
        found.append(f"прогонов DAG {len(runs)}, а сравниваются два")
        return found
    for number, (run, expected) in enumerate(zip(runs, EXPECTED, strict=True), start=1):
        for task, state in expected.items():
            actual = run["tasks"].get(task)
            if actual != state:
                found.append(
                    f"прогон {number}, задача {task}: ожидалось {state}, получено {actual}"
                )
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("before", "after"), required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--states", type=Path, nargs="*", default=[])
    parser.add_argument("--seconds", type=float, nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=ROOT / "results/dag-rerun.json")
    args = parser.parse_args()

    if args.stage == "before":
        state = corpus_state()
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"корпус до прогонов: {state['chunks']} фрагментов, отпечаток {state['digest']}")
        return 0

    before = json.loads(args.state.read_text(encoding="utf-8"))
    after = corpus_state()
    runs = [{"states_file": path.name, "tasks": task_states(path)} for path in sorted(args.states)]
    found = problems(before, after, runs)
    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "image": os.environ.get("AIRFLOW_IMAGE", "не записан"),
        # Номер прогона CI, в котором снят замер: по нему открывается лог задач
        # в артефактах. На маке поля нет, и это видно по `machine`.
        "ci_run": os.environ.get("GITHUB_RUN_ID") or None,
        "dag_id": DAG_ID,
        "peak_memory_mb": int(os.environ.get("RUFINRAG_DAG_PEAK_MB", 0)) or None,
        "tasks": len(EXPECTED[0]),
        # Время прохода снимает вызывающий скрипт: ему видно начало и конец
        # всего прогона, включая разбор DAG и работу базы, а задачам нет.
        "seconds": [round(value, 1) for value in args.seconds],
        "runs": runs,
        "skipped_on_repeat": sorted(
            task
            for task, state in (runs[1]["tasks"].items() if len(runs) > 1 else [])
            if state == "skipped"
        ),
        "corpus_before": before,
        "corpus_after": after,
        "added_chunks": after["chunks"] - before["chunks"],
        "duplicate_chunk_ids": after["chunks"] - after["unique_chunk_ids"],
        "passed": not found,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if found:
        print("повторный прогон DAG разошелся с ожиданием:", file=sys.stderr)
        for line in found:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(
        f"два прогона подряд: добавлено фрагментов {report['added_chunks']}, "
        f"повторов идентификаторов {report['duplicate_chunk_ids']}, "
        f"пропущено на повторе {len(report['skipped_on_repeat'])} задач из {report['tasks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
