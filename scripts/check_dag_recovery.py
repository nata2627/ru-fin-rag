"""Проверка падения шага в середине регламента и восстановления на следующем прогоне.

Третий замер задачи 5.4. Идемпотентность проверена отдельно (два прогона подряд,
`check_dag_rerun.py`), здесь проверяется другое: что происходит с уже сделанной
работой, когда шаг в середине цепочки падает, и доделывает ли ее следующий
прогон.

Падение делается кодом, а не рассуждением. `scripts/run_dag.sh` подставляет
задачам интерпретатор-обманку, который отказывается запускать один шаг
(`scripts/build_index.py`) и возвращает код 9, а остальные шаги уходят
настоящему интерпретатору. Так падает ровно середина цепочки, причем падает так,
как упал бы настоящий шаг: подпроцесс вернул ненулевой код.

Ожидание, с которым сверяется исход, записано здесь списком, а не читается из
лога глазами:

1. Упавшая задача делает три попытки (одна плюс две из `retries`) и уходит в
   `failed`, прогон красный.
2. Шаги после нее не идут вовсе: `none_failed` пропускает падение докачки, но не
   падение переиндексации.
3. Индекс прошлого прогона на диске цел и не тронут: сервис продолжает отвечать
   на нем, а не остается без индекса.
4. Отметки о шаге нет, то есть следующий прогон не сочтет работу сделанной.
   Это и есть восстановление: датчик свежести видит отсутствие отметки и
   запускает шаг заново.
5. Отметка живости не записана: зеленая цепочка приемки не продолжается прогоном,
   который до конца не дошел.
6. Следующий прогон без обманки доводит все четыре задачи до конца, и словарь
   пересобранного индекса тот же, что был до падения.

Запуск идет в два этапа, потому что состояние надо снять до падения:
    python scripts/check_dag_recovery.py --stage before --state <файл>
    python scripts/check_dag_recovery.py --stage after --state <файл> \
        --states states-3.json states-4.json --log dag-test-3.log
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

from rufinrag.pipeline import INDEX, REGRESSION, digest, stamp_path  # noqa: E402

DAG_ID = "corpus_refresh"
FAILING_STEP = "build_index"
FAILING_SCRIPT = "scripts/build_index.py"
ATTEMPTS_EXPECTED = 3
WATCHED_FILES = ("data/index/bm25.npz", "data/index/texts.blocks", "results/regression.json")
EXPECTED = (
    {
        "fetch_acts": "skipped",
        "build_index": "failed",
        "run_regression": "upstream_failed",
        "record_heartbeat": "upstream_failed",
    },
    {
        "fetch_acts": "skipped",
        "build_index": "success",
        "run_regression": "success",
        "record_heartbeat": "success",
    },
)


def state_now() -> dict:
    index = json.loads((ROOT / "results/index.json").read_text(encoding="utf-8"))
    regulation = json.loads((ROOT / "results/regulation.json").read_text(encoding="utf-8"))
    return {
        "files": {
            name: digest(ROOT / name) if (ROOT / name).exists() else None for name in WATCHED_FILES
        },
        "stamps": {step.name: stamp_path(ROOT, step).exists() for step in (INDEX, REGRESSION)},
        "index_vocab": index["vocab"],
        "index_chunks": index["chunks"],
        "last_heartbeat": regulation["last_success"],
    }


def task_states(path: Path) -> dict[str, str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["task_id"]: row["state"] for row in rows}


def attempts(log: Path) -> int:
    """Сколько раз шаг действительно пробовали запустить.

    Считается по строке, которую печатает сама обманка: число попыток это факт
    запуска, а не пересказ состояний из базы Airflow.
    """
    if not log.exists():
        return 0
    return log.read_text(encoding="utf-8", errors="replace").count(
        f"учебная тревога: {FAILING_SCRIPT}"
    )


def problems(
    before: dict, after_fail: dict, after_fix: dict, runs: list[dict], tries: int
) -> list[str]:
    found = []
    if len(runs) != len(EXPECTED):
        return [f"прогонов {len(runs)}, а сравниваются два: падение и восстановление"]
    for number, (run, expected) in enumerate(zip(runs, EXPECTED, strict=True), start=1):
        for task, state in expected.items():
            actual = run["tasks"].get(task)
            if actual != state:
                found.append(
                    f"прогон {number}, задача {task}: ожидалось {state}, получено {actual}"
                )
    if tries != ATTEMPTS_EXPECTED:
        found.append(f"попыток упавшего шага {tries}, ожидалось {ATTEMPTS_EXPECTED}")
    changed = [
        name for name, value in after_fail["files"].items() if value != before["files"].get(name)
    ]
    if changed:
        found.append(f"падение шага тронуло файлы прошлого прогона: {', '.join(changed)}")
    if after_fail["stamps"][INDEX.name]:
        found.append(
            "после падения осталась отметка шага: следующий прогон сочтет работу сделанной"
        )
    if after_fail["last_heartbeat"] != before["last_heartbeat"]:
        found.append("отметка живости записана прогоном, который до конца не дошел")
    if not after_fix["stamps"][INDEX.name]:
        found.append("восстановление прошло, а отметки шага нет")
    if after_fix["index_vocab"] != before["index_vocab"]:
        found.append(
            f"словарь пересобранного индекса {after_fix['index_vocab']} против "
            f"{before['index_vocab']} до падения"
        )
    if after_fix["last_heartbeat"] == before["last_heartbeat"]:
        found.append("восстановившийся прогон не записал отметку живости")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("before", "fail", "after"), required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--states", type=Path, nargs="*", default=[])
    parser.add_argument("--log", type=Path)
    parser.add_argument("--seconds", type=float, nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=ROOT / "results/dag-recovery.json")
    args = parser.parse_args()

    if args.stage in ("before", "fail"):
        state = state_now()
        path = args.state if args.stage == "before" else args.state.with_suffix(".fail.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"состояние {args.stage}: словарь индекса {state['index_vocab']}, "
            f"отметка шага {state['stamps'][INDEX.name]}, живость {state['last_heartbeat']}"
        )
        return 0

    before = json.loads(args.state.read_text(encoding="utf-8"))
    after_fail = json.loads(args.state.with_suffix(".fail.json").read_text(encoding="utf-8"))
    after_fix = state_now()
    runs = [{"states_file": path.name, "tasks": task_states(path)} for path in sorted(args.states)]
    tries = attempts(args.log) if args.log else 0
    found = problems(before, after_fail, after_fix, runs, tries)

    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        # Средняя загрузка машины на момент замера. Без нее время прохода
        # обманывает: тот же прогон на занятом маке шел 234 с против 37 с на
        # свободном, и объяснить это число потом нечем.
        "load_average": [round(value, 2) for value in os.getloadavg()],
        "cpu_count": os.cpu_count(),
        "image": os.environ.get("AIRFLOW_IMAGE", "не записан"),
        "ci_run": os.environ.get("GITHUB_RUN_ID") or None,
        "dag_id": DAG_ID,
        "failing_step": FAILING_STEP,
        "failing_script": FAILING_SCRIPT,
        "attempts": tries,
        "tasks": len(EXPECTED[0]),
        "tasks_not_started": sorted(
            task for task, state in runs[0]["tasks"].items() if state == "upstream_failed"
        )
        if runs
        else [],
        "runs": runs,
        "seconds": [round(value, 1) for value in args.seconds],
        "before": before,
        "after_failure": after_fail,
        "after_recovery": after_fix,
        "files_touched_by_failure": [
            name
            for name, value in after_fail["files"].items()
            if value != before["files"].get(name)
        ],
        "stamp_after_failure": after_fail["stamps"][INDEX.name],
        # Поле называет то, что в нем лежит: True значит «отметка живости
        # записана прогоном, который до конца не дошел», то есть провал.
        # Прежнее имя значило обратное своему смыслу (AGENTS.md, раздел 2.1).
        "heartbeat_written_after_failure": (
            after_fail["last_heartbeat"] != before["last_heartbeat"]
        ),
        "passed": not found,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if found:
        print("падение шага в середине разошлось с ожиданием:", file=sys.stderr)
        for line in found:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(
        f"падение шага {FAILING_STEP}: попыток {tries}, задач после него не пошло "
        f"{len(report['tasks_not_started'])}, файлов прошлого прогона тронуто "
        f"{len(report['files_touched_by_failure'])}, восстановление на следующем прогоне прошло"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
