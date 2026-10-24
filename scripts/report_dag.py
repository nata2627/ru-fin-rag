"""Отчет по регламенту на Airflow: три замера задачи 5.4 одним файлом.

`results/dag.md` собирается кодом, а не пишется руками: все три числа уже лежат
в отчетах прогонов, и переписывание их в текст это ровно то место, где число
расходится с замером.

Вход:
    results/dag-rerun.json      два прогона подряд на маке, пик памяти, время
    results/dag-rerun-ci.json   тот же замер в прогоне CI, другая машина
    results/dag-recovery.json   учебное падение шага в середине цепочки
    results/corpus-update.json  последняя докачка актов, плановый прогон
    results/corpus-update-first.json  первый прогон регламента, заморожен

Запуск: python scripts/report_dag.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

TASKS = ("fetch_acts", "build_index", "run_regression", "record_heartbeat")


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def date_ru(stamp: str) -> str:
    """Дата так, как она пишется в текстах проекта: 24.10.2026."""
    year, month, day = stamp.split("-")
    return f"{day}.{month}.{year}"


def ru(value: float, digits: int = 1) -> str:
    text = f"{value:,.{digits}f}" if abs(value) >= 10_000 else f"{value:.{digits}f}"
    return text.replace(",", " ").replace(".", ",")


def seconds_text(report: dict, number: int) -> str:
    values = report.get("seconds") or []
    if len(values) <= number:
        return "не записано"
    return f"{ru(values[number])} с"


def main() -> int:
    local = load("dag-rerun.json")
    ci = load("dag-rerun-ci.json")
    recovery = load("dag-recovery.json")
    update = load("corpus-update.json")
    first = load("corpus-update-first.json")

    failing = recovery["failing_step"]
    fail_run, fix_run = recovery["runs"]
    lines = [
        "# Регламент на Airflow: время прохода, акты за прогон, падение шага в середине",
        "",
        "Файл собирает `scripts/report_dag.py` из отчетов прогонов, руками он не правится.",
        "Числа ниже это задача 5.4 плана: три замера, каждый из своего файла.",
        "",
        "## Время полного прохода",
        "",
        "| Прогон | Машина | Первый проход | Повтор | Пик памяти |",
        "|---|---|---|---|---|",
        f"| контейнер на маке, {date_ru(local['measured_at'])} | {local['machine']} | "
        f"{seconds_text(local, 0)} | {seconds_text(local, 1)} | {local['peak_memory_mb']} МБ |",
        f"| прогон CI {ci['ci_run']}, {date_ru(ci['measured_at'])} | {ci['machine']} | "
        f"{seconds_text(ci, 0)} | {seconds_text(ci, 1)} | {ci['peak_memory_mb']} МБ |",
        "",
        f"Образ один и тот же, {local['image']}, лимит памяти контейнера 2 ГБ стоит в"
        " `scripts/run_dag.sh`.",
        "Повтор короче первого прохода, потому что датчики свежести видят неизменившийся вход и",
        f"пропускают {len(local['skipped_on_repeat'])} задачи из {local['tasks']}:"
        f" {', '.join(local['skipped_on_repeat'])}.",
        "Пик памяти на двух машинах разный, и это свойство замера, а не ошибка: число без",
        "названной машины здесь не значит ничего. Лимит выбран по большему из двух.",
        f"Время прохода в прогоне CI запишет следующий прогон `dag`: отчет прогона {ci['ci_run']}",
        "снят до того, как замер времени попал в файл, а число из лога прогона сюда не",
        "переносится.",
        "",
        "## Число актов за прогон",
        "",
        "| Что | Число | Файл |",
        "|---|---|---|",
        f"| актов в корпусе | {local['corpus_after']['acts']} | dag-rerun.json |",
        f"| фрагментов в корпусе | {ru(local['corpus_after']['chunks'], 0)} | dag-rerun.json |",
        f"| новых актов в перечне у cbr.ru, последний плановый прогон |"
        f" {update['new_acts_in_index']} | corpus-update.json |",
        f"| из них добавлено в корпус | {update['new_acts_added']} | corpus-update.json |",
        f"| новых фрагментов за прогон | {update['new_chunks']} | corpus-update.json |",
        f"| добавлено фрагментов за два прогона DAG | {local['added_chunks']} | dag-rerun.json |",
        f"| повторов идентификаторов после двух прогонов | {local['duplicate_chunk_ids']} |"
        " dag-rerun.json |",
        "",
        f"Ноль добавленных актов это не простой регламента. Первый прогон"
        f" {date_ru(first['checked_at'][:10])} нашел {first['new_acts_in_index']} недостающих"
        f" актов и не взял ни одного: все {first['skipped_total']} отложены с причиной в"
        " `data/corpus/skipped.json` (не опознаются в выпуске при нарезке PDF или сканы с"
        " испорченным OCR).",
        "Установившийся прогон спрашивает перечень, сверяет его с корпусом и с памятью о",
        "неудачах, поэтому ничего не добавляет, пока не выйдет новый «Вестник». В контейнере шаг",
        "докачки пропускается всегда: боевой режим там выключен, и сеть к cbr.ru прогон CI не",
        "трогает вовсе.",
        "",
        "## Поведение при падении шага в середине",
        "",
        f"Падение сделано кодом: интерпретатор-обманка отказывается запускать"
        f" `{recovery['failing_script']}` и возвращает код 9, остальные шаги уходят настоящему"
        " интерпретатору. Проверка и ожидание лежат в `scripts/check_dag_recovery.py`.",
        "",
        "| Что проверено | Исход |",
        "|---|---|",
        f"| попыток упавшего шага {failing} | {recovery['attempts']} (одна и две из `retries`) |",
        f"| задач после него не пошло |"
        f" {len(recovery['tasks_not_started'])} из {recovery['tasks']}:"
        f" {', '.join(recovery['tasks_not_started'])} |",
        f"| файлов прошлого прогона тронуто |"
        f" {len(recovery['files_touched_by_failure'])} из"
        f" {len(recovery['before']['files'])} |",
        "| отметка шага после падения | "
        + ("осталась" if recovery["stamp_after_failure"] else "снята")
        + ", значит следующий прогон шаг повторит |",
        f"| отметка живости прогоном, который не дошел до конца |"
        f" {'записана' if not recovery['heartbeat_after_failure'] else 'не записана'} |",
        f"| словарь индекса до падения и после восстановления |"
        f" {ru(recovery['before']['index_vocab'], 0)} и"
        f" {ru(recovery['after_recovery']['index_vocab'], 0)} |",
        "",
        "Состояния задач по прогонам:",
        "",
        "| Прогон | " + " | ".join(f"`{task}`" for task in TASKS) + " |",
        "|---|" + "---|" * len(TASKS),
        "| с обманкой | " + " | ".join(fail_run["tasks"][task] for task in TASKS) + " |",
        "| следующий, без обманки | " + " | ".join(fix_run["tasks"][task] for task in TASKS) + " |",
        "",
        "Сервис от падения не остается без индекса: индекс прошлого прогона на диске цел, и",
        "отвечает он на нем. Работу доделывает следующий прогон, а не третья попытка упавшего",
        "шага: отметки шага нет, датчик свежести это видит и запускает переиндексацию заново.",
        "",
        "## Чего эти числа не показывают",
        "",
        "Планировщика Airflow здесь нет: прогон идет командой `airflow dags test` на SQLite, одним",
        "процессом и по порядку. Расписание проекту дает GitHub Actions, поэтому времена выше это",
        "время регламента, а не время реакции планировщика.",
        f"Падение проверено на одном шаге из {recovery['tasks']}, на `{failing}`. Он выбран",
        "потому, что стоит в середине цепочки и после него есть и регрессия, и отметка живости.",
        "Два прогона подряд проверяют идемпотентность на неизменившемся входе. Случай, когда",
        "корпус за время прогона вырос, в этом замере не проверяется.",
        "",
    ]
    out = RESULTS / "dag.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"собрано: {out.relative_to(ROOT)}, строк {len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
