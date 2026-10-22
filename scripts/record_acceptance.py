"""Начало приемки автономности: первый зеленый прогон по расписанию.

Дата берется не из памяти автора, а из истории прогонов GitHub: нужен прогон,
запущенный событием `schedule`, а не кнопкой. Ручные запуски проверяют код,
расписание проверяет автономность, и смешивать их в этом счете нельзя.

Запуск: python scripts/record_acceptance.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = (
    "четырнадцать календарных дней подряд: плановый прогон проходит без участия человека, "
    "коммиты в истории делает только бот, постоянных расходов нет"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", default="nightly.yml")
    parser.add_argument("--out", type=Path, default=ROOT / "results/acceptance.json")
    args = parser.parse_args()

    raw = subprocess.run(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            args.workflow,
            "--limit",
            "100",
            "--json",
            "databaseId,event,conclusion,createdAt,url",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    runs = [r for r in json.loads(raw) if r["event"] == "schedule" and r["conclusion"] == "success"]
    if not runs:
        raise SystemExit("планового зеленого прогона еще не было, фиксировать нечего")

    first = min(runs, key=lambda r: r["createdAt"])
    started = first["createdAt"]
    days = (time.time() - time.mktime(time.strptime(started, "%Y-%m-%dT%H:%M:%SZ"))) / 86400
    report = {
        "started_at": started,
        "first_scheduled_run": first["url"],
        "scheduled_runs_green": len(runs),
        "days_passed": round(days, 1),
        "days_required": 14,
        "rule": RULE,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
