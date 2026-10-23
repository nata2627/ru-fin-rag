"""Счет приемки автономности по истории плановых прогонов.

Дата берется не из памяти автора, а из истории прогонов GitHub: нужен прогон,
запущенный событием `schedule`, а не кнопкой. Ручные запуски проверяют код,
расписание проверяет автономность, и смешивать их в этом счете нельзя.

Правило приемки записано до того, как пошли числа, и читается буквально:
четырнадцать календарных дней **подряд**. Значит красный плановый прогон рвет
цепочку, и отсчет начинается заново с первого зеленого планового прогона после
него. Поэтому в отчете два разных числа, и путать их нельзя:
`days_since_first_run` это сколько прошло с самого первого зеленого прогона за
историю, а `streak_days` это длина текущей непрерывной цепочки, то есть то
единственное, что сравнивается с четырнадцатью днями. Знаменатель публикуется
рядом: сколько плановых прогонов было всего и сколько из них зеленые.

Запуск: python scripts/record_acceptance.py
"""

from __future__ import annotations

import argparse
import calendar
import itertools
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = (
    "четырнадцать календарных дней подряд: плановый прогон проходит без участия человека, "
    "коммиты в истории делает только бот, постоянных расходов нет"
)


def elapsed_days(stamp: str) -> float:
    """Сколько суток прошло с момента в UTC.

    `timegm`, а не `mktime`: второй читает разобранное время как местное, и счет
    дней уезжал бы на часовой пояс машины, где идет прогон.
    """
    return (time.time() - calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))) / 86400


def decided_scheduled(runs: list[dict]) -> list[dict]:
    """Плановые прогоны с известным исходом, по возрастанию времени.

    Прогоны без вывода это те, что идут прямо сейчас: исхода у них еще нет, и в
    счет они не идут ни с одной стороны.
    """
    return sorted(
        (r for r in runs if r["event"] == "schedule" and r["conclusion"]),
        key=lambda r: r["createdAt"],
    )


def streak_of(scheduled: list[dict]) -> list[dict]:
    """Текущая непрерывная цепочка зеленых плановых прогонов.

    Считается с конца: правило требует четырнадцать дней подряд, поэтому
    зеленые прогоны до красного в цепочку не входят, сколько бы их ни было.
    """
    tail = itertools.takewhile(lambda r: r["conclusion"] == "success", reversed(scheduled))
    return list(tail)[::-1]


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
    scheduled = decided_scheduled(json.loads(raw))
    green = [r for r in scheduled if r["conclusion"] == "success"]
    if not green:
        raise SystemExit("планового зеленого прогона еще не было, фиксировать нечего")

    streak = streak_of(scheduled)
    broken_by = next((r for r in reversed(scheduled) if r["conclusion"] != "success"), None)

    first = green[0]
    report = {
        "started_at": first["createdAt"],
        "first_scheduled_run": first["url"],
        "scheduled_runs_total": len(scheduled),
        "scheduled_runs_green": len(green),
        "scheduled_runs_red": len(scheduled) - len(green),
        "streak_green_runs": len(streak),
        "streak_started_at": streak[0]["createdAt"] if streak else None,
        "streak_days": round(elapsed_days(streak[0]["createdAt"]), 1) if streak else 0.0,
        "days_required": 14,
        "days_since_first_run": round(elapsed_days(first["createdAt"]), 1),
        "broken_by": (
            {
                "run": broken_by["url"],
                "at": broken_by["createdAt"],
                "conclusion": broken_by["conclusion"],
            }
            if broken_by
            else None
        ),
        "rule": RULE,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
