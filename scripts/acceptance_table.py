"""Итог приемки автономности: четырнадцать дней по дням, замороженным файлом.

`results/acceptance.json` каждый день перезаписывает свой прогон, поэтому итог
из него брать нельзя: завтра там будет другое число. Этот скрипт запускается
один раз, когда цепочка дожила до четырнадцати дней, и пишет
`results/acceptance-passed.json`: окно ровно в четырнадцать суток от первого
прогона цепочки, плановые прогоны окна по дням и коммиты за те же дни по
авторам. Готовый файл повторный запуск не трогает.

Коммиты считаются отдельно от прогонов нарочно. Правило говорит про регламент,
а репозиторий в эти дни жил: шла работа над частью А. Сколько в окне было не
ботовых коммитов и какие из них задели файл регламента, написано числом, чтобы
«без участия человека» не читалось шире, чем измерено.

Запуск: python scripts/acceptance_table.py [--print]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import record_acceptance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "acceptance-passed.json"
WORKFLOW = ".github/workflows/nightly.yml"
BOT = "github-actions[bot]"
STAMP = "%Y-%m-%dT%H:%M:%SZ"
DAYS = 14


def git_lines(*args: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def commits(since: str, until: str) -> dict:
    span = [f"--since={since}", f"--until={until}"]
    authors = git_lines("log", *span, "--format=%an")
    touched = git_lines("log", *span, "--format=%h %an", "--", WORKFLOW)
    return {
        "bot": sum(a == BOT for a in authors),
        "not_bot": sum(a != BOT for a in authors),
        "not_bot_touching_workflow": [
            line.split()[0] for line in touched if line.split(" ", 1)[1] != BOT
        ],
    }


def window(runs: list[dict]) -> dict:
    """Окно в четырнадцать суток от первого прогона текущей цепочки.

    Прогоны цепочки после конца окна в итог не идут: приемка это четырнадцать
    суток, а цепочка продолжает расти и дальше.
    """
    streak = record_acceptance.streak_of(record_acceptance.decided_scheduled(runs))
    if not streak:
        raise SystemExit("цепочки зеленых плановых прогонов нет, итог писать не из чего")
    start = streak[0]["createdAt"]
    end = (dt.datetime.strptime(start, STAMP) + dt.timedelta(days=DAYS)).strftime(STAMP)
    inside = [r for r in streak if r["createdAt"] <= end]
    days: dict[str, list[dict]] = {}
    for run in inside:
        days.setdefault(run["createdAt"][:10], []).append(run)
    return {
        "window_start": start,
        "window_end": end,
        "days_required": DAYS,
        "scheduled_runs": len(inside),
        "scheduled_runs_green": sum(r["conclusion"] == "success" for r in inside),
        "manual_runs": sum(
            r["event"] != "schedule" and start <= r["createdAt"] <= end for r in runs
        ),
        "days": [
            {"date": date, "runs": [{"id": r["databaseId"], "url": r["url"]} for r in day]}
            for date, day in sorted(days.items())
        ],
    }


def build() -> dict:
    raw = subprocess.run(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            "nightly.yml",
            "--limit",
            "100",
            "--json",
            "databaseId,event,conclusion,createdAt,url",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    report = window(json.loads(raw))
    start = report["window_start"]
    if record_acceptance.elapsed_days(start) < DAYS:
        raise SystemExit(f"цепочка идет с {start} и до {DAYS} дней еще не дожила")
    report["commits"] = commits(start, report["window_end"])
    report["rule"] = record_acceptance.RULE
    return report


def day(stamp: str) -> str:
    return f"{stamp[8:10]}.{stamp[5:7]}.{stamp[:4]}"


def rows(report: dict) -> list[str]:
    """Строки таблицы по дням ровно в том виде, в каком они стоят в README."""
    return [
        f"| {day(d['date'])} | {len(d['runs'])} | "
        + ", ".join(f"[{r['id']}]({r['url']})" for r in d["runs"])
        + " |"
        for d in report["days"]
    ]


def summary(report: dict) -> str:
    c = report["commits"]
    return (
        f"**Приемка пройдена: с {day(report['window_start'])} по {day(report['window_end'])}, "
        f"{report['days_required']} суток подряд, плановых прогонов "
        f"{report['scheduled_runs']}, зеленых {report['scheduled_runs_green']} из "
        f"{report['scheduled_runs']}, запусков руками {report['manual_runs']}.** "
        f"Коммитов бота за эти дни {c['bot']}, моих {c['not_bot']}, из них файл регламента "
        f"задел {len(c['not_bot_touching_workflow'])}."
    )


def table(report: dict) -> str:
    head = ["| День (UTC) | Плановых прогонов | Прогоны |", "|---|---|---|"]
    return "\n".join([summary(report), "", *head, *rows(report)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="напечатать таблицу по готовому файлу")
    args = ap.parse_args()
    if args.print:
        print(table(json.loads(OUT.read_text(encoding="utf-8"))))
        return
    if OUT.exists():
        raise SystemExit(f"{OUT.name} уже записан, итог приемки не пересчитывается")
    report = build()
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(table(report))


if __name__ == "__main__":
    main()
