"""Мониторинг в ночном прогоне: дрейф запросов, доля без хорошего ответа, регрессия.

Три сигнала в одном шаге и в одном отчете. Первые два считаются здесь по
журналу запросов, третий берется готовым из `results/regression.json`: его
считает прогон регрессии, и переcчитывать его тут значило бы развести два числа
про одно и то же (задача 5.3).

Пороги читаются из замера (`results/monitoring-thresholds.json`), а не стоят в
коде. Правило включается, когда в журнале набралось окно: пока запросов меньше,
отчет говорит статус `dormant` и печатает, сколько их из нужных. Это не
поломка, а честное состояние сервиса, на который еще никто не ходил.

Прогон не краснеет от сработавшего алерта, и это сознательно. Красный плановый
прогон обнуляет часы приемки автономности, а дрейф запросов и доля без ответа
это не поломка регламента: сигнал уходит тем же путем, которым уходит отказ
cbr.ru, то есть issue при зеленом прогоне. Шаг печатает в `GITHUB_OUTPUT`
`alert=fired`, по нему срабатывает шаг issue, а текст с числами issue читает из
отчета. Ключ `--strict` возвращает ненулевой код: он нужен учебной лестнице
(`scripts/drill_monitoring.py`) и запуску руками, а не прогону.

Запуск: python scripts/check_monitoring.py [--events data/events] [--strict]
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

from rufinrag.benchmark import load_queries  # noqa: E402
from rufinrag.config import data_dir  # noqa: E402
from rufinrag.monitoring import (  # noqa: E402
    answered_window,
    answerless_share,
    drift,
    event_totals,
    load_index,
    monitoring_rule,
    read_events,
    shape_of_event,
    shape_of_text,
)

REGRESSION = "results/regression.json"


def regression_signal(root: Path) -> dict:
    """Третий сигнал: деградация на эталонном наборе, посчитанная прогоном регрессии."""
    path = root / REGRESSION
    if not path.exists():
        return {
            "name": "benchmark_regression",
            "fired": False,
            "measured": False,
            "text": "",
            "note": f"нет {REGRESSION}: регрессия в этом прогоне не считалась",
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    baseline = report.get("baseline", {})
    # Блок `alert` появился в отчете задачей 5.3, и в репозитории лежит отчет
    # прогона постарше. Поэтому сигнал читается из блока, если он есть, и из
    # результата сравнения с эталоном, если его еще нет: иначе мониторинг
    # молчал бы про деградацию до первого нового прогона.
    alert = report.get("alert")
    if alert is None:
        drop = baseline.get("drop")
        threshold = baseline.get("tolerance")
        fired = not report.get("passed", True)
        text = (
            f"NDCG@10 упал на {drop} против эталона от {baseline.get('measured_at')} "
            f"при пороге {threshold} на {report.get('queries_scored')} размеченных вопросах"
            if fired
            else ""
        )
    else:
        drop = alert.get("drop")
        threshold = alert.get("threshold")
        fired = bool(alert.get("fired"))
        text = alert.get("text", "")
    return {
        "name": "benchmark_regression",
        "fired": fired,
        "measured": True,
        "metric": "NDCG@10",
        "value": drop,
        "threshold": threshold,
        "denominator": report.get("queries_scored"),
        "measured_at": report.get("measured_at"),
        "report_has_alert_block": alert is not None,
        "text": text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results/monitoring.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    rule = monitoring_rule(ROOT)
    directory = args.events or data_dir() / "events"
    events = read_events(directory) if directory.is_dir() else []
    totals = event_totals(events)
    window_events = answered_window(events, rule["window"])

    index = load_index()
    queries = load_queries()
    reference = [shape_of_text(query.text, index.vocab) for query in queries.values()]
    edges = rule["token_bin_edges"]

    signals = [regression_signal(ROOT)]
    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "journal": {"dir": str(directory), **totals},
        "window": {
            "needed": rule["window"],
            "collected": len(window_events),
            "days": totals["days"],
        },
        "thresholds": rule,
        "vocab": len(index.vocab),
        "reference_queries": len(reference),
    }

    if len(window_events) < rule["window"]:
        report["status"] = "dormant"
        report["reason"] = (
            f"запросов в журнале {len(window_events)} из {rule['window']} нужных для окна: "
            "правило дрейфа и правило доли без хорошего ответа не включены. Пороги выведены "
            "из разброса на замороженном наборе, а не из трафика"
        )
        print(report["reason"], flush=True)
    else:
        report["status"] = "watching"
        shapes = [shape_of_event(event, index.vocab) for event in window_events]
        found = drift(shapes, reference, edges)
        share = answerless_share(shapes, rule["support_feature"], rule["support_floor"])
        signals.insert(
            0,
            {
                "name": "query_drift",
                "fired": found["psi"] > rule["drift_threshold"],
                "measured": True,
                "value": found["psi"],
                "threshold": rule["drift_threshold"],
                "denominator": len(shapes),
                "worst_feature": found["worst_feature"],
                "by_feature": found["by_feature"],
                "text": (
                    f"дрейф распределения запросов: PSI {found['psi']:.4f} по признаку "
                    f"{found['worst_feature']} при пороге {rule['drift_threshold']} на окне "
                    f"{len(shapes)} запросов из {rule['window']} нужных, эталон "
                    f"{len(reference)} замороженных вопросов"
                ),
            },
        )
        signals.insert(
            1,
            {
                "name": "answerless_share",
                "fired": share["share"] > rule["share_threshold"],
                "measured": True,
                "value": share["share"],
                "threshold": rule["share_threshold"],
                "denominator": share["answered"],
                "answerless": share["answerless"],
                "no_hits": share["no_hits"],
                "text": (
                    f"запросов без хорошего ответа {share['answerless']} из "
                    f"{share['answered']} ({share['share']}) при пороге "
                    f"{rule['share_threshold']}: опора лучшего кандидата ниже "
                    f"{rule['support_floor']} по признаку {rule['support_feature']}"
                ),
            },
        )
        report["drift"] = found
        report["answerless"] = share

    report["signals"] = signals
    fired = [signal for signal in signals if signal["fired"]]
    report["alert"] = {
        "fired": bool(fired),
        "signals_fired": [signal["name"] for signal in fired],
        # Текст собирается здесь целиком: issue ночного прогона его печатает, и
        # расходиться числу в issue и числу в отчете негде.
        "text": ". ".join(signal["text"] for signal in fired),
    }
    report["seconds"] = round(time.monotonic() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"alert={'fired' if fired else 'quiet'}\n")

    for signal in signals:
        state = "поднят" if signal["fired"] else ("молчит" if signal.get("measured") else "нет")
        print(f"{signal['name']}: {state} {signal.get('text') or signal.get('note', '')}".strip())
    if fired:
        print(f"алерт мониторинга: {report['alert']['text']}", file=sys.stderr)
        return 1 if args.strict else 0
    print(f"мониторинг: алертов нет, числа в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
