"""Учебная лестница мониторинга: алерты обязаны сработать и обязаны промолчать.

Пороги измерены (`scripts/measure_monitoring_thresholds.py`), а здесь
проверяется другое: что по журналу запросов алерт действительно поднимается и
что он молчит на окне, набранном из той же совокупности, на которой выведены
пороги. Без этой проверки мониторинг держится на честном слове, и мертвый алерт
выглядит точно так же, как работающий.

Откуда берутся события. Живых запросов в журнале ноль, и выдумывать их нельзя.
Поэтому лестница прогоняет через настоящий поиск сервиса замороженные вопросы
набора и пишет события тем же кодом, которым их пишет сервис
(`rufinrag.events`, `EventLog`). Это учебные события, и так они и называются:
живым трафиком они не становятся и в `data/events` не попадают.

Окна подбираются прицельно, потому что проверяется размер сдвига, а не его
правдоподобие. Дрейф набирается долей живых вопросов разъяснений Банка России:
это настоящая другая совокупность вопросов (длиннее, почти всегда с номером
акта), а не испорченный текст. Доля без хорошего ответа набирается долей
запросов, у которых опора лучшего кандидата измеренно ниже порога.

Ступень «короткое окно» проверяет третье: пока запросов меньше окна, правило
обязано спать, а не срабатывать на двух запросах.

Запуск: python scripts/drill_monitoring.py
Стоит 35,3 с на маке: восемь ступеней, в каждой своя проверка подпроцессом.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_queries  # noqa: E402
from rufinrag.events import EventLog, answered, new_request_id, utc_now  # noqa: E402
from rufinrag.metrics import DEPTH  # noqa: E402
from rufinrag.monitoring import answerless, monitoring_rule, shape_of_event  # noqa: E402
from rufinrag.search import SearchService  # noqa: E402

SEED = 20261024
LIVE_ORIGIN = "вопрос из разъяснений Банка России"
SERVICE_VERSION = "drill"
LIVE_SHARES = (0.0, 0.5, 1.0)
ANSWERLESS_SHARES = (0.0, 0.5, 0.75)


def collect(service: SearchService, queries: dict) -> dict[str, dict]:
    """Событие на каждый замороженный вопрос, записанное путем сервиса."""
    out: dict[str, dict] = {}
    for query_id, query in queries.items():
        result = service.search(query.text, top=DEPTH, config="bm25")
        out[query_id] = answered(
            new_request_id(), utc_now(), DEPTH, result, service_version=SERVICE_VERSION
        )
    return out


def write_window(directory: Path, events: list[dict]) -> int:
    if directory.exists():
        shutil.rmtree(directory)
    log = EventLog(directory=directory)
    for event in events:
        log.append(event)
    log.close()
    if log.failed:
        raise SystemExit(f"события учебного окна не записались: {log.last_error}")
    return log.written


def run_check(directory: Path, out: Path) -> tuple[int, dict]:
    done = subprocess.run(
        [
            sys.executable,
            "scripts/check_monitoring.py",
            "--events",
            str(directory),
            "--out",
            str(out),
            "--strict",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return done.returncode, json.loads(out.read_text(encoding="utf-8"))


def signal_of(report: dict, name: str) -> dict:
    for signal in report["signals"]:
        if signal["name"] == name:
            return signal
    return {}


def problems(rungs: list[dict], rule: dict) -> list[str]:
    """Правила, по которым исход лестницы считается сошедшимся.

    Проверка идет в обе стороны: алерт, который кричит всегда, и алерт, который
    молчит всегда, обязаны не проходить одинаково.
    """
    found: list[str] = []
    control = next((rung for rung in rungs if rung["kind"] == "control"), None)
    if control is None:
        found.append("нет ступени из эталонной смеси: алерт, который кричит всегда, не поймать")
    else:
        for name in ("query_drift", "answerless_share"):
            if control["signals"][name]["fired"]:
                found.append(
                    f"ступень из эталонной смеси подняла {name}: значение "
                    f"{control['signals'][name]['value']} при пороге "
                    f"{control['signals'][name]['threshold']}"
                )

    short = next((rung for rung in rungs if rung["kind"] == "short"), None)
    if short is None:
        found.append("нет ступени с коротким окном: спящее правило не проверено")
    else:
        if short["status"] != "dormant":
            found.append(
                f"короткое окно ({short['events']} из {rule['window']}) не усыпило правило: "
                f"статус {short['status']}"
            )
        if short["alert_fired"]:
            found.append("короткое окно подняло алерт: правило обязано спать до полного окна")

    for rung in rungs:
        if rung["status"] != "watching":
            continue
        for name, threshold in (
            ("query_drift", rule["drift_threshold"]),
            ("answerless_share", rule["share_threshold"]),
        ):
            signal = rung["signals"][name]
            above = signal["value"] > threshold
            if above and not signal["fired"]:
                found.append(
                    f"{rung['name']}: {name} {signal['value']} больше порога {threshold}, "
                    "а алерта нет"
                )
            if not above and signal["fired"]:
                found.append(
                    f"{rung['name']}: {name} {signal['value']} в пределах порога {threshold}, "
                    "а алерт сработал"
                )
        if rung["alert_fired"] and rung["exit_code"] == 0:
            found.append(f"{rung['name']}: алерт есть, а проверка вернула ноль")
        if not rung["alert_fired"] and rung["exit_code"] != 0:
            found.append(f"{rung['name']}: алерта нет, а проверка вернула {rung['exit_code']}")

    for name in ("query_drift", "answerless_share"):
        if not any(
            rung["status"] == "watching" and rung["signals"][name]["fired"] for rung in rungs
        ):
            found.append(f"ни одна ступень не подняла {name}: лестница это не проверила")

    for rung in rungs:
        if not rung["alert_fired"]:
            continue
        text = rung["alert_text"]
        for name in ("query_drift", "answerless_share"):
            signal = rung["signals"].get(name, {})
            if not signal.get("fired"):
                continue
            for part in (str(signal["value"]), str(signal["threshold"]), str(rung["events"])):
                if part not in text:
                    found.append(f"{rung['name']}: в тексте алерта нет {part!r}: {text!r}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=ROOT / "logs/monitoring-drill")
    parser.add_argument("--out", type=Path, default=ROOT / "results/monitoring-drill.json")
    args = parser.parse_args()

    started = time.monotonic()
    rule = monitoring_rule(ROOT)
    size = rule["window"]
    queries = load_queries()
    service = SearchService.build(with_dense=False)
    print(
        f"порог дрейфа {rule['drift_threshold']} PSI, порог доли {rule['share_threshold']}, "
        f"окно {size} запросов, корпус {len(service.corpus)} фрагментов",
        flush=True,
    )

    pool = collect(service, queries)
    shapes = {
        query_id: shape_of_event(event, service.bm25.vocab) for query_id, event in pool.items()
    }
    live = [qid for qid, query in queries.items() if query.origin == LIVE_ORIGIN]
    bad = [
        qid
        for qid, shape in shapes.items()
        if answerless(shape, rule["support_feature"], rule["support_floor"])
    ]
    good = [qid for qid in shapes if qid not in set(bad)]
    print(
        f"учебных событий {len(pool)}: живых вопросов {len(live)}, с опорой ниже порога "
        f"{len(bad)}, с опорой выше {len(good)}",
        flush=True,
    )

    rng = random.Random(SEED)
    plan: list[tuple[str, str, list[str]]] = [
        ("эталонная смесь", "control", rng.sample(sorted(shapes), size)),
        ("короткое окно", "short", rng.sample(sorted(shapes), size - 1)),
    ]
    for share in LIVE_SHARES:
        count = round(share * size)
        plan.append(
            (
                f"живых вопросов {count} из {size}",
                "drift",
                rng.sample(live, count) + rng.sample(good, size - count),
            )
        )
    for share in ANSWERLESS_SHARES:
        count = round(share * size)
        plan.append(
            (
                f"без опоры {count} из {size}",
                "answerless",
                rng.sample(bad, count) + rng.sample(good, size - count),
            )
        )

    rungs: list[dict] = []
    for number, (name, kind, ids) in enumerate(plan, start=1):
        t0 = time.monotonic()
        directory = args.work / f"rung-{number}"
        written = write_window(directory, [pool[qid] for qid in ids])
        code, report = run_check(directory, args.work / f"monitoring-{number}.json")
        rungs.append(
            {
                "name": name,
                "kind": kind,
                "events": written,
                "live_in_window": sum(1 for qid in ids if queries[qid].origin == LIVE_ORIGIN),
                "status": report["status"],
                "exit_code": code,
                "alert_fired": report["alert"]["fired"],
                "alert_text": report["alert"]["text"],
                "signals": {
                    name: {
                        "fired": signal.get("fired", False),
                        "value": signal.get("value"),
                        "threshold": signal.get("threshold"),
                        "worst_feature": signal.get("worst_feature"),
                    }
                    for name, signal in (
                        (key, signal_of(report, key))
                        for key in ("query_drift", "answerless_share", "benchmark_regression")
                    )
                },
                "seconds": round(time.monotonic() - t0, 1),
            }
        )
        last = rungs[-1]
        print(
            f"{name}: событий {written}, статус {last['status']}, PSI "
            f"{last['signals']['query_drift']['value']}, доля без ответа "
            f"{last['signals']['answerless_share']['value']}, код {code}, алерт "
            f"{'поднят' if last['alert_fired'] else 'молчит'}, {last['seconds']} с",
            flush=True,
        )

    found = problems(rungs, rule)
    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "seed": SEED,
        "rule": rule,
        "events_source": (
            "замороженные вопросы набора, прогнанные через поиск сервиса и записанные кодом "
            "журнала. Это учебные события, не живой трафик"
        ),
        "pool": {
            "events": len(pool),
            "live_queries": len(live),
            "support_below_floor": len(bad),
            "support_above_floor": len(good),
        },
        "steps": len(rungs),
        "first_drift_alert": next(
            (
                rung["signals"]["query_drift"]["value"]
                for rung in rungs
                if rung["signals"]["query_drift"]["fired"]
            ),
            None,
        ),
        "largest_silent_drift": max(
            (
                rung["signals"]["query_drift"]["value"]
                for rung in rungs
                if rung["status"] == "watching" and not rung["signals"]["query_drift"]["fired"]
            ),
            default=None,
        ),
        "first_share_alert": next(
            (
                rung["signals"]["answerless_share"]["value"]
                for rung in rungs
                if rung["signals"]["answerless_share"]["fired"]
            ),
            None,
        ),
        "largest_silent_share": max(
            (
                rung["signals"]["answerless_share"]["value"]
                for rung in rungs
                if rung["status"] == "watching" and not rung["signals"]["answerless_share"]["fired"]
            ),
            default=None,
        ),
        "rungs": rungs,
        "seconds": round(time.monotonic() - started, 1),
        "passed": not found,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if found:
        print("учебная лестница мониторинга разошлась с ожиданием:", file=sys.stderr)
        for line in found:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(
        f"дрейф: алерт поднят на PSI {report['first_drift_alert']}, молчал до "
        f"{report['largest_silent_drift']} при пороге {rule['drift_threshold']}. Доля без "
        f"ответа: алерт поднят на {report['first_share_alert']}, молчал до "
        f"{report['largest_silent_share']} при пороге {rule['share_threshold']}. Числа в "
        f"{args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
