"""Учебная деградация метрики: проверка, что алерт срабатывает и не врет.

Порог алерта измерен (`scripts/measure_alert_threshold.py`,
`results/alert-threshold.json`), и прогон регрессии его читает. Здесь
проверяется другое: что при падении метрики алерт действительно поднимается, а
при падении в пределах шума молчит. Без этой проверки мониторинг держится на
честном слове, и мертвый алерт выглядит точно так же, как работающий.

Деградация делается прицельно: из корпуса убирается доля фрагментов, которые в
разметке отмечены как ответ на вопрос. Прицельно, потому что нужен сдвиг
заданного размера, а не правдоподобная причина: размер здесь и есть
проверяемое. Правдоподобные причины падения метрики в проекте уже измерены
(стеммер, разбор номера акта, доливка корпуса), и они дают от 0,003 до 0,084.

Лестница из долей идет по возрастанию, и ожидание записано до прогона:

1. Нулевая ступень это тот же корпус, пересобранный индекс и тот же эталон.
   Падения нет, алерта нет. Ступень ловит алерт, который кричит всегда.
2. Любая ступень с падением больше порога обязана поднять алерт: прогон
   регрессии возвращает ненулевой код, в отчете `alert.fired` и текст с числом.
3. Любая ступень с падением в пределах порога обязана молчать.
4. Хотя бы одна ступень обязана поднять алерт, иначе лестница ничего не
   проверила.
5. Текст алерта обязан называть число, порог и знаменатель: issue ночного
   прогона печатает именно его, и «метрика упала» без числа это не алерт.

Алерт идет тем же путем, что ночью: `scripts/run_regression.py` на поврежденном
корпусе, подпроцессом и с теми же переменными окружения. Поврежденный корпус
лежит во временной папке, рабочий индекс и рабочие отчеты прогон не трогает.

Запуск: python scripts/check_alert.py [--shares 0 0.01 0.02 0.04 0.08]
Стоит около 40 с на ступень.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_qrels  # noqa: E402
from rufinrag.corpus import CORPUS_FILE, corpus_path, load_corpus  # noqa: E402
from rufinrag.pipeline import alert_threshold  # noqa: E402
from rufinrag.retrieval.bm25 import BM25Index  # noqa: E402
from rufinrag.search import TEXTS_FILE  # noqa: E402

WATCHED = "NDCG@10"
SEED = 20261024
SHARES = (0.0, 0.01, 0.02, 0.04, 0.08)


def relevant_chunks(qrels: dict[str, dict[str, int]]) -> list[str]:
    found = {chunk_id for grades in qrels.values() for chunk_id, grade in grades.items() if grade}
    return sorted(found)


def damage(source: Path, target: Path, removed: set[str]) -> int:
    """Поврежденный корпус: те же строки минус убранные фрагменты.

    Строки переписываются как есть, без разбора и сборки заново: иначе разница
    между рабочим и поврежденным корпусом включала бы еще и мою сериализацию.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with (
        gzip.open(source, "rt", encoding="utf-8") as src,
        gzip.open(target, "wt", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                continue
            if json.loads(line)["chunk_id"] in removed:
                continue
            dst.write(line)
            kept += 1
    return kept


def build_index(data: Path) -> dict:
    corpus = load_corpus(corpus_path(data))
    index = BM25Index.build(corpus.ids, list(corpus.iter_texts()))
    index.save(data / "index/bm25.npz")
    corpus.texts.save(data / TEXTS_FILE)
    return {"chunks": len(corpus), "vocab": len(index.vocab)}


def run_regression(data: Path, out: Path) -> tuple[int, dict, str]:
    env = {
        **os.environ,
        "RUFINRAG_DATA_DIR": str(data),
        "RUFINRAG_DISABLE_DENSE": "1",
    }
    done = subprocess.run(
        [sys.executable, "scripts/run_regression.py", "--config", "bm25", "--out", str(out)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return done.returncode, json.loads(out.read_text(encoding="utf-8")), done.stderr.strip()


def problems(rungs: list[dict], threshold: float) -> list[str]:
    found = []
    control = rungs[0]
    if control["share"] != 0.0:
        found.append("нулевой ступени нет: алерт, который кричит всегда, лестница не поймает")
    elif control["alert_fired"]:
        found.append(
            f"алерт сработал на целом корпусе: падение {control['drop']} при пороге {threshold}"
        )
    for rung in rungs:
        above = rung["drop"] > threshold
        if above and not rung["alert_fired"]:
            found.append(
                f"доля {rung['share']}: падение {rung['drop']} больше порога {threshold}, "
                "а алерта нет"
            )
        if not above and rung["alert_fired"]:
            found.append(
                f"доля {rung['share']}: падение {rung['drop']} в пределах порога {threshold}, "
                "а алерт сработал"
            )
        if above and rung["exit_code"] == 0:
            found.append(f"доля {rung['share']}: алерт есть, а прогон регрессии зеленый")
    fired = [rung for rung in rungs if rung["alert_fired"]]
    if not fired:
        found.append(
            f"ни одна ступень не подняла алерт: наибольшее падение "
            f"{max(rung['drop'] for rung in rungs)} при пороге {threshold}"
        )
    for rung in fired:
        text = rung["alert_text"]
        for part in (f"{rung['drop']:.4f}", str(threshold), str(rung["queries_scored"])):
            if part not in text:
                found.append(f"доля {rung['share']}: в тексте алерта нет {part!r}: {text!r}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shares", type=float, nargs="*", default=list(SHARES))
    parser.add_argument("--work", type=Path, default=ROOT / "logs/alert-drill")
    parser.add_argument("--out", type=Path, default=ROOT / "results/alert-drill.json")
    args = parser.parse_args()

    started = time.monotonic()
    threshold = alert_threshold(ROOT)
    qrels = load_qrels()
    relevant = relevant_chunks(qrels)
    source = corpus_path()
    baseline = json.loads((ROOT / "results/regression-baseline.json").read_text(encoding="utf-8"))
    print(
        f"порог {threshold} из results/alert-threshold.json, эталон "
        f"{baseline['metrics'][WATCHED]} от {baseline['measured_at']}, "
        f"размеченных как ответ фрагментов {len(relevant)}",
        flush=True,
    )

    data = args.work / "data"
    data.mkdir(parents=True, exist_ok=True)
    link = data / "benchmark"
    if not link.exists():
        link.symlink_to(ROOT / "data/benchmark")

    # Порядок один на всю лестницу: ступени вложены друг в друга, и разница
    # между ними это только размер потери, а не другой набор фрагментов.
    order = random.Random(SEED).sample(relevant, len(relevant))

    rungs: list[dict] = []
    for share in sorted(args.shares):
        t0 = time.monotonic()
        count = round(share * len(relevant))
        removed = set(order[:count])
        kept = damage(source, data / CORPUS_FILE, removed)
        index = build_index(data)
        code, report, stderr = run_regression(data, args.work / f"regression-{share}.json")
        alert = report["alert"]
        rungs.append(
            {
                "share": share,
                "relevant_removed": count,
                "relevant_total": len(relevant),
                "chunks_kept": kept,
                "vocab": index["vocab"],
                WATCHED: report["metrics"][WATCHED],
                "drop": report["baseline"]["drop"],
                "queries_scored": report["queries_scored"],
                "exit_code": code,
                "alert_fired": alert["fired"],
                "alert_text": alert["text"],
                "stderr": stderr,
                "seconds": round(time.monotonic() - t0, 1),
            }
        )
        print(
            f"доля {share}: убрано размеченных {count} из {len(relevant)}, корпус {kept}, "
            f"{WATCHED} {report['metrics'][WATCHED]:.4f}, падение {report['baseline']['drop']}, "
            f"код {code}, алерт {'поднят' if alert['fired'] else 'молчит'}, "
            f"{time.monotonic() - t0:.1f} с",
            flush=True,
        )

    found = problems(rungs, threshold)
    fired = [rung for rung in rungs if rung["alert_fired"]]
    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "metric": WATCHED,
        "config": "bm25",
        "threshold": threshold,
        "threshold_from": "results/alert-threshold.json",
        "baseline": {
            "measured_at": baseline["measured_at"],
            WATCHED: baseline["metrics"][WATCHED],
        },
        "relevant_chunks": len(relevant),
        "seed": SEED,
        "steps": len(rungs),
        "first_fired_share": fired[0]["share"] if fired else None,
        "first_fired_drop": fired[0]["drop"] if fired else None,
        "silent_steps": [rung["share"] for rung in rungs if not rung["alert_fired"]],
        "largest_silent_drop": max(
            (rung["drop"] for rung in rungs if not rung["alert_fired"]), default=None
        ),
        "rungs": rungs,
        "seconds": round(time.monotonic() - started, 1),
        "passed": not found,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if found:
        print("учебная деградация разошлась с ожиданием:", file=sys.stderr)
        for line in found:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(
        f"алерт поднят на доле {report['first_fired_share']} (падение "
        f"{report['first_fired_drop']} при пороге {threshold}), молчал на "
        f"{len(report['silent_steps'])} ступенях из {len(rungs)}, наибольшее "
        f"падение без алерта {report['largest_silent_drop']}, числа в {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
