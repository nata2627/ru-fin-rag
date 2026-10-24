"""Пороги мониторинга трафика, выведенные из измеренного разброса на эталоне.

Два порога: дрейф распределения запросов и доля запросов без хорошего ответа.
Правила записаны до чисел (`orchestration/EXPERIMENTS.md`, S22) и применяются
здесь без правок.

Откуда берется разброс. Живых запросов в журнале сервиса ноль, и на нуле порог
не выводится. Но мера и ее шум это свойство статистики и размера окна, а не
трафика: окно размера `n`, набранное из одной и той же совокупности, дает PSI
не ноль, а некоторое распределение, и его потолок измерим без всякого трафика.
Поэтому шум считается по 200 окнам, набранным из 1182 замороженных вопросов
набора, а порог берется с двойным запасом от потолка шума.

Порог, который ничего не ловит, бесполезен, поэтому у правила есть потолок, и он
тоже измерен. Настоящим сдвигом считается PSI окна, набранного из 325 живых
вопросов разъяснений Банка России, против того же эталона: это две
по-настоящему разные совокупности вопросов (живой вопрос длиннее, почти всегда
называет номер акта), и мера, которая их не различает, не различит ничего.
Порог обязан быть меньше половины медианы этого сдвига, иначе замер не прошел.

Что здесь не делается. Дрейф не считается на синтетике и не выдается за живой:
окна выше это окна эталонного набора, и так они и названы в отчете. Числа про
живой трафик в отчете одно: сколько его в журнале (ноль).

Признак опоры («хороший ответ») выбирается из двух кандидатов по J Юдена на 991
размеченном вопросе, оба числа публикуются. Выдача берется прямо у индекса
BM25: что это ровно тот же путь, которым идет связка `bm25` сервиса, измерено
замером порога деградации (`results/alert-threshold.json`, поле
`same_path_as_service`, 991 выдача из 991).

Запуск: python scripts/measure_monitoring_thresholds.py [--draws 200]
Стоит 7,0 с на маке, не считая запуска интерпретатора.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_qrels, load_queries  # noqa: E402
from rufinrag.config import data_dir  # noqa: E402
from rufinrag.metrics import DEPTH  # noqa: E402
from rufinrag.monitoring import (  # noqa: E402
    FEATURES,
    SUPPORT_FEATURES,
    QueryShape,
    answerless_share,
    counts_of,
    drift,
    load_index,
    shape_of_text,
    token_edges,
    youden,
)

SEED = 20261024
# Запас два, а не пять, как у порога деградации: ложная тревога здесь не красит
# плановый прогон (issue открывается при зеленом прогоне, как при отказе
# cbr.ru), а пропущенный дрейф дороже лишнего issue.
SAFETY = 2.0
DRIFT_STEP = 0.05
SHARE_STEP = 0.05
WINDOWS = (50, 100, 200)
LIVE_ORIGIN = "вопрос из разъяснений Банка России"


def round_up(value: float, step: float) -> float:
    return max(step, math.ceil(value / step) * step)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "results/monitoring-thresholds.json")
    args = parser.parse_args()

    started = time.monotonic()
    queries = load_queries()
    qrels = load_qrels()
    index = load_index()
    print(
        f"эталон {len(queries)} вопросов, словарь индекса {len(index.vocab)}, "
        f"фрагментов {len(index.ids)}",
        flush=True,
    )

    shapes: dict[str, QueryShape] = {}
    good: dict[str, bool] = {}
    for query_id, query in queries.items():
        hits = index.search(query.text, k=DEPTH)
        shapes[query_id] = shape_of_text(
            query.text, index.vocab, hits=len(hits), top_score=hits[0][1] if hits else None
        )
        rel = qrels.get(query_id, {})
        if any(grade > 0 for grade in rel.values()):
            good[query_id] = any(rel.get(chunk_id, 0) > 0 for chunk_id, _ in hits)
    print(
        f"выдача посчитана: размеченных вопросов {len(good)}, из них с релевантным "
        f"фрагментом в первой десятке {sum(good.values())}",
        flush=True,
    )

    edges = token_edges(list(shapes.values()))
    reference = list(shapes.values())
    live_ids = [qid for qid, query in queries.items() if query.origin == LIVE_ORIGIN]

    # Признак опоры: сравниваются оба кандидата на одних и тех же вопросах.
    judged = sorted(good)
    labels = [good[qid] for qid in judged]
    candidates = {
        feature: youden([shapes[qid].support(feature) for qid in judged], labels)
        for feature in SUPPORT_FEATURES
    }
    support = max(SUPPORT_FEATURES, key=lambda name: candidates[name]["j"])
    floor = candidates[support]["floor"]
    for name, found in candidates.items():
        print(
            f"признак {name}: J {found['j']}, порог {found['floor']:.4f}, "
            f"с ответом ниже порога {found['good_below_floor']} из {found['positives']}, "
            f"без ответа выше порога {found['bad_above_floor']} из {found['negatives']}",
            flush=True,
        )
    print(f"выбран признак {support} с порогом опоры {floor:.4f}", flush=True)

    base = answerless_share(reference, support, floor)
    by_origin = {
        origin: answerless_share(
            [shapes[qid] for qid, query in queries.items() if query.origin == origin],
            support,
            floor,
        )
        for origin in sorted({query.origin for query in queries.values()})
    }

    rng = random.Random(SEED)
    ladder: list[dict] = []
    for size in WINDOWS:
        noise = []
        noise_shares = []
        for _ in range(args.draws):
            window = rng.sample(reference, size)
            noise.append(drift(window, reference, edges)["psi"])
            noise_shares.append(answerless_share(window, support, floor)["share"])
        real = []
        real_shares = []
        real_by_feature = {feature: [] for feature in FEATURES}
        for _ in range(args.draws):
            window = [shapes[qid] for qid in rng.sample(live_ids, min(size, len(live_ids)))]
            found = drift(window, reference, edges)
            real.append(found["psi"])
            real_shares.append(answerless_share(window, support, floor)["share"])
            for feature, value in found["by_feature"].items():
                real_by_feature[feature].append(value)
        ceiling = max(noise)
        threshold = round_up(SAFETY * ceiling, DRIFT_STEP)
        median_real = statistics.median(real)
        cap = median_real / 2
        row = {
            "window": size,
            "draws": args.draws,
            "noise_psi_max": round(ceiling, 4),
            "noise_psi_median": round(statistics.median(noise), 4),
            "noise_share_max": round(max(noise_shares), 4),
            "noise_share_median": round(statistics.median(noise_shares), 4),
            "live_window": min(size, len(live_ids)),
            # Доля без хорошего ответа на окнах из живых вопросов, для сравнения с
            # окнами эталонной смеси: порог берется по правилу из вторых, а чего
            # это стоит, видно по первым.
            "live_share_max": round(max(real_shares), 4),
            "live_share_median": round(statistics.median(real_shares), 4),
            "real_psi_median": round(median_real, 4),
            "real_psi_min": round(min(real), 4),
            "real_psi_by_feature_median": {
                feature: round(statistics.median(values), 4)
                for feature, values in real_by_feature.items()
            },
            "drift_threshold": round(threshold, 4),
            "cap": round(cap, 4),
            "share_threshold": round(round_up(max(noise_shares), SHARE_STEP), 4),
            "fits": threshold < cap,
        }
        ladder.append(row)
        print(
            f"окно {size}: шум PSI до {ceiling:.4f}, порог {threshold:.2f}, настоящий сдвиг "
            f"медианой {median_real:.4f} при потолке {cap:.4f}, доля без ответа в окнах шума до "
            f"{max(noise_shares):.4f}, правило {'укладывается' if row['fits'] else 'не лезет'}",
            flush=True,
        )

    chosen = next((row for row in ladder if row["fits"]), None)
    journal = data_dir() / "events"
    live_lines = sum(
        1
        for path in sorted(journal.glob("queries-*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "config": "bm25",
        "corpus_chunks": len(index.ids),
        "vocab": len(index.vocab),
        "reference": {
            "queries": len(reference),
            "judged": len(good),
            "judged_with_hit": sum(good.values()),
            "live_queries": len(live_ids),
            "live_origin": LIVE_ORIGIN,
            "note": (
                "это замороженный набор вопросов, а не живой трафик. Окна шума и окна "
                "настоящего сдвига набраны из него"
            ),
        },
        "live_journal_lines": live_lines,
        "rule": {
            "safety": SAFETY,
            "drift_step": DRIFT_STEP,
            "share_step": SHARE_STEP,
            "seed": SEED,
            "draws": args.draws,
            "windows": list(WINDOWS),
            "text": (
                "порог дрейфа это потолок PSI по 200 окнам из эталона, умноженный на два и "
                "округленный вверх до шага 0,05, но не меньше шага; он обязан быть меньше "
                "половины медианы PSI окон из живых вопросов против эталона. Порог доли без "
                "хорошего ответа это наибольшая доля среди тех же окон шума, округленная "
                "вверх до шага 0,05. Размер окна берется наименьшим из 50, 100, 200, при "
                "котором правило укладывается в потолок"
            ),
            "written_before": "orchestration/EXPERIMENTS.md, S22",
        },
        "features": {
            "drift": list(FEATURES),
            "token_bin_edges": edges,
            "support_candidates": candidates,
            "support_chosen": support,
            "support_floor": floor,
        },
        "answerless_on_reference": {**base, "by_origin": by_origin},
        "psi_of_live_against_reference": drift([shapes[qid] for qid in live_ids], reference, edges),
        "reference_counts": counts_of(reference, edges),
        "ladder": ladder,
        "seconds": round(time.monotonic() - started, 1),
    }

    if chosen is None:
        report["passed"] = False
        report["rule_applied"] = None
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "ни один размер окна не уложился в потолок: на таком окне дрейф измерять нечем",
            file=sys.stderr,
        )
        return 1

    report["passed"] = True
    report["rule_applied"] = {
        "measured_at": report["measured_at"],
        "window": chosen["window"],
        "drift_threshold": chosen["drift_threshold"],
        "share_threshold": chosen["share_threshold"],
        "support_feature": support,
        "support_floor": floor,
        "token_bin_edges": edges,
        "noise_psi_max": chosen["noise_psi_max"],
        "real_psi_median": chosen["real_psi_median"],
        "reference_queries": len(reference),
        "base_share": base["share"],
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"окно {chosen['window']} запросов, порог дрейфа {chosen['drift_threshold']:.2f} PSI, "
        f"порог доли без хорошего ответа {chosen['share_threshold']:.2f} при базовой доле "
        f"{base['share']} ({base['answerless']} из {base['answered']}), числа в {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
