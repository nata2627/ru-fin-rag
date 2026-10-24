"""Нужный размер выборки и правило остановки для сравнения двух конфигураций поиска.

Правило записано до того, как посчитан хоть один замер сравнения, и этот файл
есть его хранимая копия: папка рабочих документов в репозиторий не едет, поэтому
единственная проверяемая запись правила это код и его результат в git. Коммит
этого файла стоит раньше любого замера задачи 6.2, и это видно по истории.

Зачем вообще считать размер выборки заранее. Без него момент остановки выбирает
тот, кто смотрит на числа, и выбирает его там, где картинка нравится. Сорок
прогонов доливки в `results/alert-threshold.json` уже показали, что метрика
шевелится сама по себе на 0,0037, и на такой помехе «посмотрели, понравилось,
остановились» дает значимый результат из ничего.

Что объявлено до чисел и в этом файле не меняется:

- уровень значимости 0,05 двусторонний, мощность 0,80;
- офлайн обязаны увидеть сдвиг 0,020 NDCG@10, это порог алерта из S16, то есть
  наименьший сдвиг, который проект уже называет настоящим;
- онлайн обязаны увидеть долю выигранных запросов 0,55 против 0,50; пересчета
  сдвига NDCG@10 в долю выигранных запросов в проекте нет, поэтому онлайн сдвиг
  объявлен в своих единицах, а не выведен из офлайн;
- взглядов не больше четырнадцати, по одному в сутки ночным прогоном;
- взгляд на выборке меньше тридцати информативных запросов пропускается, а не
  считается с поправкой: нормальное приближение там не работает;
- поправка на подглядывание это постоянная граница Покока вместо 1,96, она
  подбирается моделированием под те же 0,05 на все взгляды вместе.

Что считается здесь и чего здесь сознательно нет. Здесь считаются знаменатели и
разбросы: сколько запросов нужно, какую долю запросов интерливинг вообще делает
информативной, какой сдвиг различим на имеющихся 991. Кто из двух конфигураций
выигрывает, здесь не считается: в интерливинге берется только модуль перевеса,
знак нигде не сохраняется, и победителя объявляет задача 6.2 после этого
коммита. Иначе правило остановки писалось бы с подсмотренным ответом.

Запуск: python scripts/plan_sample_size.py [--simulations 20000] [--seeds 20]
Стоит около минуты на маке, карта и сеть не нужны.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from statistics import NormalDist, mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_qrels, load_queries  # noqa: E402
from rufinrag.corpus import read_jsonl  # noqa: E402
from rufinrag.metrics import DEPTH, per_query  # noqa: E402

WATCHED = "NDCG@10"
SEED = 20261024
RULE_WRITTEN_AT = "2026-10-24"

ALPHA = 0.05
POWER = 0.80
LOOKS = 14
MIN_LOOK = 30
OFFLINE_TARGET = 0.020
OFFLINE_GRID = (0.005, 0.010, 0.020, 0.040)
ONLINE_TARGET = 0.05
ONLINE_GRID = (0.02, 0.05, 0.10)
TRAFFIC_RATES = (1, 5, 10, 50)

RUNS = {
    "bm25": "data/benchmark/run-bm25.jsonl",
    "dense": "data/benchmark/run-dense.jsonl",
    "hybrid": "data/benchmark/run-hybrid.jsonl",
    "bm25-rerank": "results/rerank-bm25-rufin.jsonl",
    "hybrid-rerank": "results/rerank-hybrid-rufin.jsonl",
}
# Связки с переранжированием стоят 48,05 с на запрос на CPU мака
# (results/artifacts.md), в бесплатный тариф такой ответ не влезает.
SERVABLE = ("bm25", "dense", "hybrid")
CANDIDATE_PAIR = ("bm25", "hybrid")
PAIRS = (
    ("bm25", "hybrid"),
    ("bm25", "dense"),
    ("hybrid", "dense"),
    ("bm25", "bm25-rerank"),
    ("hybrid", "hybrid-rerank"),
    ("bm25-rerank", "hybrid-rerank"),
)


def z(p: float) -> float:
    return NormalDist().inv_cdf(p)


def load_run_file(path: str) -> dict[str, list[str]]:
    return {row["query_id"]: list(row["ranked"]) for row in read_jsonl(ROOT / path)}


def merge_team_draft(
    left: list[str], right: list[str], rng: random.Random, depth: int = DEPTH
) -> tuple[list[str], list[int]]:
    """Слияние двух выдач интерливингом команд: список и хозяин каждой позиции.

    Правило из исходной работы: следующую позицию берет та команда, у которой
    позиций меньше, а при равенстве монета. Поэтому ни одна конфигурация не
    получает систематического преимущества в начале списка, и предпочтение
    нельзя будет объяснить тем, кто стоял первым.
    """
    merged: list[str] = []
    owners: list[int] = []
    taken: set[str] = set()
    counts = [0, 0]
    cursors = [0, 0]
    lists = (left, right)
    while len(merged) < depth:
        for side in (0, 1):
            while cursors[side] < len(lists[side]) and lists[side][cursors[side]] in taken:
                cursors[side] += 1
        has = [cursors[side] < len(lists[side]) for side in (0, 1)]
        if not any(has):
            break
        if not (has[0] and has[1]):
            side = 0 if has[0] else 1
        elif counts[0] < counts[1]:
            side = 0
        elif counts[1] < counts[0]:
            side = 1
        else:
            side = 0 if rng.random() < 0.5 else 1
        chunk_id = lists[side][cursors[side]]
        taken.add(chunk_id)
        merged.append(chunk_id)
        owners.append(side)
        counts[side] += 1
    return merged, owners


def interleaving_margin(
    left: list[str], right: list[str], rel: dict[str, int], rng: random.Random
) -> int:
    """Модуль перевеса по одному запросу. Знак не возвращается нарочно.

    За клик принято попадание размеченного релевантного фрагмента в первую
    десятку слитого списка. Это допущение офлайн интерливинга: предполагается,
    что пользователь нажал бы ровно на то, что разметка считает релевантным.
    На живых людях оно не проверено, потому что живых людей у сервиса нет.
    """
    merged, owners = merge_team_draft(left, right, rng)
    credit = [0, 0]
    for chunk_id, side in zip(merged, owners, strict=True):
        if rel.get(chunk_id, 0) > 0:
            credit[side] += 1
    return abs(credit[0] - credit[1])


def informative_share(
    left: dict[str, list[str]],
    right: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    ids: list[str],
    seeds: int,
) -> dict[str, float]:
    """Доля запросов, на которых интерливинг вообще дает перевес.

    Это знаменатель мощности онлайн теста: запрос с нулевым перевесом в знаковый
    тест не входит, и считать его в размер выборки нельзя.
    """
    shares: list[float] = []
    margins: list[int] = []
    for seed in range(seeds):
        rng = random.Random(SEED + seed)
        hits = 0
        for query_id in ids:
            margin = interleaving_margin(
                left.get(query_id, []), right.get(query_id, []), qrels[query_id], rng
            )
            margins.append(margin)
            if margin > 0:
                hits += 1
        shares.append(hits / len(ids))
    return {
        "mean": round(mean(shares), 4),
        "min": round(min(shares), 4),
        "max": round(max(shares), 4),
        "seeds": seeds,
        "queries": len(ids),
        "mean_margin_when_informative": round(
            mean([m for m in margins if m > 0]) if any(margins) else 0.0, 3
        ),
    }


def offline_n(sd: float, effect: float) -> int:
    """Парный тест: сколько запросов нужно, чтобы увидеть сдвиг заданного размера."""
    return math.ceil(((z(1 - ALPHA / 2) + z(POWER)) * sd / effect) ** 2)


def offline_mde(sd: float, n: int) -> float:
    return (z(1 - ALPHA / 2) + z(POWER)) * sd / math.sqrt(n)


def online_fixed_n(excess: float) -> int:
    """Знаковый тест на долю выигранных запросов, один взгляд, без поправки."""
    p = 0.5 + excess
    numerator = z(1 - ALPHA / 2) * 0.5 + z(POWER) * math.sqrt(p * (1 - p))
    return math.ceil((numerator / excess) ** 2)


def look_sizes(n_max: int, looks: int) -> list[int]:
    return [max(1, round(n_max * k / looks)) for k in range(1, looks + 1)]


def simulate_looks(ns: list[int], p: float, simulations: int, seed: int) -> np.ndarray:
    """Статистика знакового теста на каждом взгляде для множества прогонов."""
    rng = np.random.default_rng(seed)
    sizes = np.asarray(ns, dtype=float)
    increments = np.diff(np.concatenate(([0.0], sizes))).astype(int)
    wins = rng.binomial(increments, p, size=(simulations, len(ns))).cumsum(axis=1)
    return (wins - sizes / 2) / np.sqrt(sizes / 4)


def pocock_boundary(ns: list[int], simulations: int, seed: int) -> float:
    """Постоянная граница, при которой все взгляды вместе дают ровно ALPHA.

    Граница подбирается моделированием, а не берется из таблицы: таблица
    предполагает взгляды равного шага и все взгляды состоявшимися, а здесь
    взгляды с выборкой меньше MIN_LOOK пропускаются, и шаг получается не тот.

    Берется не квантиль, а наименьшее наблюденное значение, у которого хвост
    уже не толще ALPHA. При сотне запросов на взгляд статистика принимает мало
    различных значений, квантиль садится ровно на такое значение, и граница
    держит не 0,05, а заметно больше: проверка на независимом зерне показывала
    0,057 вместо 0,05.
    """
    statistics = np.abs(simulate_looks(ns, 0.5, simulations, seed))
    usable = [i for i, n in enumerate(ns) if n >= MIN_LOOK]
    if not usable:
        return float("inf")
    worst = np.sort(statistics[:, usable].max(axis=1))
    candidates = np.unique(worst)
    tail = 1.0 - np.searchsorted(worst, candidates, side="left") / worst.size
    safe = candidates[tail <= ALPHA]
    return float(safe[0]) if safe.size else float("inf")


def run_sequential(
    ns: list[int], p: float, boundary: float, simulations: int, seed: int
) -> dict[str, float]:
    statistics = np.abs(simulate_looks(ns, p, simulations, seed))
    usable = [i for i, n in enumerate(ns) if n >= MIN_LOOK]
    crossed = statistics[:, usable] >= boundary
    rejected = crossed.any(axis=1)
    first = np.where(rejected, crossed.argmax(axis=1), len(usable) - 1)
    stop_n = np.asarray([ns[usable[i]] for i in first], dtype=float)
    return {
        "share": float(rejected.mean()),
        "expected_n": float(stop_n.mean()),
    }


def sequential_plan(excess: float, simulations: int) -> dict[str, object]:
    """Какой n_max нужен последовательному тесту, чтобы мощность осталась 0,80.

    Граница пересчитывается под каждый кандидат n_max, потому что расписание
    взглядов и граница объявляются вместе: граница, посчитанная для другого
    расписания, дала бы не ту вероятность ложного срабатывания.
    """
    fixed = online_fixed_n(excess)
    low, high = fixed, fixed * 4
    best: dict[str, object] | None = None
    for _ in range(12):
        if high - low <= LOOKS:
            break
        middle = (low + high) // 2
        ns = look_sizes(middle, LOOKS)
        boundary = pocock_boundary(ns, simulations, SEED)
        power = run_sequential(ns, 0.5 + excess, boundary, simulations, SEED + 1)
        if power["share"] >= POWER:
            high = middle
            best = {"n_max": middle, "boundary": boundary, **power}
        else:
            low = middle
    if best is None:
        ns = look_sizes(high, LOOKS)
        boundary = pocock_boundary(ns, simulations, SEED)
        power = run_sequential(ns, 0.5 + excess, boundary, simulations, SEED + 1)
        best = {"n_max": high, "boundary": boundary, **power}
    ns = look_sizes(int(best["n_max"]), LOOKS)
    boundary = float(best["boundary"])
    false_positive = run_sequential(ns, 0.5, boundary, simulations, SEED + 2)
    return {
        "excess": excess,
        "win_rate": round(0.5 + excess, 3),
        "fixed_sample_n": fixed,
        "n_max": int(best["n_max"]),
        "inflation": round(int(best["n_max"]) / fixed, 3),
        "boundary_z": round(boundary, 3),
        "boundary_z_without_correction": round(z(1 - ALPHA / 2), 3),
        "looks_used": sum(1 for n in ns if n >= MIN_LOOK),
        "looks_skipped_as_small": sum(1 for n in ns if n < MIN_LOOK),
        "look_sizes": ns,
        "power_simulated": round(float(best["share"]), 4),
        "expected_n_under_effect": round(float(best["expected_n"]), 1),
        "false_positive_simulated": round(false_positive["share"], 4),
        "simulations": simulations,
    }


def live_traffic() -> dict[str, object]:
    """Сколько живого трафика реально есть. Считается по диску, а не по отчету."""
    events = json.loads((ROOT / "results/events.json").read_text(encoding="utf-8"))
    files = sorted(p.name for p in (ROOT / "data/events").glob("queries-*.jsonl"))
    lines = 0
    for name in files:
        with open(ROOT / "data/events" / name, encoding="utf-8") as handle:
            lines += sum(1 for _ in handle)
    load = json.loads((ROOT / "results/load.json").read_text(encoding="utf-8"))
    synthetic = max(run["endpoints"]["/search"]["requests"] for run in load["runs"])
    return {
        "source": "data/events, results/events.json",
        "files_on_disk": len(files),
        "lines_on_disk": lines,
        "query_events_reported": events["repo"]["query_events"],
        "days_reported": events["repo"]["days"],
        "drains": events["drains"],
        "rehearsal_queries": 25,
        "rehearsal_note": (
            "репетиция выгрузки на локальном процессе, results/events-rehearsal.json, "
            "трафиком не является"
        ),
        "synthetic_load_requests": synthetic,
        "synthetic_load_note": (
            "запросы нагрузочного замера подавал locust, results/load.json, трафиком не является"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", default="results/sample-size.json")
    args = parser.parse_args()

    started = time.time()
    qrels = load_qrels()
    queries = load_queries()
    runs = {name: load_run_file(path) for name, path in RUNS.items()}

    scored_ids, _ = per_query(runs["bm25"], qrels, WATCHED)
    values = {
        name: dict(zip(*per_query(run, qrels, WATCHED), strict=True)) for name, run in runs.items()
    }
    subsamples = {
        "все": scored_ids,
        # Подвыборка та же, что во всех таблицах проекта: 142 вопроса из
        # разъяснений Банка России. Три ручных вопроса сюда не входят, иначе
        # знаменатель разойдется с остальными числами.
        "живые": [i for i in scored_ids if queries[i].podvyborka == "живые"],
    }

    baseline = json.loads((ROOT / "results/regression-baseline.json").read_text(encoding="utf-8"))
    threshold = json.loads((ROOT / "results/alert-threshold.json").read_text(encoding="utf-8"))

    pairs: list[dict[str, object]] = []
    for left, right in PAIRS:
        entry: dict[str, object] = {
            "pair": f"{left} / {right}",
            "online_servable": left in SERVABLE and right in SERVABLE,
            "candidate_for_6_2": (left, right) == CANDIDATE_PAIR,
            "subsamples": {},
        }
        for label, ids in subsamples.items():
            diffs = [values[right][i] - values[left][i] for i in ids]
            spread = stdev(diffs)
            share = informative_share(runs[left], runs[right], qrels, ids, args.seeds)
            entry["subsamples"][label] = {
                "queries": len(ids),
                "sd_of_paired_diff": round(spread, 4),
                "zero_diff_queries": sum(1 for d in diffs if d == 0.0),
                "mde_at_this_n": round(offline_mde(spread, len(ids)), 4),
                "n_for_effect": {f"{e:.3f}": offline_n(spread, e) for e in OFFLINE_GRID},
                "interleaving": share,
            }
        pairs.append(entry)

    sequential = [sequential_plan(e, args.simulations) for e in ONLINE_GRID]
    target_plan = next(p for p in sequential if p["excess"] == ONLINE_TARGET)
    fallback_plan = next(p for p in sequential if p["excess"] == max(ONLINE_GRID))

    candidate = next(p for p in pairs if p["candidate_for_6_2"])
    candidate_share = candidate["subsamples"]["все"]["interleaving"]["mean"]
    queries_needed = math.ceil(int(target_plan["n_max"]) / candidate_share)
    fallback_informative = math.ceil(int(fallback_plan["n_max"]))

    traffic = live_traffic()
    report = {
        # Дата правила, а не дата прогона: она обязана остаться прежней при
        # любом пересчете, иначе файл начнет утверждать, что правило написано
        # позже, чем оно написано. Когда числа пересчитаны, стоит рядом.
        "written_at": RULE_WRITTEN_AT,
        "recomputed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "rule": {
            "written_before_data": True,
            "text": (
                "Уровень значимости 0,05 двусторонний, мощность 0,80. Офлайн обязаны увидеть "
                "сдвиг 0,020 NDCG@10, это порог алерта из S16. Онлайн обязаны увидеть долю "
                "выигранных запросов 0,55 против 0,50, сдвиг объявлен в своих единицах, потому "
                "что пересчета NDCG@10 в долю выигранных запросов в проекте нет. Взглядов не "
                "больше четырнадцати, по одному в сутки ночным прогоном, взгляд на выборке "
                "меньше тридцати информативных запросов пропускается. Поправка на подглядывание "
                "это постоянная граница Покока вместо 1,96, подобранная моделированием под те же "
                "0,05 на все взгляды вместе. Замер прекращается на первом взгляде за границей "
                "либо на четырнадцатом взгляде, и тогда результат это «различия не установлено» "
                "плюс интервал, а конфигурация сервиса не меняется. Трафика не хватило, если на "
                "четырнадцатом взгляде информативных запросов меньше, чем нужно для доли 0,60 "
                "против 0,50: тогда онлайн замер объявляется не проведенным и заменяется офлайн "
                "интерливингом на замороженном наборе, где за клик принято попадание "
                "размеченного релевантного фрагмента в первую десятку слитого списка."
            ),
            "alpha": ALPHA,
            "power": POWER,
            "looks": LOOKS,
            "min_queries_per_look": MIN_LOOK,
            "offline_target_effect": OFFLINE_TARGET,
            "offline_grid": list(OFFLINE_GRID),
            "online_target_excess": ONLINE_TARGET,
            "online_grid": list(ONLINE_GRID),
            "seed": SEED,
            "winner_not_computed_here": (
                "в интерливинге берется только модуль перевеса, знак нигде не сохраняется: "
                "победителя объявляет задача 6.2 после коммита этого правила"
            ),
            "also_written_in": "orchestration/EXPERIMENTS.md, S19",
        },
        "inputs": {
            "metric": WATCHED,
            "benchmark": {
                "queries_scored": baseline["queries_scored"],
                "queries_total": baseline["queries_total"],
                "live": len(subsamples["живые"]),
                "synthetic": sum(
                    1 for i in subsamples["все"] if queries[i].origin == "синтетический"
                ),
                "manual": sum(1 for i in subsamples["все"] if queries[i].origin == "ручной"),
                "level_bm25": baseline["metrics"][WATCHED],
                "source": "results/regression-baseline.json",
            },
            "noise_ceiling": threshold["delivery_noise"]["ceiling"],
            "noise_runs": threshold["delivery_noise"]["runs"],
            "alert_threshold": threshold["threshold"],
            "smallest_real_shift": threshold["real_shift_same_corpus"],
            "noise_source": "results/alert-threshold.json",
            "live_traffic": traffic,
        },
        "offline": {
            "test": "парный перестановочный тест на всем наборе, один раз, подглядывания нет",
            "mde_formula": "(z(1-α/2) + z(мощность)) * sd(разности) / sqrt(n)",
            "pairs": pairs,
        },
        "online": {
            "design": (
                "интерливинг команд, знаковый тест на долю выигранных запросов, "
                "последовательный с постоянной границей Покока"
            ),
            "plans": sequential,
            "candidate_pair": candidate["pair"],
            "candidate_informative_share": candidate_share,
            "queries_needed_at_target": queries_needed,
            "days_needed": {str(rate): round(queries_needed / rate, 1) for rate in TRAFFIC_RATES},
            # Сколько запросов в сутки нужно, чтобы замер уложился в объявленные
            # четырнадцать взглядов. Это и есть проверка «хватает ли трафика».
            "queries_per_day_needed": math.ceil(queries_needed / LOOKS),
            "fallback_trigger_informative_queries": fallback_informative,
        },
        "decision": {
            "online_possible": traffic["lines_on_disk"] > 0,
            "reason": (
                f"живого трафика в репозитории {traffic['lines_on_disk']} строк за "
                f"{traffic['days_reported']} суток, выгрузок {traffic['drains']}"
            ),
            "replacement": (
                "офлайн сравнение тем же интерливингом на замороженном наборе, клики заменены "
                "разметкой, подвыборки «все 991» и «живые 142» печатаются отдельно и не "
                "усредняются"
            ),
        },
        "seconds": round(time.time() - started, 1),
    }

    out = ROOT / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"правило и размеры выборки: {args.out}, {report['seconds']} с")
    print(
        f"офлайн: на {len(subsamples['все'])} запросах различим сдвиг "
        f"{candidate['subsamples']['все']['mde_at_this_n']} {WATCHED} для пары "
        f"{candidate['pair']}, объявленный {OFFLINE_TARGET} требует "
        f"{candidate['subsamples']['все']['n_for_effect'][f'{OFFLINE_TARGET:.3f}']} запросов"
    )
    print(
        f"онлайн: доля {target_plan['win_rate']} требует {target_plan['n_max']} информативных "
        f"запросов при границе {target_plan['boundary_z']} вместо "
        f"{target_plan['boundary_z_without_correction']}, то есть {queries_needed} запросов "
        f"трафика при информативной доле {candidate_share}"
    )
    print(
        f"трафика есть {traffic['lines_on_disk']} строк за {traffic['days_reported']} суток: "
        f"{'онлайн замер возможен' if report['decision']['online_possible'] else 'работает замена'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
