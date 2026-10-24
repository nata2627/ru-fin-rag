"""Сравнение двух конфигураций поиска по правилу, записанному до данных.

Задача 6.2. Онлайн теста нет и не будет: живого трафика у сервиса ноль строк за
ноль суток, и это посчитано в задаче 6.1 (`results/sample-size.json`, раздел
`decision`). Там же объявлена замена: офлайн интерливинг команд на замороженном
наборе, где клик заменен разметкой. Здесь эта замена выполняется, и впервые
берется знак перевеса: правило остановки уже закоммичено, подсмотреть ответ
раньше правила было нельзя.

Правило не переписывается в этом файле, оно читается из `results/sample-size.json`:
уровень 0,05 двусторонний, не больше четырнадцати взглядов, взгляд на выборке
меньше тридцати информативных запросов пропускается, поправка на подглядывание
это постоянная граница Покока. Если в файле правила стоит, что оно записано не до
данных, скрипт отказывается считать.

Считается три числа на каждой подвыборке, и подвыборки не усредняются:

1. Сдвиг NDCG@10 парным бутстрэпом и перестановочным тестом. Это офлайн тест из
   правила, один взгляд, подглядывания нет.
2. Доля запросов, которые выиграл интерливинг, с интервалом Уилсона. Знаменатель
   это информативные запросы (перевес не ноль), а не все: запрос с нулевым
   перевесом в знаковый тест не входит.
3. Что решило бы правило остановки. Четырнадцать взглядов делаются по
   перемешанному порядку замороженного набора с записанным зерном, граница Покока
   подбирается моделированием под размеры этих взглядов.

Чего это измерение не дает, сказано прямо, потому что иначе его прочтут как
онлайн результат. Взгляды офлайн это перетасовка одного и того же набора, а не
четырнадцать суток независимого трафика: правило проверяется на том, что есть, и
отвечает на вопрос «что оно решило бы на этих данных». Клик заменен разметкой, то
есть предпочтение пользователя смоделировано, а не наблюдено.

Запуск: python scripts/compare_configs.py [--seeds 20] [--simulations 20000]
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
from statistics import NormalDist, mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.benchmark import load_qrels, load_queries  # noqa: E402
from rufinrag.corpus import read_jsonl  # noqa: E402
from rufinrag.interleaving import margin  # noqa: E402
from rufinrag.metrics import paired_diff_ci, per_query  # noqa: E402

WATCHED = "NDCG@10"
RULE = ROOT / "results/sample-size.json"
PAIR = ("bm25", "hybrid")
SERVED = "hybrid"
LIVE_ORIGIN = "вопрос из разъяснений Банка России"
SYNTHETIC_ORIGIN = "синтетический"
RUNS = {
    "bm25": "data/benchmark/run-bm25.jsonl",
    "hybrid": "data/benchmark/run-hybrid.jsonl",
}


def load_rule() -> dict:
    rule = json.loads(RULE.read_text(encoding="utf-8"))["rule"]
    if not rule.get("written_before_data"):
        raise ValueError(
            f"в {RULE.name} правило не помечено как записанное до данных: "
            "сравнение по нему ничего не значит"
        )
    return rule


def signed_margins(
    left: dict[str, list[str]],
    right: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    ids: list[str],
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    return {
        query_id: margin(left.get(query_id, []), right.get(query_id, []), qrels[query_id], rng)
        for query_id in ids
    }


def wilson(wins: int, total: int) -> tuple[float, float]:
    """Интервал доли по Уилсону: на 30 наблюдениях нормальный врет, этот держит."""
    if total == 0:
        return (0.0, 0.0)
    z = NormalDist().inv_cdf(0.975)
    share = wins / total
    denominator = 1 + z**2 / total
    center = (share + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / denominator
    return (round(center - half, 4), round(center + half, 4))


def sign_statistic(wins: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (wins - total / 2) / math.sqrt(total / 4)


def pocock_boundary(sizes: list[int], alpha: float, simulations: int, seed: int) -> float:
    """Та же граница, что в правиле: подбирается моделированием под эти взгляды.

    Берется наименьшее наблюденное значение, у которого хвост не толще alpha, а не
    квантиль: при сотне запросов на взгляд статистика принимает мало различных
    значений, и квантиль садится ровно на одно из них, отпуская уровень выше
    объявленного.
    """
    if not sizes:
        return float("inf")
    rng = np.random.default_rng(seed)
    counts = np.asarray(sizes, dtype=float)
    increments = np.diff(np.concatenate(([0.0], counts))).astype(int)
    wins = rng.binomial(increments, 0.5, size=(simulations, len(sizes))).cumsum(axis=1)
    statistics = np.abs((wins - counts / 2) / np.sqrt(counts / 4))
    worst = np.sort(statistics.max(axis=1))
    candidates = np.unique(worst)
    tail = 1.0 - np.searchsorted(worst, candidates, side="left") / worst.size
    safe = candidates[tail <= alpha]
    return float(safe[0]) if safe.size else float("inf")


def sequential(
    margins: dict[str, int], rule: dict, seed: int, simulations: int
) -> dict[str, object]:
    """Что решило бы правило остановки на этом наборе при этом порядке запросов."""
    order = list(margins)
    random.Random(seed).shuffle(order)
    looks = int(rule["looks"])
    minimum = int(rule["min_queries_per_look"])
    edges = [max(1, round(len(order) * k / looks)) for k in range(1, looks + 1)]

    table: list[dict[str, object]] = []
    for number, edge in enumerate(edges, start=1):
        seen = order[:edge]
        informative = [margins[q] for q in seen if margins[q] != 0]
        wins = sum(1 for value in informative if value > 0)
        table.append(
            {
                "look": number,
                "queries_seen": len(seen),
                "informative": len(informative),
                "wins_right": wins,
                "statistic": round(sign_statistic(wins, len(informative)), 4),
                "skipped": len(informative) < minimum,
            }
        )

    usable = [row for row in table if not row["skipped"]]
    boundary = pocock_boundary(
        [int(row["informative"]) for row in usable], float(rule["alpha"]), simulations, seed
    )
    crossed = next((row for row in usable if abs(float(row["statistic"])) >= boundary), None)
    for row in table:
        row["crossed"] = (not row["skipped"]) and abs(float(row["statistic"])) >= boundary
    return {
        "order_seed": seed,
        "looks_planned": looks,
        "looks_skipped": sum(1 for row in table if row["skipped"]),
        "boundary": round(boundary, 4),
        "stopped_at_look": crossed["look"] if crossed else None,
        "informative_at_stop": crossed["informative"] if crossed else None,
        "statistic_at_stop": crossed["statistic"] if crossed else None,
        "table": table,
    }


def subsample_report(
    ids: list[str],
    values: dict[str, dict[str, float]],
    left: dict[str, list[str]],
    right: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    rule: dict,
    seeds: int,
    simulations: int,
) -> dict[str, object]:
    base_seed = int(rule["seed"])
    ndcg = paired_diff_ci(
        [values[PAIR[1]][i] for i in ids], [values[PAIR[0]][i] for i in ids], seed=base_seed
    )

    draws = [signed_margins(left, right, qrels, ids, base_seed + s) for s in range(seeds)]
    informative = [sum(1 for v in draw.values() if v != 0) for draw in draws]
    wins = [sum(1 for v in draw.values() if v > 0) for draw in draws]
    losses = [sum(1 for v in draw.values() if v < 0) for draw in draws]
    shares = [w / n if n else 0.0 for w, n in zip(wins, informative, strict=True)]

    runs = [sequential(draw, rule, base_seed + s, simulations) for s, draw in enumerate(draws)]
    stopped = [r for r in runs if r["stopped_at_look"] is not None]

    # Один взгляд на всем, для сравнения: сколько стоило подглядывание.
    median_draw = sorted(range(seeds), key=lambda s: shares[s])[seeds // 2]
    fixed_total = informative[median_draw]
    fixed_wins = wins[median_draw]
    fixed_statistic = sign_statistic(fixed_wins, fixed_total)

    return {
        "queries": len(ids),
        "ndcg": {
            "left": round(mean(values[PAIR[0]][i] for i in ids), 4),
            "right": round(mean(values[PAIR[1]][i] for i in ids), 4),
            "diff_right_minus_left": ndcg["mean"],
            "ci": [ndcg["lo"], ndcg["hi"]],
            "p_permutation": ndcg["p"],
            "target_effect": rule["offline_target_effect"],
            "above_target": abs(ndcg["mean"]) >= float(rule["offline_target_effect"]),
        },
        "interleaving": {
            "seeds": seeds,
            "informative_mean": round(mean(informative), 1),
            "informative_min": min(informative),
            "informative_max": max(informative),
            "wins_right_mean": round(mean(wins), 1),
            "wins_left_mean": round(mean(losses), 1),
            "ties_mean": round(len(ids) - mean(informative), 1),
            "win_share_mean": round(mean(shares), 4),
            "win_share_min": round(min(shares), 4),
            "win_share_max": round(max(shares), 4),
        },
        "fixed_look": {
            "seed_used": "медианное зерно по доле, чтобы не брать ни лучшее, ни худшее",
            "informative": fixed_total,
            "wins_right": fixed_wins,
            "win_share": round(fixed_wins / fixed_total, 4) if fixed_total else 0.0,
            "ci_wilson": list(wilson(fixed_wins, fixed_total)),
            "statistic": round(fixed_statistic, 4),
            "boundary_single_look": round(NormalDist().inv_cdf(1 - float(rule["alpha"]) / 2), 4),
            "significant": abs(fixed_statistic)
            >= NormalDist().inv_cdf(1 - float(rule["alpha"]) / 2),
        },
        "sequential": {
            "runs": seeds,
            "stopped_runs": len(stopped),
            "boundary_mean": round(mean(r["boundary"] for r in runs), 4),
            "stopped_at_look_min": min((r["stopped_at_look"] for r in stopped), default=None),
            "stopped_at_look_max": max((r["stopped_at_look"] for r in stopped), default=None),
            "informative_at_stop_mean": round(
                mean(r["informative_at_stop"] for r in stopped), 1
            )
            if stopped
            else None,
            "looks_skipped_mean": round(mean(r["looks_skipped"] for r in runs), 1),
            "example": runs[0],
        },
    }


def null_arm(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    ids: list[str],
    rule: dict,
    seeds: int,
    simulations: int,
) -> dict[str, object]:
    """Та же процедура на двух копиях одной связки: сколько она врет на пустом.

    Нужна потому, что интерливинг команд на одинаковых выдачах не дает ничью.
    Общий фрагмент достается той стороне, которая его вытянула, поэтому на двух
    копиях bm25 перевес бывает не нулевым, и такой запрос попадает в
    информативные. Доля выигранных на пустой арке обязана накрывать 0,5, а
    правило остановки обязано срабатывать на малой доле порядков: это и есть
    его уровень ложной тревоги, замеренный, а не обещанный.
    """
    base_seed = int(rule["seed"])
    draws = [signed_margins(run, run, qrels, ids, base_seed + s) for s in range(seeds)]
    informative = [sum(1 for v in draw.values() if v != 0) for draw in draws]
    wins = [sum(1 for v in draw.values() if v > 0) for draw in draws]
    shares = [w / n if n else 0.0 for w, n in zip(wins, informative, strict=True)]
    runs = [sequential(draw, rule, base_seed + s, simulations) for s, draw in enumerate(draws)]
    stopped = sum(1 for r in runs if r["stopped_at_look"] is not None)
    # Интервал считается по одному розыгрышу, а не по всем двадцати вместе:
    # розыгрыши идут по одним и тем же 991 запросам, и сложенные знаменатели
    # сузили бы интервал в разы, объявив перекос там, где его нет.
    median = sorted(range(seeds), key=lambda s: shares[s])[seeds // 2]
    interval = wilson(wins[median], informative[median])
    return {
        "pair": f"{PAIR[0]} / {PAIR[0]}",
        "queries": len(ids),
        "informative_mean": round(mean(informative), 1),
        "informative_share": round(mean(informative) / len(ids), 4),
        "win_share_mean": round(mean(shares), 4),
        "win_share_min": round(min(shares), 4),
        "win_share_max": round(max(shares), 4),
        "win_share_median_draw": round(shares[median], 4),
        "informative_median_draw": informative[median],
        "win_share_ci_wilson": list(interval),
        "covers_half": interval[0] <= 0.5 <= interval[1],
        "stopped_runs": stopped,
        "runs": seeds,
        "alpha": rule["alpha"],
    }


def questions(count: int) -> str:
    """«1 вопрос», «2 вопроса», «991 вопрос»: число идет в текст решения."""
    last_two, last = count % 100, count % 10
    if 11 <= last_two <= 14 or last == 0 or last >= 5:
        return f"{count} вопросов"
    return f"{count} вопрос" if last == 1 else f"{count} вопроса"


def verdict(part: dict, label: str) -> dict[str, object]:
    """Решение по подвыборке: сначала объявленный тест, потом правило остановки."""
    ndcg = part["ndcg"]
    fixed = part["fixed_look"]
    seq = part["sequential"]
    winner = PAIR[1] if ndcg["diff_right_minus_left"] > 0 else PAIR[0]
    established = (
        bool(ndcg["above_target"])
        and ndcg["p_permutation"] < 0.05
        and bool(fixed["significant"])
    )
    text = (
        f"подвыборка «{label}», {questions(part['queries'])}: сдвиг NDCG@10 "
        f"{ndcg['diff_right_minus_left']:+.4f} [{ndcg['ci'][0]}; {ndcg['ci'][1]}], "
        f"перестановочный тест p={ndcg['p_permutation']}, объявленный порог "
        f"{ndcg['target_effect']} {'взят' if ndcg['above_target'] else 'не взят'}. "
        f"Интерливинг: доля выигранных {fixed['win_share']} "
        f"[{fixed['ci_wilson'][0]}; {fixed['ci_wilson'][1]}] на {fixed['informative']} "
        f"информативных запросах из {part['queries']}, знаковый тест "
        f"{fixed['statistic']} против 1,96. Правило остановки сработало бы на "
        f"{seq['stopped_runs']} порядках из {seq['runs']} при границе {seq['boundary_mean']}"
    )
    if established:
        text += f". Вывод: выигрывает {winner}"
    else:
        text += ". Вывод: различия не установлено"
    return {
        "subsample": label,
        "queries": part["queries"],
        "established": established,
        "winner": winner if established else None,
        "text": text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=ROOT / "results/compare-configs.json")
    args = parser.parse_args()

    started = time.time()
    rule = load_rule()
    qrels = load_qrels()
    queries = load_queries()
    runs = {name: {row["query_id"]: list(row["ranked"]) for row in read_jsonl(ROOT / path)}
            for name, path in RUNS.items()}
    values = {
        name: dict(zip(*per_query(run, qrels, WATCHED), strict=True)) for name, run in runs.items()
    }
    scored_ids, _ = per_query(runs[PAIR[0]], qrels, WATCHED)

    # Подвыборки печатаются отдельно и не усредняются: живые вопросы и
    # синтетика отвечают на разные вопросы, и среднее по ним не значит ничего.
    # Делится по происхождению вопроса, а не по имени подвыборки: подвыборки
    # проекта вложены друг в друга («dev», «невиданные акты»), и «синтетика» в
    # этом поле означает лишь одну из них, 150 вопросов из 846 синтетических.
    subsamples = {
        "все": scored_ids,
        "живые": [i for i in scored_ids if queries[i].origin == LIVE_ORIGIN],
        "синтетика": [i for i in scored_ids if queries[i].origin == SYNTHETIC_ORIGIN],
    }

    report: dict[str, object] = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "pair": f"{PAIR[0]} / {PAIR[1]}",
        "rule_from": str(RULE.relative_to(ROOT)),
        "rule_written_at": json.loads(RULE.read_text(encoding="utf-8"))["written_at"],
        "online_test": "не проводился: живого трафика 0 строк за 0 суток",
        "click_model": "за клик принято попадание размеченного релевантного фрагмента "
                       "в первую десятку слитого списка",
        "limits": [
            "взгляды офлайн это перетасовка одного набора, а не четырнадцать суток "
            "независимого трафика",
            "предпочтение пользователя смоделировано разметкой, а не наблюдено",
            "подвыборки не усредняются, у каждой свой знаменатель",
            "интерливинг команд на одинаковых выдачах ничьей не дает: общий фрагмент "
            "достается той стороне, которая его вытянула, поэтому часть информативных "
            "запросов это шум отбора, и его размер замерен пустой аркой",
        ],
    }
    for label, ids in subsamples.items():
        report[label] = subsample_report(
            ids, values, runs[PAIR[0]], runs[PAIR[1]], qrels, rule, args.seeds, args.simulations
        )
    decisions = [verdict(report[label], label) for label in subsamples]
    # Решение сервиса принимается по живым вопросам: сервис отвечает на них, а
    # не на синтетике, и усреднять эти подвыборки правило запрещает.
    live = next(d for d in decisions if d["subsample"] == "живые")
    whole = next(d for d in decisions if d["subsample"] == "все")
    report["null_arm"] = null_arm(
        runs[PAIR[0]], qrels, subsamples["все"], rule, args.seeds, args.simulations
    )
    report["decision"] = {
        "per_subsample": decisions,
        "served_config": SERVED,
        "change_served_config": False,
        "why": (
            f"на живых вопросах различия не установлено ({questions(live['queries'])}), "
            f"а на всем наборе выигрывает {whole['winner']}, который и стоит в сервисе: "
            "менять нечего"
        )
        if not live["established"] and whole["winner"] == SERVED
        else "решение требует разбора: подвыборки разошлись с тем, что стоит в сервисе",
    }
    report["seconds"] = round(time.time() - started, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in report["decision"]["per_subsample"]:
        print(item["text"])
    null = report["null_arm"]
    print(
        f"пустая арка {null['pair']}: информативных {null['informative_mean']} из "
        f"{null['queries']}, доля выигранных {null['win_share_mean']} "
        f"[{null['win_share_ci_wilson'][0]}; {null['win_share_ci_wilson'][1]}], "
        f"правило остановки сработало на {null['stopped_runs']} порядках из {null['runs']}"
    )
    print(f"связка сервиса {SERVED}: {report['decision']['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
