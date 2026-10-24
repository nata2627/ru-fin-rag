"""Замер слоя ответа: порог отказа, поля цитаты, цена опоры.

Правило порога записано до чисел (`orchestration/EXPERIMENTS.md`, S21) и здесь
применяется без правок: порог это 1-процентный квантиль признака опоры по 991
размеченному запросу эталонного набора, округленный вниз до двух значащих
цифр, то есть бюджет ложных отказов это 9 запросов из 991. Признак выбирается
из трех по доле отказов на половине «выбор» набора вопросов вне корпуса,
число для отчета берется с половины «проверка», которая в выборе не участвует.

Выдача для признака берется лексической ступенью (связка `bm25`): признак
опоры лексический, в онлайне стоит та же связка, и монитор считает по журналу
ту же величину (`rufinrag.monitoring`).

Запуск: python scripts/measure_answer.py [--sample N] [--out results/answer.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.answer import (  # noqa: E402
    IDF_COVERAGE,
    SUPPORT_FEATURES,
    clauses_in,
    coverage_of,
    header_length,
    idf_weights,
    pick_quote,
)
from rufinrag.config import load_catalog  # noqa: E402
from rufinrag.corpus import read_jsonl  # noqa: E402
from rufinrag.monitoring import SUPPORT_FEATURES as MONITOR_FEATURES  # noqa: E402
from rufinrag.monitoring import shape_of_text  # noqa: E402
from rufinrag.retrieval.text import stem, token_spans  # noqa: E402
from rufinrag.search import SearchService  # noqa: E402

BENCHMARK = ROOT / "data/benchmark"
OUT_FILE = ROOT / "results/answer.json"

# Бюджет ложных отказов и глубина, по которой берется лучший кандидат. Оба
# числа объявлены в S21 до замера и здесь только применяются.
CHOSEN_FEATURE = "idf_coverage"
BUDGET = 0.01
DEPTH = 10
# Предел связи признака с длиной запроса, односторонний. Беда только у
# признака, который с длиной растет: порог выведен на вопросах набора, а они
# длинные, и короткий вопрос окажется ниже порога по длине, а не по отсутствию
# опоры. Признак, который с длиной падает, короткому вопросу не вредит, и его
# своя беда (отказ на длинном многочастном вопросе) видна в разборе по бинам.
# Граница 0,3 назначена как обычная граница слабой монотонной связи.
LENGTH_TIE_MAX = 0.3
# Короткий запрос это три токена и меньше: столько слов набирает человек,
# который пришел по ссылке из резюме и пробует сервис одним словосочетанием.
SHORT_TOKENS = 3
QUOTE_CHARS = 400
CONFIG = "bm25"


def floor_two_digits(value: float) -> float:
    """Округление вниз до двух значащих цифр.

    Вниз, а не к ближайшему: порог, округленный вверх, вышел бы за объявленный
    бюджет ложных отказов, то есть правило нарушило бы само себя.
    """
    if value <= 0:
        return 0.0
    step = 10 ** (math.floor(math.log10(value)) - 1)
    return round(math.floor(round(value / step, 9)) * step, 12)


def quantile_floor(values: list[float], share: float) -> float:
    """Значение, ниже которого лежит не более `share` значений."""
    ordered = sorted(values)
    index = math.floor(share * len(ordered))
    return ordered[min(index, len(ordered) - 1)]


def quartile_bins(values: list[int]) -> list[tuple[int, int]]:
    """Границы квартилей по числу токенов запроса, включительно с двух сторон."""
    ordered = sorted(values)
    cuts = [
        ordered[min(int(share * len(ordered)), len(ordered) - 1)] for share in (0.25, 0.5, 0.75)
    ]
    edges = [ordered[0], *cuts, ordered[-1] + 1]
    return [(edges[i], edges[i + 1] - 1) for i in range(4)]


def load_queries() -> dict[str, dict]:
    return {row["query_id"]: row for row in read_jsonl(BENCHMARK / "queries.jsonl")}


def load_qrels() -> dict[str, dict[str, int]]:
    rel: dict[str, dict[str, int]] = defaultdict(dict)
    with open(BENCHMARK / "qrels.tsv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rel[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return dict(rel)


def features_of(service: SearchService, query: str, depth: int = DEPTH) -> dict:
    """Все три признака опоры по одному запросу, плюс цена шага."""
    t0 = time.monotonic()
    hits = service.bm25.search(query, k=depth)
    weights = idf_weights(service.bm25, query)
    best, best_id = 0.0, ""
    for chunk_id, _ in hits:
        text = service.corpus.text(chunk_id)
        value = coverage_of(weights, {stem(token) for token, _, _ in token_spans(text)})
        if value > best or not best_id:
            best, best_id = value, chunk_id
    shape = shape_of_text(
        query,
        service.bm25.vocab,
        hits=len(hits),
        top_score=hits[0][1] if hits else None,
    )
    values = {name: shape.support(name) for name in MONITOR_FEATURES}
    values[IDF_COVERAGE] = best
    return {
        "values": values,
        "hits": len(hits),
        "best_chunk_id": best_id,
        "tokens": shape.tokens,
        "found": shape.found,
        "ms": (time.monotonic() - t0) * 1000,
    }


def refused(feature: str, threshold: float, row: dict) -> bool:
    return row["hits"] == 0 or row["values"][feature] < threshold


def share(count: int, total: int) -> dict:
    return {"refused": count, "of": total, "share": round(count / total, 4) if total else None}


def refusals(feature: str, threshold: float, rows: list[dict]) -> dict:
    return share(sum(1 for row in rows if refused(feature, threshold, row)), len(rows))


def corpus_fields(service: SearchService) -> dict:
    """Чем на самом деле располагает ссылка: разделы, пункты, шапка фрагмента.

    Извлечение номера пункта проверяется на неубывание внутри акта: номера
    пунктов в акте идут по возрастанию, и выражение, которое ловит номера
    строк таблиц или позиции перечислений, это условие нарушит.
    """
    with_section = with_clauses = with_header = 0
    pairs = ordered = 0
    previous: dict[str, tuple[int, ...]] = {}
    for index, chunk in enumerate(service.corpus.chunks):
        text = service.corpus.texts.get(index)
        head = header_length(chunk, text)
        with_header += 1 if head else 0
        with_section += 1 if chunk.section else 0
        found = clauses_in(text[head:])
        if not found:
            continue
        with_clauses += 1
        first = tuple(int(part) for part in found[0].split("."))
        earlier = previous.get(chunk.act_id)
        if earlier is not None:
            pairs += 1
            ordered += 1 if first >= earlier else 0
        previous[chunk.act_id] = first
    total = len(service.corpus.chunks)
    return {
        "chunks": total,
        "with_section": with_section,
        "with_clauses": with_clauses,
        "with_header": with_header,
        "clause_pairs_in_act": pairs,
        "clause_pairs_non_decreasing": ordered,
        "clause_order_share": round(ordered / pairs, 4) if pairs else None,
    }


def quote_stats(service: SearchService, rows: list[dict], queries: dict[str, dict]) -> dict:
    """Попадают ли слова запроса в выбранную выдержку."""
    lengths: list[int] = []
    with_terms = 0
    for row in rows:
        chunk_id = row["best_chunk_id"]
        if not chunk_id:
            continue
        chunk = service.corpus.get(chunk_id)
        text = service.corpus.text(chunk_id)
        terms = set(idf_weights(service.bm25, queries[row["query_id"]]["text"]))
        start, end, found = pick_quote(text, header_length(chunk, text), terms, QUOTE_CHARS)
        lengths.append(end - start)
        with_terms += 1 if found else 0
    return {
        "quotes": len(lengths),
        "with_query_terms": with_terms,
        "share_with_query_terms": round(with_terms / len(lengths), 4) if lengths else None,
        "median_chars": int(statistics.median(lengths)) if lengths else None,
        "limit_chars": QUOTE_CHARS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="проба: столько размеченных запросов")
    parser.add_argument("--out", default=str(OUT_FILE))
    args = parser.parse_args()

    catalog = load_catalog()
    if CONFIG not in catalog.configs:
        raise SystemExit(f"нет связки {CONFIG} в configs/retrieval.yaml")
    service = SearchService.build(catalog=catalog, with_dense=False)

    queries = load_queries()
    qrels = load_qrels()
    labeled = [qid for qid in queries if any(grade > 0 for grade in qrels.get(qid, {}).values())]
    order = list(queries)
    if args.sample:
        labeled = labeled[: args.sample]
        rest = [qid for qid in order if qid not in set(labeled)]
        order = labeled + rest[: args.sample]

    t0 = time.monotonic()
    rows: dict[str, dict] = {}
    for qid in order:
        row = features_of(service, queries[qid]["text"])
        row["query_id"] = qid
        row["podvyborka"] = queries[qid]["podvyborka"]
        rows[qid] = row
    seconds = time.monotonic() - t0
    print(f"признаки посчитаны: {len(rows)} запросов за {seconds:.1f} с")

    labeled_set = set(labeled)
    labeled_rows = [rows[qid] for qid in labeled]
    live_rows = [
        row
        for row in rows.values()
        if row["podvyborka"] == "живые" and row["query_id"] not in labeled_set
    ]
    # Короткие вопросы и ручные вопросы людей: на них проверяется, переносится
    # ли бюджет ложных отказов с набора на запрос живого человека. Оба среза
    # взяты из замороженного набора, а не составлены под этот замер.
    manual_rows = [row for row in rows.values() if row["podvyborka"] == "ручные"]
    short_rows = [row for row in rows.values() if row["tokens"] <= SHORT_TOKENS]

    # Бины длины запроса для разбора отказов: квартили по числу токенов на всем
    # наборе. Нужны затем, что порог, выведенный на длинных вопросах, может
    # отказывать коротким по длине, а не по отсутствию опоры.
    token_bins = quartile_bins([row["tokens"] for row in rows.values()])

    candidates = {}
    for feature in SUPPORT_FEATURES:
        values = [row["values"][feature] for row in labeled_rows]
        threshold = floor_two_digits(quantile_floor(values, BUDGET))
        length_tie = float(
            spearmanr(
                [row["values"][feature] for row in rows.values()],
                [row["tokens"] for row in rows.values()],
            ).statistic
        )
        candidates[feature] = {
            "threshold": threshold,
            "quantile_raw": round(quantile_floor(values, BUDGET), 6),
            "length_dependence": round(length_tie, 4),
            "grows_with_length": length_tie > LENGTH_TIE_MAX,
            "refusals_by_tokens": [
                {
                    "tokens_from": low,
                    "tokens_to": high,
                    **refusals(
                        feature,
                        threshold,
                        [row for row in rows.values() if low <= row["tokens"] <= high],
                    ),
                }
                for low, high in token_bins
            ],
            "false_refusals_labeled": refusals(feature, threshold, labeled_rows),
            "live_unlabeled": refusals(feature, threshold, live_rows),
            "manual_queries": refusals(feature, threshold, manual_rows),
            "short_queries": refusals(feature, threshold, short_rows),
            "value_quantiles": {
                str(q): round(quantile_floor(values, q), 4) for q in (0.01, 0.05, 0.25, 0.5)
            },
        }

    # Признак не выбирается здесь заново, он закреплен. Выбор был сделан 24.10
    # по правилу S21 на наборе вопросов вне корпуса, и этот набор решением
    # автора из проекта убран (40 вопросов, в том числе бытовых, в портфолио по
    # нормативке ЦБ им места нет). Переписать правило выбора под то, что
    # осталось, значило бы подобрать его после данных, поэтому здесь стоит
    # закрепленный исход с указанием, где он записан, а не новый отбор.
    # Признак, связанный с длиной запроса, из выбора выбывал: порог выведен на
    # вопросах набора, а они длинные, и такой признак отказывал бы короткому
    # вопросу живого человека.
    winner = CHOSEN_FEATURE
    if winner not in candidates:
        raise SystemExit(f"закрепленного признака {winner} нет среди посчитанных")
    if candidates[winner]["grows_with_length"]:
        raise SystemExit(
            f"закрепленный признак {winner} вырос с длиной запроса: правило S21 его "
            "исключает, и выбор надо пересматривать явно, а не молча"
        )
    threshold = candidates[winner]["threshold"]

    by_subsample: dict[str, dict] = {}
    for name in sorted({row["podvyborka"] for row in labeled_rows}):
        part = [row for row in labeled_rows if row["podvyborka"] == name]
        by_subsample[name] = refusals(winner, threshold, part)

    costs = [row["ms"] for row in rows.values()]
    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "corpus_version": service.corpus_version,
        "config": CONFIG,
        "depth": DEPTH,
        "rule": {
            "budget_false_refusals": BUDGET,
            "budget_queries": math.floor(BUDGET * len(labeled_rows)),
            "text": (
                "порог это 1-процентный квантиль признака опоры по размеченным запросам "
                "эталонного набора, округленный вниз до двух значащих цифр. Признак "
                "закреплен за idf_coverage выбором от 24.10.2026 (S21), набор вопросов вне "
                "корпуса, на котором шел выбор, из проекта убран решением автора"
            ),
            "written_before": "orchestration/EXPERIMENTS.md, S21",
        },
        "benchmark_queries": len(rows),
        "labeled_queries": len(labeled_rows),
        "manual_queries": len(manual_rows),
        "short_queries": {
            "of": len(rows),
            "shorter_than": SHORT_TOKENS + 1,
            "count": len(short_rows),
        },
        "live_unlabeled_queries": len(live_rows),
        "support_feature": winner,
        "support_threshold": threshold,
        "candidates": candidates,
        "chosen": {
            "feature": winner,
            "threshold": threshold,
            "false_refusals_labeled": candidates[winner]["false_refusals_labeled"],
            "live_unlabeled": candidates[winner]["live_unlabeled"],
            "by_subsample": by_subsample,
            "manual_queries": candidates[winner]["manual_queries"],
            "short_queries": candidates[winner]["short_queries"],
            "length_dependence": candidates[winner]["length_dependence"],
            "refusals_by_tokens": candidates[winner]["refusals_by_tokens"],
        },
        "citations": corpus_fields(service),
        "quotes": quote_stats(service, labeled_rows, queries),
        "support_cost_ms": {
            "queries": len(costs),
            "median": round(statistics.median(costs), 3) if costs else None,
            "p95": round(sorted(costs)[int(0.95 * (len(costs) - 1))], 3) if costs else None,
        },
        "sample": args.sample or None,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"признак {winner}, порог {threshold}")
    print(f"ложных отказов {candidates[winner]['false_refusals_labeled']}")
    print(f"записано {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
