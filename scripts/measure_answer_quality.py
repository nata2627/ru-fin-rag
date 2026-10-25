"""Качество ответа: доля ответов с корректной ссылкой на источник и погрешность этой оценки.

Задача 7.4. Правило записано до чисел (`orchestration/EXPERIMENTS.md`, S24) и
здесь только применяется: корректной считается ссылка на фрагмент, у которого в
эталонной разметке стоит оценка больше нуля, знаменатель это выданные ответы, а
не все запросы, и подвыборки печатаются отдельно.

Второе число задачи это погрешность прибора. Эталон размечен моделью, значит
доля по нему оценка сверху. Судья моделью к ответам не применяется, потому что
приемка судьи не взята (каппа с человеком 0,236 при пороге 0,40, часть А,
`data/results/judge_acceptance.json`), и вместо применения судьи печатается его
согласие с человеком и таблица «эталон против человека» на тех же 150 парах.
Доля по этой таблице на выдачу не переносится: пары отобраны равными долями по
оценке судьи, то есть выборка стратифицированная.

Человеческий раздел считается из файлов части А. Их может не быть на машине
(прогон проверки работает без них), и тогда раздел переносится из прежнего
отчета с пометкой `from_previous`, как это делает реестр версий с отпечатками
артефактов. Выдумывать его нельзя, пропускать молча тоже.

Запуск: python scripts/measure_answer_quality.py [--sample N] [--config bm25 hybrid]
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.answer import AnswerService, load_answer_settings  # noqa: E402
from rufinrag.benchmark import load_qrels, load_queries  # noqa: E402
from rufinrag.config import load_catalog  # noqa: E402
from rufinrag.search import SearchService  # noqa: E402

OUT_FILE = ROOT / "results/answer-quality.json"
DEFAULT_PART_A = Path.home() / "Desktop/my_jobs/rag_retrieval"
HUMAN_SHEET = "data/queries/human_sample_new.tsv"
JUDGE_REPORT = "data/results/judge_acceptance.json"
# Глубина выдачи, с которой слой ответа берет цитаты. Та же, что в S21.
DEPTH = 10
FULL_ANSWER = 2
# Судья, по которому считалась приемка: лучший из четырех по согласию с
# человеком. Остальные три в отчет не идут, их числа лежат в части А.
ACCEPTED_JUDGE = "Qwen2.5-14B-Instruct-AWQ"


def share(hit: int, of: int) -> dict[str, object]:
    return {"hit": hit, "of": of, "share": round(hit / of, 4) if of else None}


def human_grade(row: dict[str, str]) -> int:
    """Оценка автора проекта, собранная из двух ее ответов правилом инструкции.

    Правило то же, что в части А (`scripts/kappa.py`): «отвечает целиком» это
    2, «отвечает вместе с другими» это 1, иначе 0. Оценку человек не ставит
    сам, и это замысел: два вопроса про текст устойчивее одной цифры.
    """
    if row["otvet1_celikom"].strip() == "да":
        return FULL_ANSWER
    return 1 if row["otvet2_vmeste"].strip() == "да" else 0


def citations_quality(
    service: AnswerService, queries: dict, qrels: dict[str, dict[str, int]], config: str
) -> dict[str, object]:
    """Доли корректных ссылок по одной связке, целиком и по подвыборкам."""
    counts: Counter[str] = Counter()
    by_subsample: dict[str, Counter[str]] = {}
    acts_relevant: dict[str, set[str]] = {}
    seconds = time.monotonic()
    for query_id, query in queries.items():
        relevant = {chunk for chunk, grade in qrels[query_id].items() if grade > 0}
        full = {chunk for chunk, grade in qrels[query_id].items() if grade >= FULL_ANSWER}
        if query_id not in acts_relevant:
            acts_relevant[query_id] = {
                service.search.corpus.get(chunk).act.act_id for chunk in relevant
            }
        answer = service.answer(query.text, top=DEPTH, config=config)
        bucket = by_subsample.setdefault(query.podvyborka, Counter())
        for tally in (counts, bucket):
            tally["queries"] += 1
        if not answer.answered:
            for tally in (counts, bucket):
                tally["refused"] += 1
            continue
        cited = [c.chunk_id for c in answer.citations]
        cited_acts = {c.act_id for c in answer.citations}
        for tally in (counts, bucket):
            tally["answered"] += 1
            # Опорный фрагмент это тот, по которому сервис решил отвечать. В
            # цитаты он попадает не всегда: опора ищется по десяти фрагментам, а
            # цитат пять. Без этого числа нельзя сравнить два пути из находки
            # S24, потому что у обоих опора и цитаты берутся из разных мест.
            tally["support_not_cited"] += answer.support.best_chunk_id not in cited
            tally["strict"] += any(chunk in relevant for chunk in cited)
            tally["full"] += any(chunk in full for chunk in cited)
            tally["act"] += bool(cited_acts & acts_relevant[query_id])
            tally["first"] += bool(cited) and cited[0] in relevant
            tally["citations"] += len(cited)
    out: dict[str, object] = {
        "config": config,
        "config_version": load_catalog().get(config).version,
        "seconds": round(time.monotonic() - seconds, 1),
        **report_of(counts),
    }
    out["by_subsample"] = {
        name: report_of(tally) for name, tally in sorted(by_subsample.items())
    }
    return out


def report_of(tally: Counter[str]) -> dict[str, object]:
    answered = tally["answered"]
    return {
        "queries": tally["queries"],
        "refused": tally["refused"],
        "answered": answered,
        "citations_per_answer": round(tally["citations"] / answered, 2) if answered else None,
        "опора не попала в цитаты": share(tally["support_not_cited"], answered),
        "ссылка на релевантный фрагмент": share(tally["strict"], answered),
        "ссылка на фрагмент с полным ответом": share(tally["full"], answered),
        "ссылка на релевантный акт": share(tally["act"], answered),
        "первая ссылка релевантна": share(tally["first"], answered),
    }


def cross_support(
    service: AnswerService,
    queries: dict,
    qrels: dict[str, dict[str, int]],
    lexical: str,
    online: str,
) -> dict[str, object]:
    """Цена одного из двух путей из находки S24, посчитанная до решения.

    Путь такой: опора считается по лексической выдаче, на которой выведен
    порог, а цитаты берутся из выдачи связки по умолчанию, которой отвечает
    сервис. Здесь это только замер, поведение сервиса не меняется.

    Лексические кандидаты связка по умолчанию считает и сейчас (первая ступень
    гибрида это они же), поэтому путь не добавляет сервису ни одного поиска, и
    это видно по полю `seconds` рядом.
    """
    counts: Counter[str] = Counter()
    # Второй счет того же пути с одной поправкой: опорный фрагмент обязан стоять
    # в цитатах. Без нее сервис решает отвечать по тексту, которого на экране
    # нет, и это возражение к пути надо не обсуждать, а считать.
    pinned: Counter[str] = Counter()
    # И тот же счет, когда опорный фрагмент ставится последней цитатой: набор
    # цитат от места не меняется, а первая строка экрана остается за выдачей,
    # которая ранжирует лучше.
    appended: Counter[str] = Counter()
    by_subsample: dict[str, Counter[str]] = {}
    limit = service.settings.citations
    started = time.monotonic()
    for query_id, query in queries.items():
        relevant = {chunk for chunk, grade in qrels[query_id].items() if grade > 0}
        full = {chunk for chunk, grade in qrels[query_id].items() if grade >= FULL_ANSWER}
        support, _ = service.support_of(
            service.search.search(query.text, top=DEPTH, config=lexical), query.text
        )
        bucket = by_subsample.setdefault(query.podvyborka, Counter())
        for tally in (counts, bucket, pinned, appended):
            tally["queries"] += 1
        if not support.passed:
            for tally in (counts, bucket, pinned, appended):
                tally["refused"] += 1
            continue
        # Цитата это фрагмент выдачи: слой ответа берет первые пять хитов и
        # печатает рядом акт, раздел и пункты. Поэтому мера считается по тем же
        # первым пяти, без второго прохода по слою ответа.
        hits = service.search.search(query.text, top=DEPTH, config=online).hits[:limit]
        cited = [hit.chunk_id for hit in hits]
        cited_acts = {hit.act_id for hit in hits}
        acts_relevant = {service.search.corpus.get(chunk).act.act_id for chunk in relevant}
        for tally in (counts, bucket):
            tally["answered"] += 1
            # Слабое место этого пути, названное числом: опора нашлась в
            # лексическом фрагменте, а в пятерку цитат он не попал, то есть
            # отвечать решили по одному тексту, а печатают другие.
            tally["support_not_cited"] += support.best_chunk_id not in cited
            tally["strict"] += any(chunk in relevant for chunk in cited)
            tally["full"] += any(chunk in full for chunk in cited)
            tally["act"] += bool(cited_acts & acts_relevant)
            tally["first"] += bool(cited) and cited[0] in relevant
            tally["citations"] += len(cited)
        # Поправка: опорный фрагмент встает первым, остальные сдвигаются, число
        # цитат то же. Первым, а не последним, потому что именно по нему принято
        # решение отвечать, и первую строку экрана читают раньше прочих.
        fixed = (
            cited
            if support.best_chunk_id in cited
            else [support.best_chunk_id, *cited[: limit - 1]]
        )
        fixed_acts = {service.search.corpus.get(chunk).act.act_id for chunk in fixed}
        last = (
            cited
            if support.best_chunk_id in cited
            else [*cited[: limit - 1], support.best_chunk_id]
        )
        for tally, order, acts in (
            (pinned, fixed, fixed_acts),
            (appended, last, fixed_acts),
        ):
            tally["answered"] += 1
            tally["strict"] += any(chunk in relevant for chunk in order)
            tally["full"] += any(chunk in full for chunk in order)
            tally["act"] += bool(acts & acts_relevant)
            tally["first"] += order[0] in relevant
            tally["citations"] += len(order)
    out: dict[str, object] = {
        "путь": "опора по лексической выдаче, цитаты из связки по умолчанию",
        "опора считается на": lexical,
        "цитаты из": online,
        "лишних поисков сервису": 0,
        "seconds": round(time.monotonic() - started, 1),
        **report_of(counts),
    }
    out["by_subsample"] = {name: report_of(tally) for name, tally in sorted(by_subsample.items())}
    out["с опорой в цитатах"] = {
        "поправка": "опорный фрагмент встает первой цитатой, число цитат то же",
        **report_of(pinned),
    }
    out["с опорой последней цитатой"] = {
        "поправка": "опорный фрагмент встает последней цитатой, число цитат то же",
        **report_of(appended),
    }
    return out


def human_section(part_a: Path, qrels: dict[str, dict[str, int]]) -> dict[str, object] | None:
    """Таблица «оценка эталона против оценки человека» на 150 размеченных парах.

    Направление ошибки эталона, а не доля по выдаче: выборка стратифицирована
    по оценке судьи равными долями (S24, пункт 6), и доля по ней на корпус не
    переносится.
    """
    sheet = part_a / HUMAN_SHEET
    report = part_a / JUDGE_REPORT
    if not sheet.exists() or not report.exists():
        return None
    with open(sheet, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    table: Counter[tuple[int, int]] = Counter()
    for row in rows:
        reference = qrels.get(row["query_id"], {}).get(row["chunk_id"], 0)
        table[(reference, human_grade(row))] += 1
    pairs = sum(table.values())
    reference_above = sum(n for (ref, hum), n in table.items() if ref > 0 and hum == 0)
    human_above = sum(n for (ref, hum), n in table.items() if ref == 0 and hum > 0)
    full_in_reference = sum(n for (ref, _), n in table.items() if ref >= FULL_ANSWER)
    full_agreed = sum(
        n for (ref, hum), n in table.items() if ref >= FULL_ANSWER and hum >= FULL_ANSWER
    )
    full_any = sum(n for (ref, hum), n in table.items() if ref >= FULL_ANSWER and hum > 0)
    judge = json.loads(report.read_text(encoding="utf-8"))
    accepted = next(j for j in judge["судьи"] if j["судья"] == ACCEPTED_JUDGE)
    return {
        "лист": f"{part_a.name}/{HUMAN_SHEET}",
        "кто разметил": judge["ручная разметка"]["кто разметил"],
        "пар": pairs,
        "запросов": len({row["query_id"] for row in rows}),
        "отбор": "равными долями по оценке судьи, выборка стратифицированная, "
        "доля по ней на выдачу не переносится",
        "таблица": {
            f"эталон {ref}, человек {hum}": n for (ref, hum), n in sorted(table.items())
        },
        "эталон выше человека": share(reference_above, pairs),
        "человек выше эталона": share(human_above, pairs),
        "полный ответ по эталону": {
            "пар": full_in_reference,
            "человек согласился полностью": full_agreed,
            "человек увидел хотя бы часть": full_any,
            "человек ответа не увидел": full_in_reference - full_any,
        },
        "судья": {
            "кто": accepted["судья"],
            "каппа с человеком": accepted["с человеком"]["каппа"],
            "интервал": accepted["с человеком"]["интервал"],
            "пар": accepted["с человеком"]["пар"],
            "порог": judge["правило"]["порог каппы с человеком"],
            "приемка": accepted["приёмка"],
            "правило записано": judge["правило"]["записано"],
            "применяется к ответам": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="взять первые N запросов")
    parser.add_argument("--config", nargs="+", default=None)
    parser.add_argument("--part-a", type=Path, default=DEFAULT_PART_A)
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    parser.add_argument(
        "--only-cross",
        action="store_true",
        help="пересчитать только вариант из находки S24 и дописать его в готовый отчет",
    )
    args = parser.parse_args()

    catalog = load_catalog()
    configs = args.config or sorted({"bm25", catalog.default})
    needs_dense = any(
        catalog.get(name).first_stage != "bm25" or catalog.get(name).dense_model is not None
        for name in configs
    )
    search = SearchService.build(with_dense=needs_dense)
    service = AnswerService.build(search, settings=load_answer_settings())

    qrels = load_qrels()
    all_queries = load_queries()
    labeled = {
        query_id: query
        for query_id, query in all_queries.items()
        if any(grade > 0 for grade in qrels.get(query_id, {}).values())
    }
    if args.sample:
        labeled = dict(list(labeled.items())[: args.sample])

    if args.only_cross:
        if not args.out.exists():
            print(f"нет отчета {args.out}, считать нечего", file=sys.stderr)
            return 2
        report = json.loads(args.out.read_text(encoding="utf-8"))
        report["cross_support"] = cross_support(
            service, labeled, qrels, "bm25", catalog.default
        )
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        block = report["cross_support"]
        strict = block["ссылка на релевантный фрагмент"]
        print(
            f"вариант из S24: ссылка на релевантный фрагмент {strict['hit']} из {strict['of']}, "
            f"отказов {block['refused']} из {block['queries']}, опора не в цитатах "
            f"{block['опора не попала в цитаты']['hit']} из "
            f"{block['опора не попала в цитаты']['of']}"
        )
        return 0

    report: dict[str, object] = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "corpus_version": search.corpus_version,
        "answer_version": service.settings.version,
        "mode": "выдержки",
        "depth": DEPTH,
        "citations_limit": service.settings.citations,
        "labeled_queries": len(labeled),
        "rule": {
            "text": "корректная ссылка это цитата из фрагмента с оценкой эталона больше нуля, "
            "знаменатель это выданные ответы",
            "written_before": "orchestration/EXPERIMENTS.md, S24",
        },
        "degraded": search.degraded,
    }
    report["configs"] = {
        name: citations_quality(service, labeled, qrels, name) for name in configs
    }
    if "bm25" in configs and catalog.default in configs and catalog.default != "bm25":
        report["cross_support"] = cross_support(
            service, labeled, qrels, "bm25", catalog.default
        )

    human = human_section(args.part_a, qrels)
    if human is None:
        previous = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
        carried = previous.get("human")
        if carried is None:
            print(
                f"нет файлов части А ({args.part_a}), и прежнего отчета рядом тоже нет: "
                "человеческий раздел не выдумывается",
                file=sys.stderr,
            )
            return 2
        human = {**carried, "from_previous": True}
    report["human"] = human

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, block in report["configs"].items():
        strict = block["ссылка на релевантный фрагмент"]
        print(
            f"{name}: ссылка на релевантный фрагмент {strict['hit']} из {strict['of']} "
            f"({strict['share']}), отказов {block['refused']} из {block['queries']}"
        )
    print(f"записано в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
