#!/usr/bin/env python3
"""Сколько стоит нормализация формы слова в нашем BM25.

Вопрос пришел из контрольного прогона на чужих данных (`friend_control.py`):
там наш BM25 отставал от опубликованного чужого на 0,068 NDCG@10, и отставание
снималось одним изменением, обрезкой токена до шести символов.

Здесь на нашем корпусе и нашей разметке сравниваются три состояния поиска:
разбиение без снятия формы, обрезка токена нескольких длин (диагностика,
которой цена была измерена первой) и стеммер Snowball (решение 16, рабочая
токенизация). Все три идут на рабочем разбиении, то есть пересчитываются
вместе с ним: ось здесь нормализация формы, а не разбор токена, и его замер
лежит отдельно (`token_numbers_effect.py`). Несколько длин обрезки считаются
нарочно, чтобы число не выглядело подобранным, и стеммер сравнивается с
лучшей из них напрямую, парным бутстрэпом по запросам.

Отдельно считается то, чем нормализация может навредить: совпадение по номеру
акта. На нем держится прибавка фильтра из задачи 3.4, и если номера страдают,
выигрыш на словах уйдет на потере на адресах.

Видеокарта не нужна, все считается на маке.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.references import act_numbers  # noqa: E402
from rufin.retrieval.bm25 import BM25Index  # noqa: E402
from rufin.retrieval.text import (  # noqa: E402
    split_tokens,
    tokenize,
    truncate_tokens,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
REPORT = os.path.join(ROOT, "docs", "raw", "morphology_effect.txt")
RESULT = os.path.join(ROOT, "data", "results", "morphology.json")

TOP = 10
METRICS = ("Recall@10", "MRR@10", "NDCG@10")


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--lengths", type=int, nargs="*", default=[5, 6, 7])
    ap.add_argument("--subsets", nargs="*", default=["заголовочная", "живые"])
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    ids, texts = [], []
    for rec in read_jsonl(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl")):
        ids.append(rec["chunk_id"])
        texts.append(rec["text"])
    qrels = read_qrels(os.path.join(QDIR, "qrels.tsv"))
    queries = {q["query_id"]: q["text"]
               for q in read_jsonl(os.path.join(QDIR, "queries.jsonl"))}
    qids = sorted(q for q in qrels if q in queries)
    subsets = json.load(open(os.path.join(QDIR, "podvyborki.json"), encoding="utf-8"))
    parts = {"все": qids}
    for name in args.subsets:
        parts[name] = sorted(q for q in subsets[name] if q in qrels and q in queries)
    # Запросы, где назван номер акта: на них обрезка опаснее всего.
    with_number = sorted(q for q in qids if act_numbers(queries[q]))
    parts["с номером акта"] = with_number

    say(f"нарезка {args.chunks}: фрагментов {len(ids)}")
    say(f"разметка: запросов {len(qids)}, из них с номером акта {len(with_number)}")
    for name, part in parts.items():
        if name != "все":
            say(f"  подвыборка «{name}»: {len(part)}")
    say()

    # «как есть» это разбиение без снятия формы. До решения 16 поиск был именно
    # таким, но то число снято на разбиении, которое с тех пор починено, и
    # прежняя линия лежит в `results/regression-baseline.json` части Б строкой
    # «replaced» со своей датой, а не подставляется сюда
    variants = {"как есть": split_tokens}
    for n in args.lengths:
        variants[f"обрезка до {n}"] = (lambda n: lambda t: truncate_tokens(t, n))(n)
    variants["стеммер"] = tokenize

    runs: dict[str, dict[str, list[str]]] = {}
    # Цена нормализации в задержке меряется здесь же, тем же кодом и на тех же
    # запросах: иначе «стало медленнее» пришлось бы сравнивать замеры,
    # снятые в разные дни и на разной загрузке машины.
    index_stats: dict[str, dict] = {}
    for name, tok in variants.items():
        index = BM25Index.build(ids, texts, tokenizer=tok)
        for q in qids[:20]:
            index.search(queries[q], TOP)
        timings = []
        for q in qids:
            t0 = time.perf_counter()
            hits = index.search(queries[q], TOP)
            timings.append((time.perf_counter() - t0) * 1000)
            runs.setdefault(name, {})[q] = [c for c, _ in hits]
        index_stats[name] = {"словарь": len(index.vocab), "индекс, МБ": round(index.size_mb, 1),
                             "сборка, с": round(index.build_seconds, 1),
                             "запрос, мс, медиана": round(statistics.median(timings), 2),
                             "запрос, мс, p95": round(sorted(timings)[int(0.95 * len(timings))], 2)}
        say(f"{name:<14} словарь {len(index.vocab):>7} слов, индекс "
            f"{index.size_mb:>3.0f} МБ, сборка {index.build_seconds:.0f} с, "
            f"запрос {index_stats[name]['запрос, мс, медиана']:.2f} мс, "
            f"p95 {index_stats[name]['запрос, мс, p95']:.2f} мс")
        del index

    per_q = {part_name: {name: M.per_query({q: runs[name][q] for q in part},
                                           {q: qrels[q] for q in part})
                         for name in runs}
             for part_name, part in parts.items()}

    levels: dict[str, dict] = {}
    for part_name, part in parts.items():
        say()
        say(f"уровни, подвыборка «{part_name}» ({len(part)} запросов):")
        say(f"{'вариант':<14} " + " ".join(f"{m:>24}" for m in METRICS))
        levels[part_name] = {}
        for name in runs:
            cis = {m: M.bootstrap_ci(per_q[part_name][name][m]) for m in METRICS}
            say(f"{name:<14} " + " ".join(f"{str(cis[m]):>24}" for m in METRICS))
            levels[part_name][name] = {
                m: {"mean": round(ci.mean, 4), "lo": round(ci.lo, 4), "hi": round(ci.hi, 4)}
                for m, ci in cis.items()}

    def compare(against: str, header: str) -> list[dict]:
        say()
        say(header)
        rows = []
        for name in runs:
            if name == against:
                continue
            for part_name, part in parts.items():
                row = {"вариант": name, "против": against,
                       "подвыборка": part_name, "запросов": len(part)}
                out = []
                for m in METRICS:
                    diff, p = M.paired_diff_ci(per_q[part_name][name][m],
                                               per_q[part_name][against][m])
                    out.append(f"{m} {diff} p={p:.3f}" + (" *" if diff.lo * diff.hi > 0 else ""))
                    row[m] = {"diff": round(diff.mean, 4), "lo": round(diff.lo, 4),
                              "hi": round(diff.hi, 4), "p": round(p, 4)}
                say(f"  {name:<14} {part_name:<16} ({len(part):>3}) " + "   ".join(out))
                rows.append(row)
        return rows

    diffs = compare("как есть", "прибавка к варианту «как есть», парный бутстрэп по запросам:")
    # Главное сравнение решения 16: стеммер против той обрезки, которая вышла
    # лучшей. Без него выбор между ними опирался бы на разность двух
    # независимых интервалов, а она шире парной разницы
    diffs_vs_truncation = []
    reference = "обрезка до 6"
    if reference in runs and "стеммер" in runs:
        for part_name, part in parts.items():
            row = {"вариант": "стеммер", "против": reference,
                   "подвыборка": part_name, "запросов": len(part)}
            out = []
            for m in METRICS:
                diff, p = M.paired_diff_ci(per_q[part_name]["стеммер"][m],
                                           per_q[part_name][reference][m])
                out.append(f"{m} {diff} p={p:.3f}" + (" *" if diff.lo * diff.hi > 0 else ""))
                row[m] = {"diff": round(diff.mean, 4), "lo": round(diff.lo, 4),
                          "hi": round(diff.hi, 4), "p": round(p, 4)}
            diffs_vs_truncation.append(row)
        say()
        say(f"стеммер против «{reference}», парный бутстрэп по запросам:")
        for row in diffs_vs_truncation:
            out = "   ".join(
                f"{m} {row[m]['diff']:+.3f} [{row[m]['lo']:+.3f}; {row[m]['hi']:+.3f}] "
                f"p={row[m]['p']:.3f}" + (" *" if row[m]["lo"] * row[m]["hi"] > 0 else "")
                for m in METRICS)
            say(f"  {row['подвыборка']:<16} ({row['запросов']:>3}) " + out)

    # Чем обрезка может навредить: номер акта перестает быть отдельным словом.
    # Считается прямо, а не предполагается: сколько номеров из вопросов
    # склеиваются с другими токенами после обрезки.
    say()
    numbers = sorted({n for q in with_number for n in act_numbers(queries[q])})

    def glued(key_of) -> int:
        buckets: dict[str, set[str]] = {}
        for number in numbers:
            parts_ = split_tokens(number)
            buckets.setdefault(key_of(parts_[0]) if parts_ else "", set()).add(number)
        return sum(len(v) for v in buckets.values() if len(v) > 1)

    # «Как есть» в этом счете обязателен. Без него склейку легко записать на
    # счет нормализации, а она тут ни при чем: до 23.10.2026 пять писем вида
    # «ИН-03-23/109» сходились в токен «ин» уже на разбиении, и это починено
    # отдельно (`token_numbers_effect.py`). Счет остается затем, что обрезка
    # склеивать номера все еще может, и это надо видеть числом.
    collisions = {"как есть": glued(lambda t: t)}
    say(f"как есть: номеров актов из вопросов {len(numbers)}, "
        f"склеилось с другим номером {collisions['как есть']}")
    for n in args.lengths:
        collisions[str(n)] = glued(lambda t, n=n: t[:n])
        say(f"обрезка до {n}: номеров актов из вопросов {len(numbers)}, "
            f"склеилось с другим номером {collisions[str(n)]}")
    collisions["стеммер"] = glued(lambda t: tokenize(t)[0] if tokenize(t) else "")
    say(f"стеммер: номеров актов из вопросов {len(numbers)}, "
        f"склеилось с другим номером {collisions['стеммер']}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump({"нарезка": args.chunks, "запросов": {k: len(v) for k, v in parts.items()},
                   "индексы": index_stats, "уровни": levels, "прибавка": diffs,
                   "стеммер против обрезки": diffs_vs_truncation,
                   "склейка номеров актов": collisions},
                  f, ensure_ascii=False, indent=2)
    say()
    say(f"файлы: {os.path.relpath(REPORT, ROOT)}, {os.path.relpath(RESULT, ROOT)}")


if __name__ == "__main__":
    main()
