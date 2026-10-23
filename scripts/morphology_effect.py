#!/usr/bin/env python3
"""Сколько стоит отсутствие нормализации формы слова в нашем BM25.

Вопрос пришел из контрольного прогона на чужих данных (`friend_control.py`):
там наш BM25 отставал от опубликованного чужого на 0,068 NDCG@10, и отставание
снималось одним изменением, обрезкой токена до шести символов. Здесь та же
обрезка мерится на нашем корпусе и на нашей разметке.

Обрезка это диагностика, а не предложение. Она склеивает падежные формы без
словаря и стеммера, но заодно склеивает разные слова, поэтому в рабочий поиск
она не идет: сначала надо знать цену вопроса. Несколько длин обрезки считаются
нарочно, чтобы число не выглядело подобранным.

Отдельно считается то, чем обрезка может навредить: совпадение по номеру акта.
На нем держится прибавка фильтра из задачи 3.4, и если номера от обрезки
страдают, выигрыш на словах уйдет на потере на адресах.

Видеокарта не нужна, все считается на маке.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.references import act_numbers  # noqa: E402
from rufin.retrieval.bm25 import BM25Index  # noqa: E402
from rufin.retrieval.text import tokenize, truncate_tokens  # noqa: E402

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

    variants = {"как есть": tokenize}
    for n in args.lengths:
        variants[f"обрезка до {n}"] = (lambda n: lambda t: truncate_tokens(t, n))(n)

    runs: dict[str, dict[str, list[str]]] = {}
    for name, tok in variants.items():
        index = BM25Index.build(ids, texts, tokenizer=tok)
        runs[name] = {q: [c for c, _ in index.search(queries[q], TOP)] for q in qids}
        say(f"{name:<14} словарь {len(index.vocab):>7} слов, индекс "
            f"{index.size_mb:>3.0f} МБ, сборка {index.build_seconds:.0f} с")
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

    say()
    say("прибавка к варианту «как есть», парный бутстрэп по запросам:")
    diffs = []
    for name in runs:
        if name == "как есть":
            continue
        for part_name, part in parts.items():
            row = {"вариант": name, "подвыборка": part_name, "запросов": len(part)}
            out = []
            for m in METRICS:
                diff, p = M.paired_diff_ci(per_q[part_name][name][m],
                                           per_q[part_name]["как есть"][m])
                out.append(f"{m} {diff} p={p:.3f}" + (" *" if diff.lo * diff.hi > 0 else ""))
                row[m] = {"diff": round(diff.mean, 4), "lo": round(diff.lo, 4),
                          "hi": round(diff.hi, 4), "p": round(p, 4)}
            say(f"  {name:<14} {part_name:<16} ({len(part):>3}) " + "   ".join(out))
            diffs.append(row)

    # Чем обрезка может навредить: номер акта перестает быть отдельным словом.
    # Считается прямо, а не предполагается: сколько номеров из вопросов
    # склеиваются с другими токенами после обрезки.
    say()
    numbers = sorted({n for q in with_number for n in act_numbers(queries[q])})
    collisions = {}
    for n in args.lengths:
        buckets: dict[str, set[str]] = {}
        for number in numbers:
            key = tokenize(number)[0][:n] if tokenize(number) else ""
            buckets.setdefault(key, set()).add(number)
        collisions[n] = sum(len(v) for v in buckets.values() if len(v) > 1)
        say(f"обрезка до {n}: номеров актов из вопросов {len(numbers)}, "
            f"склеилось с другим номером {collisions[n]}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump({"нарезка": args.chunks, "запросов": {k: len(v) for k, v in parts.items()},
                   "уровни": levels, "прибавка": diffs,
                   "склейка номеров актов": collisions},
                  f, ensure_ascii=False, indent=2)
    say()
    say(f"файлы: {os.path.relpath(REPORT, ROOT)}, {os.path.relpath(RESULT, ROOT)}")


if __name__ == "__main__":
    main()
