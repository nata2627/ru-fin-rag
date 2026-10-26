#!/usr/bin/env python3
"""Сколько стоит разбор номера акта в токенизации BM25.

Дефект нашелся побочно, при замере стеммера (`morphology_effect.py`): из 24
номеров актов, названных в эталонных вопросах, 5 рассыпались в разбиении, и
сначала это списали на обрезку токена. Пересчет показал, что те же 5
рассыпаются и без всякой нормализации. Причина в регулярном выражении токена:
после дефиса оно брало только буквы и не больше трех, поэтому письмо
«ИН-03-23/109» давало токен «ин».

Здесь измеряются два следствия одной починки, раздельно, иначе они слиплись бы
в одно число:

1. «номера целиком» — формы номера добавлены отдельной ветвью, старая оговорка
   про дефис оставлена. Меняет только номера.
2. «разбор по дефису» — та же ветвь, но оговорка `{1,3}` убрана как ненужная.
   Заодно перестает резать дефисные русские слова («кредитно-денежный» давало
   «кредитно-ден» и «ежный»), а их в корпусе 30 800 вхождений.

Опора это сегодняшнее состояние поиска, то есть стеммер поверх прежнего
разбиения. Стеммер стоит во всех трех вариантах: мерится разбор, а не
нормализация, и смешивать два изменения в одном сравнении нельзя.

Подвыборки те же, что у замера морфологии, плюс одна своя: те вопросы, в
которых номер рассыпался. Их четыре, и это написано рядом с числом, потому что
на четырех запросах интервал шире любого эффекта.

Видеокарта не нужна, все считается на маке.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.references import act_numbers  # noqa: E402
from rufin.retrieval.bm25 import BM25Index  # noqa: E402
from rufin.retrieval.text import STOPWORDS, split_tokens, stem  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
REPORT = os.path.join(ROOT, "docs", "raw", "token_numbers_effect.txt")
RESULT = os.path.join(ROOT, "data", "results", "token-numbers.json")

TOP = 10
METRICS = ("Recall@10", "MRR@10", "NDCG@10")

# Прежнее выражение токена, то, что стояло в поиске до 23.10.2026. Живет здесь,
# а не в `rufin.retrieval.text`: в рабочем коде ему делать нечего, а без него
# починку не с чем сравнить.
_PREVIOUS = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z]{1,3})?", re.I)
# Промежуточное состояние: формы номера добавлены, прежняя оговорка оставлена.
# Нужно затем, чтобы отделить прибавку от номеров от прибавки от дефисных слов.
_NUMBERS_ONLY = re.compile(
    r"(?:\d+(?:/\d+)?-[а-яa-z]{1,3}(?![а-яa-z])|[а-яa-z]{2,4}-\d+(?:[-/]\d+)+)"
    r"|[а-яёa-z0-9]+(?:-[а-яёa-z]{1,3})?",
    re.I)


def make_splitter(token: re.Pattern[str]):
    """Разбиение тем же телом, что в рабочем коде, но с другим выражением."""

    def split(text: str) -> list[str]:
        out = []
        for raw in token.findall(text):
            low = raw.lower().replace("ё", "е")
            if low not in STOPWORDS:
                out.append(low)
        return out

    return split


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def shape(number: str) -> str:
    """Форма номера: цифры в «9», заглавные в «A», строчные в «a».

    Нужна затем, чтобы перечень форм выводился из данных числом, а не из
    памяти о том, какие номера бывают.
    """
    out = []
    for ch in number:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A" if ch.isupper() else "a")
        else:
            out.append(ch)
    text = "".join(out)
    for symbol in ("9", "A", "a"):
        text = re.sub(f"{symbol}+", f"{symbol}+", text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
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

    previous = make_splitter(_PREVIOUS)
    variants = {
        "прежний разбор": previous,
        "номера целиком": make_splitter(_NUMBERS_ONLY),
        "разбор по дефису": split_tokens,
    }

    # Перечень форм выводится из данных: из поля `number` у актов корпуса и из
    # номеров, названных в самих вопросах.
    act_forms = collections.Counter()
    for act in read_jsonl(os.path.join(ROOT, "data", "corpus", "acts.jsonl")):
        act_forms[shape(act["number"])] += 1
    per_query = {q: act_numbers(queries[q]) for q in qids}
    numbers = sorted({n for q in qids for n in per_query[q]})
    query_forms = collections.Counter(shape(n) for n in numbers)

    def whole(split, number: str) -> bool:
        return split(number) == [number.lower().replace("ё", "е")]

    broken = {n for n in numbers if not whole(previous, n)}
    still_broken = {n for n in numbers if not whole(split_tokens, n)}

    parts = {"все": qids}
    for name in args.subsets:
        parts[name] = sorted(q for q in subsets[name] if q in qrels and q in queries)
    parts["с номером акта"] = sorted(q for q in qids if per_query[q])
    parts["номер рассыпался"] = sorted(
        q for q in qids if any(n in broken for n in per_query[q]))

    say(f"нарезка {args.chunks}: фрагментов {len(ids)}")
    say(f"разметка: запросов {len(qids)}")
    for name, part in parts.items():
        if name != "все":
            say(f"  подвыборка «{name}»: {len(part)}")
    say()
    say("формы номера в поле `number` у актов корпуса:")
    for form, count in act_forms.most_common():
        say(f"  {form:<14} {count} из {sum(act_forms.values())}")
    say("формы номеров, названных в вопросах:")
    for form, count in query_forms.most_common():
        example = next(n for n in numbers if shape(n) == form)
        say(f"  {form:<14} {count} из {len(numbers)}  например {example}")
    say(f"прежний разбор рассыпал {len(broken)} из {len(numbers)}: "
        f"{', '.join(sorted(broken))}")
    say(f"новый разбор рассыпает {len(still_broken)} из {len(numbers)}")
    # Номер, которого нет ни в одном фрагменте, целым токеном ничего не найдет.
    # Это надо знать до того, как читать метрику: половина ожидаемой прибавки
    # снимается здесь, а не в замере.
    in_corpus = dict.fromkeys(sorted(broken), 0)
    wanted = {n: n.lower() for n in in_corpus}
    for text in texts:
        low = text.lower()
        for number, needle in wanted.items():
            if needle in low:
                in_corpus[number] += 1
    for number, count in in_corpus.items():
        say(f"  {number}: фрагментов корпуса с этим номером {count} из {len(ids)}")
    say()

    runs: dict[str, dict[str, list[str]]] = {}
    index_stats: dict[str, dict] = {}
    for name, split in variants.items():
        def tok(text: str, split=split) -> list[str]:
            return [stem(t) for t in split(text)]

        index = BM25Index.build(ids, texts, tokenizer=tok)
        for q in qids[:20]:
            index.search(queries[q], TOP)
        timings = []
        for q in qids:
            t0 = time.perf_counter()
            hits = index.search(queries[q], TOP)
            timings.append((time.perf_counter() - t0) * 1000)
            runs.setdefault(name, {})[q] = [c for c, _ in hits]
        index_stats[name] = {
            "словарь": len(index.vocab), "индекс, МБ": round(index.size_mb, 1),
            "сборка, с": round(index.build_seconds, 1),
            "запрос, мс, медиана": round(statistics.median(timings), 2),
            "запрос, мс, p95": round(sorted(timings)[int(0.95 * len(timings))], 2)}
        say(f"{name:<18} словарь {len(index.vocab):>7} слов, индекс "
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
        say(f"{'вариант':<18} " + " ".join(f"{m:>24}" for m in METRICS))
        levels[part_name] = {}
        for name in runs:
            cis = {m: M.bootstrap_ci(per_q[part_name][name][m]) for m in METRICS}
            say(f"{name:<18} " + " ".join(f"{str(cis[m]):>24}" for m in METRICS))
            levels[part_name][name] = {
                m: {"mean": round(ci.mean, 4), "lo": round(ci.lo, 4), "hi": round(ci.hi, 4)}
                for m, ci in cis.items()}

    diffs = []
    against = "прежний разбор"
    say()
    say(f"прибавка к варианту «{against}», парный бутстрэп по запросам:")
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
            say(f"  {name:<18} {part_name:<18} ({len(part):>3}) " + "   ".join(out))
            diffs.append(row)

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump({"нарезка": args.chunks,
                   "запросов": {k: len(v) for k, v in parts.items()},
                   "формы номера у актов": dict(act_forms),
                   "формы номеров в вопросах": dict(query_forms),
                   "номеров в вопросах": len(numbers),
                   "рассыпалось прежним разбором": sorted(broken),
                   "рассыпается новым разбором": sorted(still_broken),
                   "фрагментов корпуса с этим номером": in_corpus,
                   "индексы": index_stats, "уровни": levels, "прибавка": diffs},
                  f, ensure_ascii=False, indent=2)
    say()
    say(f"файлы: {os.path.relpath(REPORT, ROOT)}, {os.path.relpath(RESULT, ROOT)}")


if __name__ == "__main__":
    main()
