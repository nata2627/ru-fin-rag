#!/usr/bin/env python3
"""Контрольный прогон: наш код на чужих данных.

Вопрос один. На нормативке Банка России наш поиск дает 0,484 NDCG@10 у BM25
и 0,421 у лучшей связки на живых вопросах, а в наборе того же жанра у соседа
по тем же метрикам выходит 0,670 и 0,859. Это разница предметной области или
ошибка в нашем коде? Проверяется прямо: тот же наш BM25, те же наши метрики,
но на его корпусе, его запросах и его разметке.

Набор `alekseevpavel04/ru-law-retrieval` в формате MTEB (`fetch_friend_set.py`).
Его протокол «chunk»: индексируются фрагменты, разметка стоит на статьях,
поэтому выдача фрагментов поднимается к статье и повторы снимаются. Это
ровно тот подъем к родителю, что в `rufin.parents`, только родитель здесь
назван в самом фрагменте полем `doc_id`, и считать пересечения не нужно.

Рядом считается протокол «article»: индексируются статьи целиком. Разница
двух протоколов на одних данных это цена нарезки у него, и она сравнима с
нашим замером нарезки (документ 13).

Его опубликованные числа берутся из карточки набора и помечены как чужие:
наш код их не считал.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.retrieval.bm25 import BM25Index  # noqa: E402
from rufin.retrieval.text import tokenize, truncate_tokens  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRIEND = os.path.join(ROOT, "data", "friend")
OURS = os.path.join(ROOT, "data")
REPORT = os.path.join(ROOT, "docs", "raw", "friend_control.txt")
RESULT = os.path.join(ROOT, "data", "results", "friend-control.json")

DEEP = 200
TOP = 10
SPLITS = ("test", "golden", "tk_hard")

# Чужие числа из карточки набора huggingface.co/datasets/alekseevpavel04/
# ru-law-retrieval, протокол chunk, сплит test. Нашим кодом не считались и
# служат только точкой сравнения для строки bm25.
HIS = {
    "bm25": 0.670,
    "multilingual-e5-small-ru-law (дообученная им)": 0.859,
    "e5-large-instruct": 0.865,
    "FRIDA": 0.878,
}
# Его же BM25 в разбивке, из README репозитория ru-law-retrieval. Разбивка
# важнее уровня: она показывает, где именно наш BM25 отстает.
HIS_BY_TYPE = {"everyday": 0.265, "search": 0.822, "legal": 0.922}
HIS_TK_HARD = 0.480

# Две токенизации. Первая рабочая, вторая диагностическая: она отвечает на
# один вопрос, в морфологии ли дело.
TOKENIZERS = {"как есть": tokenize, "обрезка до 6": truncate_tokens}


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for rec in read_jsonl(path):
        score = int(rec["score"])
        if score > 0:
            out[rec["query-id"]][rec["corpus-id"]] = score
    return dict(out)


def run_split(index: BM25Index, parent_of: dict[str, str] | None,
              queries: dict[str, str], qrels: dict[str, dict[str, int]]):
    """Выдача по сплиту: поиск, подъем к статье, если он нужен, снятие повторов."""
    run = {}
    for qid in qrels:
        hits = [c for c, _ in index.search(queries[qid], DEEP)]
        if parent_of is None:
            run[qid] = hits[:TOP]
            continue
        out, seen = [], set()
        for cid in hits:
            doc = parent_of[cid]
            if doc in seen:
                continue
            seen.add(doc)
            out.append(doc)
            if len(out) == TOP:
                break
        run[qid] = out
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    chunks = list(read_jsonl(os.path.join(FRIEND, "chunks", "chunks.jsonl")))
    corpus = list(read_jsonl(os.path.join(FRIEND, "corpus", "corpus.jsonl")))
    say("контрольный прогон нашего кода на наборе alekseevpavel04/ru-law-retrieval")
    say(f"его корпус: статей {len(corpus)}, фрагментов {len(chunks)}")

    parent_of = {c["_id"]: c["doc_id"] for c in chunks}
    indexes = {}
    for tname, tok in TOKENIZERS.items():
        indexes[("chunk", tname)] = (
            BM25Index.build([c["_id"] for c in chunks], [c["text"] for c in chunks],
                            tokenizer=tok), parent_of)
        indexes[("article", tname)] = (
            BM25Index.build([d["_id"] for d in corpus],
                            [f"{d['title']}\n\n{d['text']}" for d in corpus],
                            tokenizer=tok), None)
    for (protocol, tname), (index, _) in indexes.items():
        say(f"индекс BM25, протокол {protocol}, токенизация «{tname}»: "
            f"{index.size_mb:.0f} МБ, слов в словаре {len(index.vocab)}")

    results: dict[str, dict] = {}
    say()
    for split in args.splits:
        queries = {q["_id"]: q["text"]
                   for q in read_jsonl(os.path.join(FRIEND, "queries", f"{split}.jsonl"))}
        qrels = load_qrels(os.path.join(FRIEND, "qrels", f"{split}.jsonl"))
        missing = [q for q in qrels if q not in queries]
        if missing:
            raise SystemExit(f"{split}: нет текста запроса у {len(missing)}, первый {missing[0]}")
        golds = [len(v) for v in qrels.values()]
        say(f"сплит {split}: запросов с разметкой {len(qrels)}, "
            f"эталонов на запрос медиана {statistics.median(golds):.0f}, "
            f"максимум {max(golds)}")
        results[split] = {"запросов": len(qrels),
                          "эталонов на запрос, медиана": statistics.median(golds)}
        qtype = {q["_id"]: q.get("qtype", "")
                 for q in read_jsonl(os.path.join(FRIEND, "queries", f"{split}.jsonl"))}
        for (protocol, tname), (index, parents) in indexes.items():
            run = run_split(index, parents, queries, qrels)
            per_q = M.per_query(run, qrels)
            qids = list(per_q["_qids"])
            mrr5 = np.array([M.mrr_at_k(run[q], qrels[q], 5) for q in qids])
            cis = {m: M.bootstrap_ci(per_q[m]) for m in ("Recall@10", "NDCG@10")}
            cis["MRR@5"] = M.bootstrap_ci(mrr5)
            say(f"    {protocol:<8} «{tname:<12}» "
                + "   ".join(f"{k} {v}" for k, v in cis.items()))
            results[split][f"{protocol}, {tname}"] = {
                k: {"mean": round(v.mean, 4), "lo": round(v.lo, 4), "hi": round(v.hi, 4)}
                for k, v in cis.items()}
            # разбивка по типам запроса: у него она опубликована, и именно
            # в ней видно, где расходятся реализации
            by_type = {}
            for t in sorted({qtype[q] for q in qids if qtype[q]}):
                part = [q for q in qids if qtype[q] == t]
                vals = np.array([M.ndcg_at_k(run[q], qrels[q], 10) for q in part])
                ci = M.bootstrap_ci(vals)
                by_type[t] = {"запросов": len(part), "mean": round(ci.mean, 4),
                              "lo": round(ci.lo, 4), "hi": round(ci.hi, 4)}
            if by_type:
                say("             по типам: " + "   ".join(
                    f"{t} {v['mean']:.3f} ({v['запросов']})" for t, v in by_type.items()))
                results[split][f"{protocol}, {tname}"]["NDCG@10 по типам"] = by_type

    # Сравнение с его опубликованными числами идет только там, где он
    # публиковал ту же связку на том же сплите: BM25, протокол chunk, test.
    say()
    say("его опубликованные числа (карточка набора и README, протокол chunk, сплит test):")
    for name, value in HIS.items():
        say(f"    {name:<46} NDCG@10 {value:.3f}")
    say("    его BM25 по типам: " + "   ".join(
        f"{t} {v:.3f}" for t, v in HIS_BY_TYPE.items()) + f";  tk_hard {HIS_TK_HARD:.3f}")
    for tname in TOKENIZERS:
        key = f"chunk, {tname}"
        ours = results.get("test", {}).get(key, {}).get("NDCG@10", {}).get("mean")
        if ours is None:
            continue
        say(f"наш BM25, токенизация «{tname}»: NDCG@10 {ours:.3f} против его "
            f"{HIS['bm25']:.3f}, расхождение {ours - HIS['bm25']:+.3f}")

    # Различия корпусов. Считаются здесь же по обоим наборам, чтобы таблица
    # в документе не переписывалась руками.
    say()
    our_chunks = list(read_jsonl(os.path.join(OURS, "chunks", "base.jsonl")))
    our_qrels = read_qrels(os.path.join(OURS, "queries", "qrels.tsv"))
    our_golds = {q: sum(1 for g in rel.values() if g > 0) for q, rel in our_qrels.items()}
    diff = {
        "документов (акты против статей)": (len({c["act_id"] for c in our_chunks}), len(corpus)),
        "фрагментов": (len(our_chunks), len(chunks)),
        "символов во фрагменте, медиана": (
            int(statistics.median(len(c["text"]) for c in our_chunks)),
            int(statistics.median(len(c["text"]) for c in chunks))),
        "слов во фрагменте после токенизации, медиана": (
            int(statistics.median(len(tokenize(c["text"])) for c in our_chunks)),
            int(statistics.median(len(tokenize(c["text"])) for c in chunks))),
        "запросов с разметкой": (len(our_golds), results.get("test", {}).get("запросов", 0)),
        "эталонов на запрос, медиана": (
            statistics.median(our_golds.values()),
            results.get("test", {}).get("эталонов на запрос, медиана", 0)),
    }
    say(f"{'различие':<46} {'наш корпус':>12} {'его корпус':>12}")
    for name, (a, b) in diff.items():
        say(f"{name:<46} {a:>12} {b:>12}")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump({"набор": "alekseevpavel04/ru-law-retrieval",
                   "глубина чтения": DEEP, "позиций в выдаче": TOP,
                   "его числа из карточки": HIS,
                   "наши числа": results,
                   "различия корпусов": {k: {"наш": v[0], "его": v[1]}
                                         for k, v in diff.items()}},
                  f, ensure_ascii=False, indent=2)
    say()
    say(f"файлы: {os.path.relpath(REPORT, ROOT)}, {os.path.relpath(RESULT, ROOT)}")


if __name__ == "__main__":
    main()
