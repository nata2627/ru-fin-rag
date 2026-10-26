#!/usr/bin/env python3
"""Насколько слова вопроса совпадают со словами его эталонного фрагмента.

Замер отвечает на один вопрос: далеко ли живой вопрос практика от вопроса,
который модель написала по фрагменту. Вопрос, написанный по фрагменту,
наследует его слова, поэтому доля слов вопроса, найденных в эталоне, у него
обязана быть выше. Насколько выше, и есть цена речевого режима.

Доля считается по рабочей токенизации BM25 (со стеммером) и по ней же без
стеммера: разница между этими двумя столбцами показывает, какая часть разрыва
это падежи, а какая разные слова.

Знаменатель у живых вопросов неполон (документ 11): часть относящихся
к делу фрагментов не размечена никем, и доля по эталону занижает совпадение.
Поэтому измеренный разрыв это оценка сверху.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.retrieval.text import split_tokens, tokenize  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
REPORT = os.path.join(ROOT, "docs", "raw", "lexical_gap.txt")
RESULT = os.path.join(ROOT, "data", "results", "lexical-gap.json")

# подвыборки, вопросы которых написала языковая модель
MODEL_WRITTEN = ("синтетика", "dev", "невиданные акты")


def share(query: str, gold_texts: list[str], stem: bool) -> float:
    """Доля различных слов вопроса, встречающихся в объединении эталонов."""
    tok = tokenize if stem else split_tokens
    q = set(tok(query))
    if not q:
        return float("nan")
    g: set[str] = set()
    for t in gold_texts:
        g |= set(tok(t))
    return len(q & g) / len(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--run", default=os.path.join(ROOT, "data", "runs",
                                                  "base__hybrid-rerank.jsonl"),
                    help="выдача, по которой живые вопросы делятся на провалы и остальные")
    args = ap.parse_args()

    chunks = {}
    for line in open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"), encoding="utf-8"):
        c = json.loads(line)
        chunks[c["chunk_id"]] = c["text"]

    queries = {q["query_id"]: q["text"] for q in
               (json.loads(l) for l in open(os.path.join(QDIR, "queries.jsonl"),
                                            encoding="utf-8"))}
    qrels = read_qrels(os.path.join(QDIR, "qrels.tsv"))
    subsets = json.load(open(os.path.join(QDIR, "podvyborki.json"), encoding="utf-8"))
    run = {r["query_id"]: r["ranked"] for r in
           (json.loads(l) for l in open(args.run, encoding="utf-8"))}

    def shares(qids: list[str], stem: bool, against: str = "эталон") -> np.ndarray:
        out = []
        for qid in qids:
            gold = [chunks[c] for c, g in qrels.get(qid, {}).items()
                    if g > 0 and c in chunks]
            if not gold:
                continue
            if against == "эталон":
                texts = gold
            else:
                texts = [chunks[c] for c in run.get(qid, [])[:1] if c in chunks]
            out.append(share(queries[qid], texts, stem) if texts else float("nan"))
        return np.array(out, dtype=float)

    model_written = sorted({q for name in MODEL_WRITTEN for q in subsets[name]})
    live = sorted(subsets["живые"])
    failed = [q for q in live
              if (lambda rel: rel and not (rel & set(run[q][:10])))(
                  {c for c, g in qrels.get(q, {}).items() if g > 0})]
    rest = [q for q in live if q not in set(failed)]

    groups = {
        "написанные моделью": model_written,
        "живые": live,
        "живые, провалы связки": failed,
        "живые, остальные": rest,
    }

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say(f"выдача для деления живых на провалы: {os.path.relpath(args.run, ROOT)}")
    say("доля слов вопроса, встречающихся в его эталонном фрагменте")
    say()
    say(f"{'группа':<26}{'запросов':>9}  {'эталон, стеммер':<24}"
        f"{'эталон, без стеммера':<24}{'первый найденный':<24}")
    levels = {}
    for name, qids in groups.items():
        with_stem = shares(qids, True)
        no_stem = shares(qids, False)
        found = shares(qids, True, against="первый найденный")
        a = M.bootstrap_ci(with_stem)
        b = M.bootstrap_ci(no_stem)
        c = M.bootstrap_ci(found)
        say(f"{name:<26}{with_stem.size:>9}  {str(a):<24}{str(b):<24}{str(c):<24}")
        levels[name] = {
            "запросов": int(with_stem.size),
            "со стеммером": {"mean": round(a.mean, 4), "lo": round(a.lo, 4),
                             "hi": round(a.hi, 4)},
            "без стеммера": {"mean": round(b.mean, 4), "lo": round(b.lo, 4),
                             "hi": round(b.hi, 4)},
            "первый найденный, со стеммером": {"mean": round(c.mean, 4), "lo": round(c.lo, 4),
                                               "hi": round(c.hi, 4)},
        }

    say()
    say("разрыв, непарный бутстрэп (группы из разных запросов, пары нет):")
    diffs = []
    for left, right in (("написанные моделью", "живые"),
                        ("живые, остальные", "живые, провалы связки")):
        row = {"слева": left, "справа": right}
        parts = []
        for label, stem in (("со стеммером", True), ("без стеммера", False)):
            d = M.unpaired_diff_ci(shares(groups[left], stem), shares(groups[right], stem))
            parts.append(f"{label} {d}")
            row[label] = {"diff": round(d.mean, 4), "lo": round(d.lo, 4), "hi": round(d.hi, 4)}
        say(f"   {left} минус {right}: " + "   ".join(parts))
        diffs.append(row)

    say()
    say("на одних и тех же запросах: слова вопроса в первом найденном фрагменте "
        "против слов в эталоне (парный бутстрэп)")
    paired = []
    for name in ("живые, провалы связки", "живые, остальные", "написанные моделью"):
        d, p = M.paired_diff_ci(shares(groups[name], True, against="первый найденный"),
                                shares(groups[name], True))
        say(f"   {name:<26} {d} p={p:.3f}")
        paired.append({"группа": name, "запросов": len(groups[name]),
                       "diff": round(d.mean, 4), "lo": round(d.lo, 4),
                       "hi": round(d.hi, 4), "p": round(p, 4)})

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump({"что меряется": "доля различных слов вопроса, встречающихся "
                                   "в объединении его эталонных фрагментов",
                   "подвыборки, написанные моделью": list(MODEL_WRITTEN),
                   "выдача для деления живых": os.path.relpath(args.run, ROOT),
                   "уровни": levels, "разрыв": diffs,
                   "первый найденный против эталона": paired},
                  f, ensure_ascii=False, indent=2)
    say()
    say(f"файлы: {os.path.relpath(REPORT, ROOT)}, {os.path.relpath(RESULT, ROOT)}")


if __name__ == "__main__":
    main()
