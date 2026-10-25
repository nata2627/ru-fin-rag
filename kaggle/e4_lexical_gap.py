#!/usr/bin/env python3
"""Лексический разрыв обучающих выборок с живыми вопросами: прежней и E4.

Предсказание S18, записанное до прогона: примеры живых вопросов в подсказке
генератора сузят разрыв с 0,172 до 0,12. Проверить его `scripts/lexical_gap.py`
нельзя: он считает разрыв по набору запросов, а набор E4 не менял вовсе, и
число там по построению остаётся прежним. Разрыв нужно считать по обучающим
вопросам, которых в наборе нет.

Доля слов и токенизация берутся из `scripts/lexical_gap.py` импортом, а не
переписываются здесь: две реализации одной статистики рано или поздно
разойдутся, и тогда непонятно, какая из них в отчёте.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lexical_gap import share  # noqa: E402

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402


def texts_of(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {c["chunk_id"]: c["text"] for c in (json.loads(l) for l in f
                                                   if l.strip())}


def shares_train(path: str, texts: dict[str, str], stem: bool) -> list[float]:
    """Доли по обучающей выборке: эталон у каждого вопроса один, свой фрагмент."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            gold = q.get("gold_chunk_id")
            if gold not in texts:
                continue
            s = share(q["text"], [texts[gold]], stem)
            if s == s:
                out.append(s)
    return out


def shares_live(texts: dict[str, str], stem: bool) -> list[float]:
    """Доли по живым вопросам набора: эталонов у вопроса может быть несколько."""
    qrels = read_qrels(os.path.join(ROOT, "data", "queries", "qrels.tsv"))
    out = []
    with open(os.path.join(ROOT, "data", "queries", "queries.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            if q.get("origin") == "синтетический":
                continue
            gold = [c for c, g in (qrels.get(q["query_id"]) or {}).items()
                    if g > 0 and c in texts]
            if not gold:
                continue
            s = share(q["text"], [texts[c] for c in gold], stem)
            if s == s:
                out.append(s)
    return out


def ci(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    # Интервал считает тот же бутстрэп, которым посчитаны все остальные
    # интервалы проекта (`rufin.metrics.bootstrap_ci`), и возвращает он
    # `Interval`, а не пару.
    iv = M.bootstrap_ci(a)
    return {"запросов": len(xs), "среднее": round(float(a.mean()), 4),
            "lo": round(float(iv.lo), 4), "hi": round(float(iv.hi), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks/base.jsonl")
    ap.add_argument("--trains", nargs="+", default=[
        "data/queries/synthetic_train.jsonl",
        "data/queries/synthetic_train_e4_filtered.jsonl",
    ])
    ap.add_argument("--result", default="data/results/e4_lexical_gap.json")
    ap.add_argument("--report", default="docs/raw/e4_lexical_gap.txt")
    args = ap.parse_args()

    texts = texts_of(os.path.join(ROOT, args.chunks))
    живые = {stem: shares_live(texts, stem) for stem in (True, False)}
    out = {"живые": {("со стеммером" if s else "без стеммера"): ci(v)
                     for s, v in живые.items()},
           "обучающие": {}, "разрыв с живыми": {}}
    for p in args.trains:
        полный = os.path.join(ROOT, p)
        if not os.path.exists(полный):
            continue
        out["обучающие"][p] = {}
        out["разрыв с живыми"][p] = {}
        for stem in (True, False):
            v = shares_train(полный, texts, stem)
            имя = "со стеммером" if stem else "без стеммера"
            out["обучающие"][p][имя] = ci(v)
            out["разрыв с живыми"][p][имя] = round(
                float(np.mean(v)) - float(np.mean(живые[stem])), 4)

    with open(os.path.join(ROOT, args.result), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    L = ["доля слов вопроса, встречающихся в его эталонном фрагменте", "",
         "живые вопросы набора (эталонов у вопроса может быть несколько):"]
    for имя, d in out["живые"].items():
        L.append(f"   {имя:<16} {d['среднее']:.4f}, запросов {d['запросов']}")
    L += ["", "обучающие выборки (эталон у вопроса один, свой фрагмент):"]
    for p, d in out["обучающие"].items():
        L.append(f"   {p}")
        for имя, v in d.items():
            L.append(f"      {имя:<16} {v['среднее']:.4f}, вопросов "
                     f"{v['запросов']}, разрыв с живыми "
                     f"{out['разрыв с живыми'][p][имя]:+.4f}")
    L += ["", "предсказание S18 было: разрыв со стеммером 0,172 сузится до 0,12",
          "",
          "про знаменатель живых. Здесь их 145 по правилу «origin не",
          "синтетический и эталон размечен», а в подвыборке «живые» проекта",
          "их 142, и ворота считают по 142. Среднее здесь 0,5829 против 0,578",
          "у scripts/lexical_gap.py, то есть разница в три вопроса вывода",
          "не двигает. Разрывы обучающих выборок между собой сравнимы: они",
          "посчитаны к одному и тому же знаменателю."]
    with open(os.path.join(ROOT, args.report), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
