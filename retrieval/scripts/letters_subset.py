#!/usr/bin/env python3
"""Подвыборка «письма»: живые вопросы длиннее 600 знаков отдельно от остальных.

Правило в `rufin.benchmark.letter_subsets`, сборка набора
(`scripts/build_benchmark.py`) применяет его сама. Этот скрипт дописывает две
подвыборки в готовый `podvyborki.json`, не пересобирая набор: состав прочих
подвыборок и разметка не меняются. Подвыборка «живые» остается прежней, на
ней посчитаны опубликованные числа и ворота.

Отдельно считается состав писем среди всех 325 живых вопросов, включая
неразмеченные: он понадобится задаче 11.6.

    python3 scripts/letters_subset.py
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin import punkt as P  # noqa: E402
from rufin.benchmark import LETTER_CHARS, LETTERS, LIVE_NO_LETTERS, letter_subsets  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
SUBSETS = os.path.join(QDIR, "podvyborki.json")
RESULT = os.path.join(ROOT, "data", "results", "pisma.json")
RUNDIR = os.path.join(ROOT, "data", "runs")
DOC = os.path.join(ROOT, "docs", "18-nabor-do-i-posle.md")
BLOCK_START = "<!-- tablica: letters_subset.py -->"
BLOCK_END = "<!-- /tablica pisma -->"
LIVE = "живые"
# Связки, по которым письма сравниваются с остальными живыми вопросами.
RUNS = ("bm25", "dense-user-bge-m3", "hybrid", "hybrid-rerank")


def describe(ids: list[str], texts: dict[str, str]) -> dict:
    n = [len(texts[q]) for q in ids]
    return {"запросов": len(ids), "медиана длины": statistics.median(n) if n else None,
            "максимум": max(n) if n else None}


def compare(test: dict[str, list[str]]) -> list[dict]:
    """NDCG@10 по пунктам (задача 11.1) на письмах и на остальных живых, разница
    с интервалом. Запросы в группах разные, поэтому интервал непарный."""
    units = P.read_units(os.path.join(QDIR, "qrels_punkt.tsv"))
    out = []
    for cfg in RUNS:
        with open(os.path.join(RUNDIR, f"base__{cfg}.jsonl"), encoding="utf-8") as f:
            run = {r["query_id"]: r["ranked"] for r in map(json.loads, f)}
        a, b = (np.array([P.metrics(run[q], units[q])["NDCG@10"] for q in test[name]])
                for name in (LETTERS, LIVE_NO_LETTERS))
        d = M.unpaired_diff_ci(a, b)
        out.append({"связка": cfg, LETTERS: round(float(a.mean()), 4),
                    LIVE_NO_LETTERS: round(float(b.mean()), 4),
                    "разница": {"среднее": round(d.mean, 4), "lo": round(d.lo, 4), "hi": round(d.hi, 4)}})
        print(f"{cfg:20s} письма {a.mean():.3f}, без писем {b.mean():.3f}, "
              f"разница {d.mean:+.3f} [{d.lo:+.3f}; {d.hi:+.3f}]")
    return out


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def signed(x: float) -> str:
    return ("+" if x >= 0 else "−") + num(abs(x))


def doc_block(result: dict) -> str:
    t, a = result["живой тест"], result["все живые"]
    out = [f"В живом тесте писем {t[LETTERS]['запросов']} из {t[LIVE]['запросов']}, среди всех живых "
           f"вопросов, включая неразмеченные, {a[LETTERS]['запросов']} из {a[LIVE]['запросов']}. "
           f"Медиана длины письма в тесте {t[LETTERS]['медиана длины']:g} знаков, остальных "
           f"{t[LIVE_NO_LETTERS]['медиана длины']:g}.",
           "", f"| Связка | Письма, {t[LETTERS]['запросов']} | Живые без писем, "
               f"{t[LIVE_NO_LETTERS]['запросов']} | Разница [95%] |", "|---|---|---|---|"]
    for r in result["NDCG@10 по пунктам"]:
        d = r["разница"]
        out.append(f"| {r['связка']} | {num(r[LETTERS])} | {num(r[LIVE_NO_LETTERS])} | "
                   f"{signed(d['среднее'])} [{signed(d['lo'])}; {signed(d['hi'])}] |")
    return "\n".join(out)


def main() -> None:
    with open(os.path.join(QDIR, "queries.jsonl"), encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]
    texts = {q["query_id"]: q["text"] for q in queries}
    subsets = json.load(open(SUBSETS, encoding="utf-8"))
    test = letter_subsets(texts, subsets[LIVE])
    subsets.update(test)
    with open(SUBSETS, "w", encoding="utf-8") as f:
        json.dump(subsets, f, ensure_ascii=False, indent=1)

    all_live = [q["query_id"] for q in queries if q["podvyborka"] == LIVE]
    everything = letter_subsets(texts, all_live)
    result = {"порог знаков": LETTER_CHARS,
              "живой тест": {LIVE: describe(subsets[LIVE], texts),
                             LETTERS: describe(test[LETTERS], texts),
                             LIVE_NO_LETTERS: describe(test[LIVE_NO_LETTERS], texts)},
              "все живые": {LIVE: describe(all_live, texts),
                            LETTERS: describe(everything[LETTERS], texts),
                            LIVE_NO_LETTERS: describe(everything[LIVE_NO_LETTERS], texts)},
              "письма теста": test[LETTERS],
              "NDCG@10 по пунктам": compare(test)}
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    body = open(DOC, encoding="utf-8").read()
    pattern = re.compile(re.escape(BLOCK_START) + ".*?" + re.escape(BLOCK_END), re.S)
    if not pattern.search(body):
        raise SystemExit(f"в {DOC} нет меток {BLOCK_START} ... {BLOCK_END}")
    with open(DOC, "w", encoding="utf-8") as f:
        f.write(pattern.sub(lambda _: f"{BLOCK_START}\n{doc_block(result)}\n{BLOCK_END}", body))
    for part, d in result.items():
        if isinstance(d, dict):
            print(part, {k: v["запросов"] for k, v in d.items()})
    print(f"записано: {os.path.relpath(SUBSETS, ROOT)}, {os.path.relpath(RESULT, ROOT)}")


if __name__ == "__main__":
    main()
