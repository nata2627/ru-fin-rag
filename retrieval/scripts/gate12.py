#!/usr/bin/env python3
"""Task 12.18: one application of the stage 12 gate to the live test.

The candidate is accepted when, on the 674 live test questions (clause NDCG@10),
the lower bound of the 95% interval of the paired difference with hybrid-rerank
is above zero and the point difference is at least +0.025. The resolution of
the test is 0.021-0.023, a smaller gain is not told apart from noise.

The gate is computed twice, on the whole test and without the questions where a
positive found by the 12.16 labeling is a fragment the candidate trained on.
The verdict is the stricter of the two; this rule was written before the count.

    python3 scripts/gate12.py --reason "<why this application>"
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from live12_runs import read_jsonl  # noqa: E402
from trunc12 import load_run  # noqa: E402

from rufin import metrics as M  # noqa: E402
from rufin import punkt as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CAND = os.path.join(ROOT, "data", "kaggle_out", "ru-fin-test12-k", "test_runs.jsonl")
LEAK = os.path.join(ROOT, "data", "interim", "live12", "leak_top10_questions.json")
LOG = os.path.join(ROOT, "docs", "raw", "gate_log.txt")
RESULT = os.path.join(ROOT, "data", "results", "gate12.json")
MIN_GAIN = 0.025
CANDIDATE = "rerank-live10x2-step7"


def verdict(cand: dict, base: dict, units: dict, ids: list[str]) -> dict:
    a = np.array([P.metrics(cand[q], units[q])["NDCG@10"] for q in ids])
    b = np.array([P.metrics(base[q], units[q])["NDCG@10"] for q in ids])
    d, p = M.paired_diff_ci(a, b)
    ok = d.lo > 0 and d.mean >= MIN_GAIN
    why = ("принят" if ok else "отклонен: " + ", ".join(
        x for x, bad in ((f"разница {d.mean:+.4f} меньше +{MIN_GAIN}", d.mean < MIN_GAIN),
                         (f"нижняя граница {d.lo:+.4f} не выше нуля", d.lo <= 0)) if bad))
    return {"вопросов": len(ids), "кандидат": round(float(a.mean()), 4), "hybrid-rerank": round(float(b.mean()), 4),
            "разница": {"среднее": round(d.mean, 4), "lo": round(d.lo, 4), "hi": round(d.hi, 4), "p": round(p, 4)},
            "принят": bool(ok), "итог": why}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", required=True)
    args = ap.parse_args()
    test = sorted(q["query_id"] for q in read_jsonl(os.path.join(QDIR, "live12.jsonl")) if q["split"] == "test")
    units = P.read_units(os.path.join(QDIR, "qrels_live12_punkt.tsv"))
    cand = {r["query_id"]: r["ranked"] for r in read_jsonl(CAND)}
    base = load_run("hybrid-rerank", set(test))
    leak = set(json.load(open(LEAK, encoding="utf-8")))
    full = verdict(cand, base, units, test)
    clean = verdict(cand, base, units, [q for q in test if q not in leak])
    accepted = full["принят"] and clean["принят"]
    result = {"задача": "12.18", "кандидат": CANDIDATE, "эталон": "hybrid-rerank",
              "метрика": "NDCG@10 по пунктам, живой тест после доразметки 12.16",
              "правило": f"нижняя граница 95% интервала выше 0 и разница не меньше +{MIN_GAIN}",
              "весь тест": full, "без вопросов с утечкой 12.16": clean,
              "решение": "принят" if accepted else "отклонен", "повод": args.reason}
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{stamp}\n  ворота этапа 12 по живому тесту, метрика NDCG@10 по пунктам\n"
                f"  кандидат {CANDIDATE}, эталон hybrid-rerank\n")
        for name, v in (("весь тест", full), ("без утечки 12.16", clean)):
            d = v["разница"]
            f.write(f"  {name}, запросов {v['вопросов']}: {v['hybrid-rerank']:.4f} -> {v['кандидат']:.4f}, "
                    f"разница {d['среднее']:+.4f} [{d['lo']:+.4f}; {d['hi']:+.4f}] p={d['p']:.3f}, {v['итог']}\n")
        f.write(f"  решение: {result['решение']}\n  повод: {args.reason}\n")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
