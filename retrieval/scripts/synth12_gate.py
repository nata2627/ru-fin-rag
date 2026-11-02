"""Input gate of the stage 12 synthetic set: does it look like live questions before training.

Six earlier runs trained on questions that shared 0.750 of their words with the answer,
while live questions share 0.578, and the gate stood only at the output. Here the same
measures are taken on the live set and on the synthetic one, and the synthetic set passes
only if it sits inside the live range. Rules, fixed before the first measurement:
  word share with positives: synthetic mean at most live mean + 0.05;
  positives per question: between half and double of the live median;
  source fragment accepted by the labeler: at least 0.70 of questions (else the
  generator wrote questions the source does not answer).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import live12_build as L  # noqa: E402
from lexical_gap import share  # noqa: E402

from rufin.benchmark import read_qrels  # noqa: E402
from rufin.queryfilter import copy_score  # noqa: E402

LIVE_MARGIN = 0.05
MIN_SOURCE_OK = 0.70


def mean_ci(x: list[float], seed: int = 0) -> dict:
    a = np.array(x)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(a, len(a)).mean() for _ in range(2000)]
    return {"среднее": round(float(a.mean()), 4), "lo": round(float(np.percentile(boots, 2.5)), 4),
            "hi": round(float(np.percentile(boots, 97.5)), 4), "n": len(a)}


def profile(texts: dict[str, str], qrels: dict[str, dict[str, int]], chunks: dict[str, str]) -> dict:
    ids = [q for q in texts if qrels.get(q)]
    return {
        "вопросов": len(texts), "с позитивом": len(ids),
        "доля слов вопроса в позитивах": mean_ci([share(texts[q], [chunks[c] for c in qrels[q]], True) for q in ids]),
        "позитивов на вопрос, медиана": statistics.median(len(qrels[q]) for q in ids),
        "списано из позитива, медиана": round(statistics.median(
            max(copy_score(texts[q], chunks[c])[1] for c in qrels[q]) for q in ids), 3),
        "длина, медиана": statistics.median(len(texts[q]) for q in texts),
        "длиннее 600": sum(len(t) > 600 for t in texts.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--work", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    live = {json.loads(l)["query_id"]: json.loads(l)["text"]
            for l in open(ROOT / "data" / "queries" / "live12.jsonl", encoding="utf-8")}
    live_qrels = read_qrels(str(ROOT / "data" / "queries" / "qrels_live12.tsv"))
    synth = [json.loads(l) for l in open(a.queries, encoding="utf-8")]
    labels = {}
    for w in a.work:
        labels.update(L.work_labels(str(w))[0])
    synth_qrels: dict[str, dict[str, int]] = {}
    for (q, c), g in labels.items():
        if g > 0:
            synth_qrels.setdefault(q, {})[c] = g
    chunks = {}
    for line in open(ROOT / "data" / "chunks" / "base.jsonl", encoding="utf-8"):
        c = json.loads(line)
        chunks[c["chunk_id"]] = c["text"]

    source_ok = [q["query_id"] for q in synth if all(labels.get((q["query_id"], c), 0) > 0 for c in q["source"])]
    kept = {q["query_id"]: q["text"] for q in synth if q["query_id"] in set(source_ok)}
    p_live = profile(live, live_qrels, chunks)
    p_synth = profile(kept, synth_qrels, chunks)
    lo_pos, hi_pos = p_live["позитивов на вопрос, медиана"] / 2, p_live["позитивов на вопрос, медиана"] * 2
    checks = {
        "доля слов": p_synth["доля слов вопроса в позитивах"]["среднее"]
        <= p_live["доля слов вопроса в позитивах"]["среднее"] + LIVE_MARGIN,
        "позитивов на вопрос": lo_pos <= p_synth["позитивов на вопрос, медиана"] <= hi_pos,
        "исходный фрагмент принят": len(source_ok) / len(synth) >= MIN_SOURCE_OK,
    }
    out = {"живые": p_live, "синтетика, исходный принят": p_synth,
           "исходный фрагмент принят": f"{len(source_ok)} из {len(synth)}",
           "проверки": checks, "ворота пройдены": all(checks.values())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
