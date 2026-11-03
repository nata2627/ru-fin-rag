"""Training pairs of the stage 12 reranker: live train questions alone, and live plus synthetic.

Only labeled pairs go in: a positive is a labeler grade 1 or 2, a hard negative is a labeled 0
from the same pool (top of the hybrid run). Unlabeled pairs stay out, since a fragment nobody
read may well answer the question. Random negatives from other acts are added so the model
also sees easy cases. Every fragment, random ones included, passes the one leak rule
(rufin.leak.forbidden_chunks); a question left without a positive is dropped.

Synthetic questions enter only when the labeler accepted their source fragment: the input
gate (synth12_gate.py) was measured on that subset, 2 882 of 2 930.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import live12_build as L  # noqa: E402

from rufin.leak import forbidden_chunks  # noqa: E402

LIVE12 = ROOT / "data" / "interim" / "live12"
OUT = LIVE12 / "train"
N_RANDOM = 2
SEED = 12


def read_labels(works: list[Path]) -> dict[tuple[str, str], int]:
    labels: dict[tuple[str, str], int] = {}
    for w in works:
        for k, g in L.work_labels(str(w))[0].items():
            labels[k] = max(labels.get(k, 0), g)
    return labels


def build(questions: list[dict], labels: dict[tuple[str, str], int], act_of: dict[str, str],
          forbidden: frozenset[str], kind: str, rng: random.Random) -> list[dict]:
    by_q: dict[str, dict[str, int]] = {}
    for (q, c), g in labels.items():
        if c not in forbidden:
            by_q.setdefault(q, {})[c] = g
    pool = sorted(c for c in act_of if c not in forbidden)
    out = []
    for q in questions:
        got = by_q.get(q["query_id"], {})
        pos = {c: g for c, g in sorted(got.items()) if g > 0}
        if not pos:
            continue
        acts = {act_of[c] for c in pos} | {q["act"]}
        rand: list[str] = []
        while len(rand) < N_RANDOM:
            c = rng.choice(pool)
            if act_of[c] not in acts and c not in got and c not in rand:
                rand.append(c)
        out.append({"query_id": q["query_id"], "text": q["text"], "kind": kind, "pos": pos,
                    "neg": sorted(c for c, g in got.items() if g == 0), "neg_random": rand})
    return out


def stats(rows: list[dict]) -> dict:
    return {"вопросов": len(rows), "позитивов": sum(len(r["pos"]) for r in rows),
            "негативов из выдачи": sum(len(r["neg"]) for r in rows),
            "случайных негативов": sum(len(r["neg_random"]) for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    act_of = {}
    for line in open(ROOT / "data" / "chunks" / "base.jsonl", encoding="utf-8"):
        c = json.loads(line)
        act_of[c["chunk_id"]] = c["act_id"]
    forbidden = forbidden_chunks()
    rng = random.Random(SEED)

    live = [json.loads(l) for l in open(LIVE12 / "train_live_clean.jsonl", encoding="utf-8")]
    for q in live:
        q["act"] = q["act_id"]
    live_rows = build(live, read_labels([LIVE12 / "razmetka_train_h", LIVE12 / "razmetka_train_rr"]),
                      act_of, forbidden, "live", rng)

    synth_labels = read_labels([LIVE12 / "razmetka_synth_h", LIVE12 / "razmetka_synth_rr"])
    synth = [json.loads(l) for l in open(LIVE12 / "synth_q_clean.jsonl", encoding="utf-8")]
    accepted = [q for q in synth if all(synth_labels.get((q["query_id"], c), 0) > 0 for c in q["source"])]
    for q in accepted:
        q["act"] = act_of[q["source"][0]]
    synth_rows = build(accepted, synth_labels, act_of, forbidden, "synth", rng)

    a.out.mkdir(parents=True, exist_ok=True)
    files = {"live.jsonl": live_rows, "live_synth.jsonl": live_rows + synth_rows}
    for name, rows in files.items():
        with open(a.out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"живые": stats(live_rows), "синтетика": stats(synth_rows),
              "живые, вопросов на входе": len(live), "синтетика, вопросов на входе": len(synth),
              "синтетика, исходный принят": len(accepted),
              **{name: stats(rows) for name, rows in files.items()}}
    with open(ROOT / "data" / "results" / "train-pairs12.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
