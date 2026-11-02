"""Labeling pool for stage 12 live questions.

New questions: every pair from the first ten of the five core configurations, the same
priority 1 that the live test core was labeled on in tasks 11.5 and 11.6. Questions of
the second live set left without a single positive also get the tail of their pool down
to depth 50: for 11 of 25 of them the earlier judge did see an answer there, so the empty
result may be a cut-off, not a missing answer.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ["bm25", "hybrid", "hybrid-rerank", "dense-user-bge-m3", "dense-rufin"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--runs", type=Path, help="runs of the new questions; without it only the tail is built")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--empty-from", type=Path, help="pool of the second live set, tail taken for empty questions")
    ap.add_argument("--qrels", type=Path, help="labels of the second live set")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    # "new" in the query file means a fresh id; 20 questions dropped in the first collection
    # already had ids, so the set to pool is whatever has a run here.
    new = set()
    if a.runs:
        with open(a.runs / "base__bm25.jsonl", encoding="utf-8") as f:
            new = {json.loads(l)["query_id"] for l in f}
    pairs = {}
    for cfg in [c for c in CORE if a.runs and (a.runs / f"base__{c}.jsonl").exists()]:
        for line in open(a.runs / f"base__{cfg}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["query_id"] in new:
                for cid in r["ranked"][:a.depth]:
                    pairs[(r["query_id"], cid)] = "new"
    missing = new - {q for q, _ in pairs}
    if missing:
        raise SystemExit(f"no runs for {len(missing)} new questions")

    empty = set()
    if a.empty_from:
        with open(a.qrels, encoding="utf-8") as f:
            positive = {r["query-id"] for r in csv.DictReader(f, delimiter="\t") if int(r["score"]) > 0}
        with open(a.empty_from, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        empty = {r["query_id"] for r in rows} - positive
        for r in rows:
            if r["query_id"] in empty and r["prioritet"] == "2":
                pairs[(r["query_id"], r["chunk_id"])] = "tail"

    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["query_id", "chunk_id", "prioritet", "istochnik"])
        for (q, c), src in sorted(pairs.items()):
            w.writerow([q, c, "1", src])
    n_new = sum(v == "new" for v in pairs.values())
    print(f"pairs {len(pairs)}: new questions {n_new} over {len(new)}, tail of {len(empty)} empty questions {len(pairs) - n_new}")


if __name__ == "__main__":
    main()
