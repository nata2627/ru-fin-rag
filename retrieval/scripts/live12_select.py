"""Pick the new live questions for stage 12 from the candidate table.

Labeling cost grows with every question, and an act that brings hundreds of questions
would become the next 590-П. So new questions are capped per act and the form route is
capped overall (decision 26). Within an act the more trusted routes fill first, ties are
broken by a seeded shuffle. Questions already in the live set keep their ids.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufin.explanations import ORIGIN, assign_ids  # noqa: E402

TRUST = {"раздел": 0, "текст": 1, "тема": 2, "форма": 3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--per-act", type=int, default=40)
    ap.add_argument("--forms", type=int, default=150)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    live_ids = {json.loads(l)["query_id"] for l in open(ROOT / "data" / "queries" / "queries.jsonl", encoding="utf-8")
                if json.loads(l).get("podvyborka") == "живые"}
    rows = [json.loads(l) for l in open(a.candidates, encoding="utf-8")]
    old = [r for r in rows if r["query_id"] in live_ids]
    old_groups = {r["group"] for r in old}
    pool = [r for r in rows if r["route"] and r["query_id"] not in live_ids and r["group"] not in old_groups]

    rng = random.Random(a.seed)
    rng.shuffle(pool)
    pool.sort(key=lambda r: TRUST[r["route"]])
    per_act, forms, groups, picked = collections.Counter(), 0, set(), []
    for r in pool:
        if per_act[r["act_id"]] >= a.per_act or r["group"] in groups:
            continue
        if r["route"] == "форма":
            if forms >= a.forms:
                continue
            forms += 1
        per_act[r["act_id"]] += 1
        groups.add(r["group"])
        picked.append(r)

    known = {json.loads(l)["text"]: json.loads(l)["query_id"]
             for l in open(ROOT / "data" / "queries" / "explan_live.jsonl", encoding="utf-8")}
    ids = assign_ids([r["text"] for r in picked], known)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in old + picked:
            qid = r["query_id"] or ids[r["text"]]
            f.write(json.dumps({"query_id": qid, "origin": ORIGIN, "text": r["text"], "route": r["route"],
                                "act_id": r["act_id"], "group": r["group"], "section": r["section"],
                                "topic": r["topic"], "clauses": r["clauses"], "url": r["url"],
                                "new": r["query_id"] is None}, ensure_ascii=False) + "\n")

    print(f"old live {len(old)}, new {len(picked)} (pool {len(pool)}), total {len(old) + len(picked)}")
    print("new by route:", dict(collections.Counter(r["route"] for r in picked)))
    allacts = collections.Counter(r["act_id"] for r in old + picked)
    print("acts", len(allacts), "top:", allacts.most_common(8))
    print("over 600 chars:", sum(len(r["text"]) > 600 for r in old + picked), "of", len(old) + len(picked))


if __name__ == "__main__":
    main()
