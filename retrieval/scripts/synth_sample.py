"""Source fragments for the stage 12 synthetic set.

A fragment is drawn uniformly from the corpus, so acts come in proportion to their size,
and an act stops contributing after --per-act fragments. Gold fragments of every test and
every piece of their punkts (rufin.leak) are never drawn: a training question written from
a test answer is the leak stage 11 removed.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufin.leak import forbidden_chunks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--per-act", type=int, default=20)
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--exclude", type=Path, help="jsonl of fragments already drawn")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    gold = forbidden_chunks()
    taken = set()
    if a.exclude and a.exclude.exists():
        taken = {json.loads(l)["chunk_id"] for l in open(a.exclude, encoding="utf-8")}
    chunks = [json.loads(l) for l in open(ROOT / "data" / "chunks" / "base.jsonl", encoding="utf-8")]
    pool = [c for c in chunks
            if c["chunk_id"] not in gold and c["chunk_id"] not in taken
            and c["position"] > 0 and len(c["text"]) >= a.min_chars]
    random.Random(a.seed).shuffle(pool)

    per_act = Counter()
    out = []
    for c in pool:
        if per_act[c["act_id"]] >= a.per_act:
            continue
        per_act[c["act_id"]] += 1
        out.append({k: c[k] for k in ("chunk_id", "act_id", "number", "title", "units", "text")})
        if len(out) == a.n:
            break

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"drawn {len(out)} of {len(pool)} eligible, acts {len(per_act)}, test gold excluded {len(gold)}")


if __name__ == "__main__":
    main()
