"""Training questions that nearly repeat a question of the live set are dropped.

A training question at word Jaccard 0.8 or more with any of the 1 002 differs by a date or a
figure, so training on it is training on the test question. Same measure as the split
(live12_build.DUP). The synthetic set is checked the same way.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from live12_build import DUP  # noqa: E402
from live12_candidates import words  # noqa: E402

LIVE12 = ROOT / "data" / "interim" / "live12"


def near(texts: list[str], live: list[frozenset[str]], threshold: float = DUP) -> list[bool]:
    out = []
    for t in texts:
        a = words(t)
        out.append(any(a and b and len(a & b) / len(a | b) >= threshold for b in live))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=LIVE12 / "train_live.jsonl")
    ap.add_argument("--out", type=Path, default=LIVE12 / "train_live_clean.jsonl")
    ap.add_argument("--synth", type=Path, default=LIVE12 / "synth_q_clean.jsonl")
    a = ap.parse_args()
    live = [words(json.loads(line)["text"]) for line in open(ROOT / "data" / "queries" / "live12.jsonl", encoding="utf-8")]
    train = [json.loads(line) for line in open(a.train, encoding="utf-8")]
    hit = near([r["text"] for r in train], live)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, h in zip(train, hit):
            if not h:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train live: kept {hit.count(False)} of {len(train)}, near a live question {hit.count(True)}: "
          + " ".join(r["query_id"] for r, h in zip(train, hit) if h))
    synth = [json.loads(line) for line in open(a.synth, encoding="utf-8")]
    s_hit = near([r["text"] for r in synth], live)
    print(f"synthetic: near a live question {s_hit.count(True)} of {len(synth)}")


if __name__ == "__main__":
    main()
