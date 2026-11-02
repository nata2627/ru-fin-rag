"""Drop synthetic questions whose source (first or second fragment) is a forbidden fragment.

synth_q.jsonl stays as it was: labels already made for it are keyed by its query ids.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufin.leak import forbidden_chunks  # noqa: E402

LIVE12 = ROOT / "data" / "interim" / "live12"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=LIVE12 / "synth_q.jsonl")
    ap.add_argument("--out", type=Path, default=LIVE12 / "synth_q_clean.jsonl")
    a = ap.parse_args()
    bad = forbidden_chunks()
    rows = [json.loads(line) for line in open(a.src, encoding="utf-8")]
    kept = [r for r in rows if not any(c in bad for c in r["source"])]
    second = sum(1 for r in rows if r["source"][0] not in bad and any(c in bad for c in r["source"][1:]))
    with open(a.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)} of {len(rows)}, dropped {len(rows) - len(kept)} "
          f"(only through the second fragment {second})")


if __name__ == "__main__":
    main()
