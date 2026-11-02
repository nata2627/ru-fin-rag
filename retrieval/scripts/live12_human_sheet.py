"""Blind sheet of 200 pairs for the human check of the stage 12 labels (decision 26, item 6).

The earlier sheet of 150 pairs covered only the 142 old questions and only pairs that
already had a label. This one goes where nothing was checked: new questions (both answers
of the labeler), the reserve of 183, the old positives the labeler rejected, and questions
about reporting forms. Strata and the labeler's grade go to a separate key file, the sheet
itself shows neither.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import live12_build as L  # noqa: E402

from rufin.explanations import body_of  # noqa: E402

LIVE12 = ROOT / "data" / "interim" / "live12"
STRATA = [
    ("new_yes", 50), ("new_no", 50), ("reserve", 40), ("old_rejected", 30), ("forms", 30),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "queries" / "human_live12.tsv")
    ap.add_argument("--key", type=Path, default=LIVE12 / "human_live12_key.tsv")
    a = ap.parse_args()

    queries = {json.loads(l)["query_id"]: json.loads(l) for l in open(ROOT / "data" / "queries" / "live12.jsonl", encoding="utf-8")}
    done = {(r["query_id"], r["chunk_id"]) for r in csv.DictReader(
        open(ROOT / "data" / "queries" / "human_sample_new.tsv", encoding="utf-8"), delimiter="\t")}

    def work(name: str) -> dict:
        labels, _, _ = L.work_labels(str(LIVE12 / name))
        return labels

    new = {**work("razmetka_main"), **work("razmetka_rr")}
    reserve = L.sheet_labels(str(ROOT / "data" / "queries" / "razmetka_extra.tsv"))
    unseen = work("razmetka_unseen")
    pools = {
        "new_yes": [p for p, g in new.items() if g > 0 and queries[p[0]]["route"] != "форма"],
        "new_no": [p for p, g in new.items() if g == 0 and queries[p[0]]["route"] != "форма"],
        "reserve": list(reserve),
        "old_rejected": [p for p, g in unseen.items() if g == 0],
        "forms": [p for p in new if queries[p[0]]["route"] == "форма"],
    }
    grade = {**reserve, **unseen, **new}
    rng = random.Random(a.seed)
    picked = []
    for name, n in STRATA:
        pool = sorted(p for p in pools[name] if p not in done and p not in {q for q, _ in picked})
        picked += [(p, name) for p in rng.sample(pool, n)]
    rng.shuffle(picked)

    wanted = {c for (_, c), _ in picked}
    chunks = {}
    for line in open(ROOT / "data" / "chunks" / "base.jsonl", encoding="utf-8"):
        c = json.loads(line)
        if c["chunk_id"] in wanted:
            chunks[c["chunk_id"]] = c
    with open(a.out, "w", encoding="utf-8", newline="") as f, open(a.key, "w", encoding="utf-8", newline="") as k:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        kw = csv.writer(k, delimiter="\t", lineterminator="\n")
        w.writerow(["nomer", "otvet1_celikom", "otvet2_vmeste", "tip", "zapros", "akt", "razdel", "punkty",
                    "fragment", "query_id", "chunk_id"])
        kw.writerow(["nomer", "query_id", "chunk_id", "stratum", "labeler_grade"])
        for n, ((q, cid), name) in enumerate(picked, start=1):
            c = chunks[cid]
            flat = lambda s: s.replace("\t", " ").replace("\n", " ")  # noqa: E731
            w.writerow([n, "", "", "", flat(queries[q]["text"]), f"{c['number']} от {c['date']}",
                        flat((c.get("section") or "")[:60]), c.get("units", ""), flat(body_of(c)), q, cid])
            kw.writerow([n, q, cid, name, grade[(q, cid)]])
    print(f"sheet {len(picked)} pairs -> {a.out.relative_to(ROOT)}, key -> {a.key.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
