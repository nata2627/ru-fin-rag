#!/usr/bin/env python3
"""Task 12.17: agreement of the human sheet of 200 pairs with the stage 12 labeler.

The sheet is blind (live12_human_sheet.py), strata and the labeler's grade live
in the key. Kappa is printed for the whole sheet and per stratum, with the
direction of disagreement: a labeler that grades higher than the human lets
false positives into the test labels, one that grades lower hides answers.
A stratum where one side gives a single grade has no defined kappa, so the
raw agreement is printed next to it. The stop rule is in PLAN 12.17: kappa
below 0.40 on any large stratum stops the gate.

    python3 scripts/human_live12_kappa.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import razmetka_yadro as Y  # noqa: E402

from rufin.annotation import cohen_kappa, kappa_interval, kappa_verdict  # noqa: E402

ROOT = Y.ROOT
SHEET = os.path.join(Y.QDIR, "human_live12.tsv")
KEY = os.path.join(ROOT, "data", "interim", "live12", "human_live12_key.tsv")
RESULT = os.path.join(ROOT, "data", "results", "human-live12.json")
THRESHOLD = 0.40


def block(human: list[int], labeler: list[int]) -> dict:
    n = len(human)
    out = {"пар": n, "совпало": {"числитель": sum(h == m for h, m in zip(human, labeler)), "знаменатель": n},
           "человек": dict(sorted(collections.Counter(human).items())),
           "разметчик": dict(sorted(collections.Counter(labeler).items())),
           "разметчик выше": sum(m > h for h, m in zip(human, labeler)),
           "разметчик ниже": sum(m < h for h, m in zip(human, labeler))}
    hb, mb = [int(h > 0) for h in human], [int(m > 0) for m in labeler]
    out["позитив против нуля"] = {
        "разметчик да, человек нет": sum(m and not h for h, m in zip(hb, mb)),
        "разметчик нет, человек да": sum(h and not m for h, m in zip(hb, mb))}
    if len(set(human)) > 1 and len(set(labeler)) > 1:
        for name, weighted in (("каппа", False), ("каппа взвешенная", True)):
            k = cohen_kappa(human, labeler, weighted=weighted)
            lo, hi = kappa_interval(human, labeler, weighted=weighted)
            out[name] = {"значение": round(k, 3), "lo": round(lo, 3), "hi": round(hi, 3),
                         "словами": kappa_verdict(k)}
        k = cohen_kappa(hb, mb)
        out["каппа позитив против нуля"] = round(k, 3)
    else:
        out["каппа"] = "не определена: одна из сторон ставит одну оценку"
    return out


def main() -> None:
    sheet = {(r["query_id"], r["chunk_id"]): r for r in Y.read_tsv(SHEET)}
    key = Y.read_tsv(KEY)
    if set(sheet) != {(r["query_id"], r["chunk_id"]) for r in key}:
        raise SystemExit("sheet and key hold different pairs")
    empty = [k for k, r in sheet.items() if not r["otvet1_celikom"].strip()]
    if empty:
        raise SystemExit(f"{len(empty)} pairs have no answer")
    rows = [(r["stratum"], Y.grade(sheet[(r["query_id"], r["chunk_id"])]["otvet1_celikom"],
                                   sheet[(r["query_id"], r["chunk_id"])]["otvet2_vmeste"]),
             int(r["labeler_grade"])) for r in key]
    result = {"задача": "12.17", "лист": os.path.relpath(SHEET, ROOT),
              "человек": "разметка автора, вслепую, без ключа", "порог каппы на слое": THRESHOLD,
              "весь лист": block([h for _, h, _ in rows], [m for _, _, m in rows]),
              "слои": {s: block([h for t, h, _ in rows if t == s], [m for t, _, m in rows if t == s])
                       for s in dict.fromkeys(t for t, _, _ in rows)}}
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
