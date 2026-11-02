"""Headroom of a reranker over the hybrid top 50 on the live dev.

A reranker only reorders what the first stage brought. The oracle puts every positive punkt
reachable from the hybrid top 50 first, best grade first, so its NDCG@10 is the most any
reranker over that list can reach under the current labels. It is no bound of the true
ceiling either way: an unlabeled positive inside the top 50 would raise it, one outside
would raise IDCG and lower it. The test value is for reference only, no decision is taken
on it (stage 12 rules, S29).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import live12_build as L  # noqa: E402

from rufin import punkt as P  # noqa: E402


def oracle_ndcg(ranked: list[str], units: dict[str, P.Unit], depth: int = 50) -> float:
    """Ordering by chunk grade is not exact: a chunk grade can sit below the score of its punkt."""
    top = set(ranked[:depth])
    found = sorted((u.score for u in units.values() if u.score > 0 and u.chunks & top), reverse=True)[:10]
    ideal = sorted((u.score for u in units.values() if u.score > 0), reverse=True)[:10]

    def dcg(g):
        return sum((2 ** s - 1) / math.log2(i + 1) for i, s in enumerate(g, start=1))
    return dcg(found) / dcg(ideal)


def main() -> None:
    queries = [json.loads(l) for l in open(ROOT / "data" / "queries" / "live12.jsonl", encoding="utf-8")]
    units = P.read_units(str(ROOT / "data" / "queries" / "qrels_live12_punkt.tsv"))
    out = {}
    for split in ("dev", "test"):
        ids = {q["query_id"] for q in queries if q["split"] == split}
        runs, _ = L.load_runs(ids)
        rows = {"hybrid": [], "hybrid-rerank": [], "oracle top-50 hybrid": [], "oracle top-50 hybrid-rerank": []}
        for q in sorted(ids):
            for cfg in ("hybrid", "hybrid-rerank"):
                ranked = runs[cfg][q]
                rows[cfg].append(P.metrics(ranked, units[q])["NDCG@10"])
                rows[f"oracle top-50 {cfg}"].append(oracle_ndcg(ranked, units[q]))
        out[split] = {"вопросов": len(ids), **{k: round(float(np.mean(v)), 4) for k, v in rows.items()}}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(ROOT / "data" / "results" / "live12-ceiling.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
