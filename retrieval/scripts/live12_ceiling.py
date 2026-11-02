"""Headroom of a reranker over the hybrid top 50 on the live dev.

A reranker only reorders what the first stage brought. The oracle puts every labeled
positive of the hybrid top 50 first, best grade first, so its NDCG@10 is the most any
reranker over that list can reach. Fragments past the labeled pool count as misses for
the oracle too, so the ceiling is a lower bound of the true one.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import live12_build as L  # noqa: E402

from rufin import punkt as P  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402


def main() -> None:
    queries = [json.loads(l) for l in open(ROOT / "data" / "queries" / "live12.jsonl", encoding="utf-8")]
    units = P.read_units(str(ROOT / "data" / "queries" / "qrels_live12_punkt.tsv"))
    qrels = read_qrels(str(ROOT / "data" / "queries" / "qrels_live12.tsv"))
    out = {}
    for split in ("dev", "test"):
        ids = {q["query_id"] for q in queries if q["split"] == split}
        runs, _ = L.load_runs(ids)
        rows = {"hybrid": [], "hybrid-rerank": [], "oracle top-50 hybrid": [], "oracle top-50 hybrid-rerank": []}
        for q in sorted(ids):
            for cfg in ("hybrid", "hybrid-rerank"):
                ranked = runs[cfg][q]
                rows[cfg].append(P.metrics(ranked, units[q])["NDCG@10"])
                grades = qrels.get(q, {})
                oracle = sorted(ranked[:50], key=lambda c: -grades.get(c, 0))
                rows[f"oracle top-50 {cfg}"].append(P.metrics(oracle, units[q])["NDCG@10"])
        out[split] = {"вопросов": len(ids), **{k: round(float(np.mean(v)), 4) for k, v in rows.items()}}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(ROOT / "data" / "results" / "live12-ceiling.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
