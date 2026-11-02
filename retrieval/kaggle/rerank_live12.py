"""Rerank the live12 hybrid runs with bge-reranker-v2-m3 on a Kaggle GPU.

Same procedure as kaggle/gpu_search.rerank_runs: depth 50, fp16, batch 32,
max_length 512, candidates sorted by descending score (stable). The model is
loaded with transformers directly from the attached Kaggle Models input, the
hub is never called.

Before the 677 new queries, 20 old live queries are reranked from their
original hybrid rows and compared with data/runs/base__hybrid-rerank.jsonl.
Fewer than 10 of 20 identical top-10 lists means a wrong model: stop.

Output is appended per query and the run resumes from what is already written.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import glob  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

TOP = 50
BATCH = 32
MAX_LEN = 512
OUT = "/kaggle/working/runs"
MIN_CONTROL = 10


def find(name):
    # Kaggle may unpack .gz when the dataset is created
    hits = (glob.glob(f"/kaggle/input/**/{name}", recursive=True)
            or glob.glob(f"/kaggle/input/**/{name.removesuffix('.gz')}", recursive=True))
    if not hits:
        raise SystemExit(f"not found among inputs: {name}")
    return hits[0]


def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def model_dir():
    for cfg in glob.glob("/kaggle/input/models/**/config.json", recursive=True):
        if "reranker" in cfg.lower():
            return os.path.dirname(cfg)
    raise SystemExit("reranker is not attached as a model input")


def main():
    path = model_dir()
    print(f"transformers {transformers.__version__}, torch {torch.__version__}", flush=True)
    print(f"model: {path}", flush=True)
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True, **{dtype_key: torch.float16}).to("cuda").eval()
    assert next(model.parameters()).dtype == torch.float16

    texts = {c["chunk_id"]: c["text"] for c in read_jsonl(find("chunks_candidates.jsonl.gz"))}
    print(f"chunks {len(texts)}", flush=True)

    @torch.inference_mode()
    def rerank(query, cand):
        scores = []
        for i in range(0, len(cand), BATCH):
            part = cand[i:i + BATCH]
            enc = tok([query] * len(part), [texts[c] for c in part], padding=True,
                      truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
            scores.extend(model(**enc).logits.view(-1).float().tolist())
        return [c for c, _ in sorted(zip(cand, scores), key=lambda x: -x[1])]

    ctl_q = {q["query_id"]: q["text"] for q in read_jsonl(find("control_queries.jsonl"))}
    ctl_h = {r["query_id"]: r["ranked"][:TOP] for r in read_jsonl(find("control_hybrid.jsonl"))}
    ctl_r = {r["query_id"]: r["ranked"] for r in read_jsonl(find("control_hybrid_rerank.jsonl"))}
    mine = {q: rerank(ctl_q[q], ctl_h[q])[:10] for q in ctl_q}
    same = sum(mine[q] == ctl_r[q][:10] for q in ctl_q)
    sets = sum(set(mine[q]) == set(ctl_r[q][:10]) for q in ctl_q)
    print(f"control: identical top-10 {same} of {len(ctl_q)}, same top-10 set {sets} "
          f"of {len(ctl_q)}", flush=True)
    if same < MIN_CONTROL:
        raise SystemExit("control failed: the attached model does not reproduce "
                         "base__hybrid-rerank.jsonl")

    queries = {q["query_id"]: q["text"] for q in read_jsonl(find("queries.jsonl"))}
    hybrid = {r["query_id"]: r["ranked"][:TOP] for r in read_jsonl(find("base__hybrid.jsonl"))}
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "base__hybrid-rerank.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["query_id"] for l in open(out_path, encoding="utf-8") if l.strip()}
    todo = [q for q in hybrid if q not in done]
    print(f"queries {len(hybrid)}, already done {len(done)}, pairs {len(todo) * TOP}",
          flush=True)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as out:
        for n, qid in enumerate(todo, 1):
            ranked = rerank(queries[qid], hybrid[qid])
            out.write(json.dumps({"query_id": qid, "ranked": ranked}, ensure_ascii=False) + "\n")
            out.flush()
            if n % 50 == 0 or n == len(todo):
                spent = time.time() - t0
                print(f"reranked {n}/{len(todo)}, {spent:.0f} s, "
                      f"left ~{spent / n * (len(todo) - n) / 60:.1f} min", flush=True)
    print(f"done: {out_path}", flush=True)


main()
