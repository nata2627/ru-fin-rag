"""Long questions and the 512-token window of bge-reranker-v2-m3, live12 dev.

Two modes over the same hybrid top-50:

* full: the question as is, the pair is cut to 512 tokens by the tokenizer
  (longest_first), the same call as kaggle/rerank_live12.py;
* q256: the question is cut to its first 256 tokens before pairing, so a long
  letter leaves at least 256 tokens to the fragment.

A question of 256 tokens or less is the same in both modes and is scored once.
The full mode doubles as the control: its top-10 is compared with the stored
hybrid-rerank rows, fewer than half identical means a wrong model.

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
Q_LEN = 256
OUT = "/kaggle/working/runs"


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


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["query_id"] for l in open(path, encoding="utf-8") if l.strip()}


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
    queries = {q["query_id"]: q["text"] for q in read_jsonl(find("queries.jsonl"))}
    hybrid = {r["query_id"]: r["ranked"][:TOP] for r in read_jsonl(find("base__hybrid.jsonl"))}
    stored = {r["query_id"]: r["ranked"][:10] for r in read_jsonl(find("base__hybrid-rerank.jsonl"))}

    @torch.inference_mode()
    def rerank(query, cand):
        scores = []
        for i in range(0, len(cand), BATCH):
            part = cand[i:i + BATCH]
            enc = tok([query] * len(part), [texts[c] for c in part], padding=True,
                      truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
            scores.extend(model(**enc).logits.view(-1).float().tolist())
        return [c for c, _ in sorted(zip(cand, scores), key=lambda x: -x[1])]

    def cut(query):
        ids = tok(query, add_special_tokens=False)["input_ids"]
        return (tok.decode(ids[:Q_LEN]), len(ids)) if len(ids) > Q_LEN else (query, len(ids))

    os.makedirs(OUT, exist_ok=True)
    full_path = os.path.join(OUT, "full.jsonl")
    cut_path = os.path.join(OUT, "q256.jsonl")
    done = done_ids(full_path) & done_ids(cut_path)
    todo = [q for q in hybrid if q not in done]
    long_q = sum(cut(queries[q])[1] > Q_LEN for q in hybrid)
    print(f"queries {len(hybrid)}, longer than {Q_LEN} tokens {long_q}, already done {len(done)}",
          flush=True)
    t0 = time.time()
    same = 0
    with open(full_path, "a", encoding="utf-8") as f_full, open(cut_path, "a", encoding="utf-8") as f_cut:
        for n, qid in enumerate(todo, 1):
            full = rerank(queries[qid], hybrid[qid])
            short, n_tok = cut(queries[qid])
            q256 = rerank(short, hybrid[qid]) if n_tok > Q_LEN else full
            same += full[:10] == stored[qid]
            for f, ranked in ((f_full, full), (f_cut, q256)):
                f.write(json.dumps({"query_id": qid, "ranked": ranked, "tokens": n_tok},
                                   ensure_ascii=False) + "\n")
                f.flush()
            if n == 20 and same < 10:
                raise SystemExit(f"control failed: full top-10 identical to stored in {same} of 20")
            if n % 25 == 0 or n == len(todo):
                spent = time.time() - t0
                print(f"reranked {n}/{len(todo)}, control identical {same} of {n}, {spent:.0f} s, "
                      f"left ~{spent / n * (len(todo) - n) / 60:.1f} min", flush=True)
    print(f"done: {OUT}", flush=True)


main()
