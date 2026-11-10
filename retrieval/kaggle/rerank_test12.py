"""Reranking of the stage 12 test with the candidate weights (tasks 12.16 and 12.18).

The candidate is the best dev step of a training kernel, attached here through
that kernel's output. Dev is reranked first as a control: it must give the dev
value the training kernel recorded, otherwise the weights or the setup differ
and the run stops before the test is touched. The test gets only questions and
hybrid lists, its labels stay on the laptop.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import glob  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

EXPECTED_DEV = 0.7088
TOP = 50
MAX_LEN = 512
OUT = "/kaggle/working"


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise SystemExit(f"not found among inputs: {name}")
    return hits[0]


def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    sys.path.insert(0, os.path.dirname(find("punkt.py")))
    import punkt as P

    path = os.path.dirname(find("best/config.json"))
    print(f"weights {path}, card {torch.cuda.get_device_name(0)}", flush=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to("cuda").eval()
    texts = {c["chunk_id"]: c["text"] for c in read_jsonl(find("chunks.jsonl.gz"))}

    @torch.inference_mode()
    def rerank(queries, lists, name):
        t0 = time.time()
        runs = {}
        for qid, q in sorted(queries.items()):
            cand, s = lists[qid], []
            for i in range(0, len(cand), 32):
                part = cand[i:i + 32]
                enc = tok([q] * len(part), [texts[c] for c in part], padding=True,
                          truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    s.extend(model(**enc).logits.view(-1).float().tolist())
            runs[qid] = [c for c, _ in sorted(zip(cand, s), key=lambda x: -x[1])]
        with open(os.path.join(OUT, f"{name}_runs.jsonl"), "w", encoding="utf-8") as f:
            for qid, ranked in runs.items():
                f.write(json.dumps({"query_id": qid, "ranked": ranked}) + "\n")
        print(f"{name}: {len(runs)} questions, {time.time() - t0:.0f} s", flush=True)
        return runs

    def load(split):
        q = {r["query_id"]: r["text"] for r in read_jsonl(find(f"{split}_queries.jsonl"))}
        h = {r["query_id"]: r["ranked"][:TOP] for r in read_jsonl(find(f"{split}_hybrid.jsonl"))}
        return q, h

    units = P.read_units(find("dev_punkt.tsv"))
    dev = rerank(*load("dev"), "dev")
    value = sum(P.metrics(r, units[q])["NDCG@10"] for q, r in dev.items()) / len(dev)
    print(f"dev control {value:.4f}, expected {EXPECTED_DEV}", flush=True)
    if abs(value - EXPECTED_DEV) > 0.002:
        raise SystemExit("dev does not reproduce the training kernel, test not reranked")
    rerank(*load("test"), "test")


if __name__ == "__main__":
    main()
