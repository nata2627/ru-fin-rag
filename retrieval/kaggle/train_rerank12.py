"""Full fine-tuning of bge-reranker-v2-m3 on the stage 12 pairs (task 12.15, decision 29).

One T4, the second card hidden: the trainer would wrap two cards into DataParallel
and run out of memory. Weights stay fp32 under fp16 autocast with a grad scaler,
gradient checkpointing keeps a group of 8 pairs at 512 tokens inside 16 GB.

A training group is one question, one positive (grade 1 or 2, drawn anew every
epoch), up to 5 labeled zeros from the same hybrid list and the 2 random fragments
of other acts; the loss is softmax cross-entropy with the positive at index 0.
Every pair already passed the leak filter when the file was built (12.13).

Dev (224 live questions) is reranked before training and at every evaluation:
hybrid top-50, clause NDCG@10, the same count as rufin.punkt.metrics. Step 0
must give the stored 0.696 within 0.01, otherwise the setup differs from the
baseline and the run stops. The best step by dev is saved, step 0 included:
if nothing beats the base model, there are no new weights.

Progress, evaluations and dev runs are written as they appear.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import glob  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROBE = False
LIVE_ONLY = False
LIVE_LIMIT = 0
LR = 1e-5
EPOCHS = 2
HARD_NEG = 5
GROUPS_PER_STEP = 2
ACCUM = 8
WARMUP = 0.1
MAX_LEN = 512
SEED = 12
EVALS_PER_EPOCH = 2
TOP = 50
BASE_DEV = 0.6964
OUT = "/kaggle/working"


def find(name):
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


def log(path, row):
    with open(os.path.join(OUT, path), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def groups_for_epoch(rows, epoch):
    rng = random.Random(SEED * 100 + epoch)
    out = []
    for r in rows:
        pos = rng.choice(sorted(r["pos"]))
        hard = rng.sample(r["neg"], min(HARD_NEG, len(r["neg"])))
        out.append((r["text"], [pos] + hard + r["neg_random"]))
    rng.shuffle(out)
    return out


def main():
    sys.path.insert(0, os.path.dirname(find("punkt.py")))
    import punkt as P

    path = model_dir()
    print(f"transformers {transformers.__version__}, torch {torch.__version__}, "
          f"card {torch.cuda.get_device_name(0)}, probe {PROBE}", flush=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    texts = {c["chunk_id"]: c["text"] for c in read_jsonl(find("chunks.jsonl.gz"))}
    train = read_jsonl(find("live_synth.jsonl"))
    dev_q = {q["query_id"]: q["text"] for q in read_jsonl(find("dev_queries.jsonl"))}
    dev_h = {r["query_id"]: r["ranked"][:TOP] for r in read_jsonl(find("dev_hybrid.jsonl"))}
    units = P.read_units(find("dev_punkt.tsv"))
    if LIVE_ONLY:
        train = [r for r in train if r["query_id"].startswith("trl")]
    if LIVE_LIMIT:
        # Nested subsets for the learning curve: one seeded order, the first N of it.
        train = sorted(train, key=lambda r: r["query_id"])
        random.Random(SEED).shuffle(train)
        train = train[:LIVE_LIMIT]
    if PROBE:
        train = train[:96]
        dev_q = dict(list(sorted(dev_q.items()))[:40])
    print(f"train questions {len(train)}, dev {len(dev_q)}, fragments {len(texts)}", flush=True)

    def scores(query, cand, batch=32):
        out = []
        for i in range(0, len(cand), batch):
            part = cand[i:i + batch]
            enc = tok([query] * len(part), [texts[c] for c in part], padding=True,
                      truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                out.extend(model(**enc).logits.view(-1).float().tolist())
        return out

    @torch.inference_mode()
    def evaluate(step):
        model.eval()
        t0 = time.time()
        runs, ndcg = {}, []
        for qid, q in sorted(dev_q.items()):
            s = scores(q, dev_h[qid])
            runs[qid] = [c for c, _ in sorted(zip(dev_h[qid], s), key=lambda x: -x[1])]
            ndcg.append(P.metrics(runs[qid], units[qid])["NDCG@10"])
        model.train()
        value = sum(ndcg) / len(ndcg)
        with open(os.path.join(OUT, "dev_runs", f"step{step}.jsonl"), "w", encoding="utf-8") as f:
            for qid, ranked in runs.items():
                f.write(json.dumps({"query_id": qid, "ranked": ranked}) + "\n")
        log("evals.jsonl", {"step": step, "dev_ndcg10": round(value, 4), "questions": len(ndcg),
                            "seconds": round(time.time() - t0)})
        print(f"eval step {step}: dev NDCG@10 {value:.4f} on {len(ndcg)}, "
              f"{time.time() - t0:.0f} s", flush=True)
        return value

    os.makedirs(os.path.join(OUT, "dev_runs"), exist_ok=True)
    best = evaluate(0)
    best_step = 0
    if not PROBE and abs(best - BASE_DEV) > 0.01:
        raise SystemExit(f"control failed: base model gives {best:.4f}, stored {BASE_DEV}")

    micro_per_epoch = -(-len(train) // GROUPS_PER_STEP)
    steps_per_epoch = micro_per_epoch // ACCUM
    total = steps_per_epoch * EPOCHS
    eval_at = {round(steps_per_epoch * (e + k / EVALS_PER_EPOCH))
               for e in range(EPOCHS) for k in range(1, EVALS_PER_EPOCH + 1)}
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, round(total * WARMUP), total)
    scaler = torch.amp.GradScaler("cuda")
    print(f"optimizer steps {total} ({steps_per_epoch} per epoch), evals at {sorted(eval_at)}",
          flush=True)

    model.train()
    step, pairs, loss_sum, loss_n, t0 = 0, 0, 0.0, 0, time.time()
    for epoch in range(EPOCHS):
        groups = groups_for_epoch(train, epoch)
        for m in range(steps_per_epoch * ACCUM):
            part = groups[m * GROUPS_PER_STEP:(m + 1) * GROUPS_PER_STEP]
            q = [g[0] for g in part for _ in g[1]]
            c = [texts[cid] for g in part for cid in g[1]]
            enc = tok(q, c, padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**enc).logits.view(-1).float()
            loss, start = 0.0, 0
            for g in part:
                n = len(g[1])
                loss = loss - torch.log_softmax(logits[start:start + n], 0)[0]
                start += n
            loss = loss / len(part)
            scaler.scale(loss / ACCUM).backward()
            loss_sum += loss.item()
            loss_n += 1
            pairs += len(c)
            if (m + 1) % ACCUM:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 10 == 0 or step == total:
                spent = time.time() - t0
                row = {"step": step, "of": total, "epoch": epoch, "loss": round(loss_sum / loss_n, 4),
                       "pairs_per_s": round(pairs / spent, 1), "train_seconds": round(spent),
                       "memory_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
                log("progress.jsonl", row)
                print(f"step {step}/{total} loss {row['loss']} {row['pairs_per_s']} pairs/s "
                      f"mem {row['memory_gb']} GB left ~{spent / step * (total - step) / 60:.0f} min "
                      f"of training", flush=True)
                loss_sum, loss_n = 0.0, 0
            if step in eval_at:
                value = evaluate(step)
                if value > best and not PROBE:
                    best, best_step = value, step
                    model.save_pretrained(os.path.join(OUT, "best"), safe_serialization=True)
                    tok.save_pretrained(os.path.join(OUT, "best"))
                log("best.jsonl", {"step": step, "best_step": best_step, "best_dev": round(best, 4)})
    print(f"done: best step {best_step}, dev NDCG@10 {best:.4f}", flush=True)


main()
