#!/usr/bin/env python3
"""Publish the stage 12 reranker (live10x2, step 7) to HuggingFace.

The weights come from the Kaggle output of ru-fin-train12-live10x2 and are not in
git (2.2 GB). The card on the hub is the English one; numbers in it are checked by
scripts/verify_numbers.py in the repository root, so the script refuses to upload
when that check fails.

Token: HF_TOKEN or `huggingface-cli login`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS = os.path.join(ROOT, "data", "kaggle_out", "ru-fin-train12-live10x2", "best")
CARD = os.path.join(ROOT, "docs", "MODEL_CARD_EN.md")
DEFAULT_REPO = "nata2627/ru-fin-reranker"
FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
BEST_STEP = 7


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(weights: str) -> dict:
    missing = [name for name in FILES if not os.path.exists(os.path.join(weights, name))]
    if missing:
        raise SystemExit(f"no {', '.join(missing)} in {weights}")
    # best/ is overwritten on every improvement, so the step it holds is read from
    # the run's own log rather than assumed.
    with open(os.path.join(weights, "..", "best.jsonl"), encoding="utf-8") as f:
        best = json.loads(f.readlines()[-1])["best_step"]
    if best != BEST_STEP:
        raise SystemExit(f"best/ holds step {best}, the card describes step {BEST_STEP}")
    with open(os.path.join(weights, "config.json"), encoding="utf-8") as f:
        arch = json.load(f)["architectures"]
    if arch != ["XLMRobertaForSequenceClassification"]:
        raise SystemExit(f"unexpected architecture {arch}")
    return {name: sha256(os.path.join(weights, name)) for name in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="checks only, no upload")
    args = ap.parse_args()

    verify = subprocess.run([sys.executable, os.path.join(ROOT, "..", "scripts", "verify_numbers.py")])
    if verify.returncode != 0:
        raise SystemExit("verify_numbers.py failed: the card would carry a stale number")
    hashes = check(args.weights)
    for name, digest in hashes.items():
        print(f"{name} {digest}")
    if args.dry_run:
        print("--dry-run: upload skipped")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.weights, repo_id=args.repo, repo_type="model",
                      allow_patterns=list(FILES))
    api.upload_file(path_or_fileobj=CARD, path_in_repo="README.md", repo_id=args.repo,
                    repo_type="model")
    print(f"published: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
