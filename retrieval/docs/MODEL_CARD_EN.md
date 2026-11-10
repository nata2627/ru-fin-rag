---
license: apache-2.0
language:
- ru
base_model: BAAI/bge-reranker-v2-m3
pipeline_tag: text-classification
library_name: transformers
tags:
- reranker
- cross-encoder
- legal
- finance
- banking-regulation
- russian
datasets:
- nata2627/ru-fin-retrieval
---

# ru-fin-reranker

`BAAI/bge-reranker-v2-m3` fine-tuned on real questions about the regulations of the Bank of
Russia. **It did not pass the acceptance rule written before training.** On 674 held-out live
questions it beats the base reranker by +0.0127 NDCG@10 [+0.0054; +0.0200], p 0.0012: the gain
is real, but smaller than the +0.025 threshold, which is one step above the resolution of the
test. The weights are published with that number, and the service of the project keeps the base
model.

* Repository and the full report: <https://github.com/nata2627/ru-fin-rag/tree/main/retrieval>,
  `docs/19-pereranzhirovshchik.md` (in Russian).
* The live test, its labels and split live in the same repository: `data/queries/live12.jsonl`,
  `data/queries/qrels_live12_punkt.tsv`, `data/queries/podvyborki_live12.json`.
* Language of the data: Russian. A Russian copy of this card is `docs/MODEL_CARD.md`.

## Use

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("nata2627/ru-fin-reranker")
model = AutoModelForSequenceClassification.from_pretrained("nata2627/ru-fin-reranker").eval()
pairs = [["Какой срок хранения кассовых документов?", "Кассовые документы хранятся ..."]]
with torch.no_grad():
    enc = tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
    scores = model(**enc).logits.view(-1)
```

Scores are logits, higher means more relevant. In the evaluation the model reorders the top 50
of a hybrid retriever (BM25 and a bge-m3 dense encoder fused by RRF).

## Result

NDCG@10 counted by act clauses: a hit is credited to the clause of the act, so that a chunking
with other boundaries is not punished for them. Test of 674 live questions, the base reranker in
the same pipeline is the reference.

| test | questions | base reranker | this model | difference |
|---|---|---|---|---|
| all | 674 | 0.6664 | 0.6791 | +0.0127 [+0.0054; +0.0200] |
| without leaked questions | 672 | 0.6662 | 0.6784 | +0.0122 [+0.0048; +0.0197] |

Two test questions got a new positive on a fragment the model was trained on; the rule set
before counting was to compute the gate twice and take the stricter verdict. Both say the same.

For scale, on the same test BM25 gives 0.4992 and the hybrid without reranking 0.5305.

## Training

* Data: 123 live questions from the "Explanations" section of cbr.ru, not in dev or test, with
  positives, labeled zeros from the hybrid top 50 and random fragments of other acts. Fragments
  that are positives of any test, dev or earlier benchmark question are excluded from training
  (4 458 fragments).
* Objective: softmax cross-entropy over a group of one positive, up to five labeled negatives
  and two random ones; 512 tokens; full fine-tuning, fp16 autocast, one T4.
* Checkpoint: one epoch at LR 2e-5 under a ten-epoch schedule with 10% warmup, chosen by dev
  (224 live questions): 0.7088 against 0.6964 for the base model. Dev chose it, so the dev
  number is optimistic; the test number above is the honest one.
* What did not work: adding 2 882 synthetic questions written by a language model made every
  checkpoint worse than the base model on dev; more epochs on live questions overfit after two to
  four epochs (dev 0.6607 at epoch ten).

## Limitations

* The relevance labels of the test are made by a language model with one instruction. The
  author labeled 200 pairs blind: agreement on 182 of 200, Cohen's kappa 0.782
  [0.678; 0.867]. The model labeler adds positives the human does not see (13 cases) more often
  than it misses them (3), so absolute NDCG values are inflated; both pipelines share that bias.
* The corpus holds acts as published in the Vestnik Banka Rossii, not their current wording.
* 123 training questions is little; the learning curve (30, 60, 123 questions) shows the gain
  appearing only at 123, and three points do not tell its shape from noise.
* Domain: regulations of the Bank of Russia in Russian. Nothing was measured outside it.
