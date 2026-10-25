---
license: mit
language:
- ru
task_categories:
- text-retrieval
size_categories:
- 10K<n<100K
pretty_name: ru-fin-retrieval
tags:
- legal
- finance
- banking-regulation
- russian
- beir
- mteb
---

# ru-fin-retrieval

A retrieval benchmark over the regulations of the Bank of Russia, with a training set built on
the same corpus. No public benchmark existed for this domain, so this one was assembled from
scratch. The important thing about it is not its size but **what it is made of and what each
part measures**.

* License of the code and the annotations: MIT. The act texts themselves are official
  documents of the Bank of Russia, published in the Vestnik Banka Rossii bulletin.
* Repository: <https://github.com/nata2627/ru-fin-retrieval>
* Format: MTEB / BeIR, files `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`,
  `qrels/dev.tsv`.
* Language of the data: Russian. This card is in English, the Russian original lives in the
  repository as `docs/DATASET_CARD.md`.

## Load it

```python
import json
from huggingface_hub import hf_hub_download

queries = [
    json.loads(line)
    for line in open(
        hf_hub_download("nata2627/ru-fin-retrieval", "queries.jsonl", repo_type="dataset"),
        encoding="utf-8",
    )
]
```

`corpus.jsonl` holds `_id`, `title`, `text`; `queries.jsonl` holds `_id`, `text`; the qrels are
tab separated with the header `query-id`, `corpus-id`, `score`.

## What is here, honestly

The set is **not complete**, and the table separates what is annotated from what is not.

| part | target | now | state |
|---|---:|---:|---|
| live questions, annotated by the clauses named in them | none | **108**, all about act 590-P | done |
| hand written queries | 75 to 100 | **11** | a list of 100 acts is ready, being filled in |
| unseen acts (synthetic) | 150 to 200 | **396** | done |
| earlier synthetic set (kept for comparability) | none | **150** | done |
| **headline number**, annotated (live + hand written + unseen) | 400 to 450 | **541** | |
| train | 10 000 to 15 000 | **6 112** | capped at 12 questions per act, then cleaned of leakage |
| dev | 250 to 300 | **300** | done |
| annotator agreement (kappa) | > 0.40 | **0.408** (weighted 0.611), on the previous guideline | measured against a guideline that no longer exists |
| live questions, annotated through a pool | none | **37 of 228**, 191 dropped out | done |

**The headline number is not free of synthetic data, and that has to be said plainly. Of the
541 queries in it, 396 were written by a language model, that is 73 percent.** The unseen acts
are generated too, only over acts that were never trained on, so they measure generalisation
rather than natural phrasing.

**The only subset with no generation in it is the live questions, 142 of them.** Their metric
is printed as a separate row and differs sharply from the headline one: 0.459 against 0.683 for
the reranking configuration. That gap is the price of conversational phrasing, and it is the
main limitation of this dataset.

## Corpus

1554 regulatory acts of the Bank of Russia, 84.4 million characters, **62 594 chunks** in the
base chunking: Regulations (281), Directives (1137), Instructions (44), Methodological
Recommendations (88), Official Clarifications (4), years 2013 to 2026.

The acts are cut out of the Vestnik bulletin issues. The standalone act files on the website
are scans run through OCR in a Latin mode: Cyrillic is irreversibly replaced by Latin
lookalikes, and **И**, **Й** and **Н** collapse into a single `H`. Of a sample of 10 such files
1 was usable; of 10 bulletin issues all 10 were.

**A limitation to know before using the data:** the bulletin prints an act as adopted. The
currently effective revision is not in the corpus. Because of that, 16 percent of the live
questions with named clauses were dropped: the clause being asked about does not exist in our
text (clause 4.11 of Regulation 590-P was added by a later directive). All 20 dropped questions
are about 590-P.

## Split: by act, not by chunk

Chunks of one act overlap and share text, so a held out chunk of a training act has in effect
been seen already. The split is fixed in `data/queries/split.json` and checked by tests.

| group | acts | chunks | share | purpose |
|---|---:|---:|---:|---|
| `train` | 1292 | 44 358 | 70.9% | generating training questions |
| `dev` | 47 | 2 044 | 3.3% | accepting every stage of the recipe |
| `test_zhivye` | 29 | 8 671 | 13.9% | acts the live questions are about |
| `test_zhanr` | 42 | 3 320 | 5.3% | Instructions, a held out genre |
| `test_nevidannye` | 144 | 4 201 | 6.7% | generalisation against memorisation |

Three decisions, each with its own cost.

**The acts behind the live questions are held out entirely.** Act 590-P is the most discussed
document in the corpus and it leaves training completely. Otherwise any gain from training
would partly be recognition of the text it was trained on.

**dev is built from fully held out acts, not from held out chunks of training acts.** The
second option would have been cheaper, but a dev set made of chunks of a training act would
accept a recipe that merely memorised that act, and dev exists precisely to reject such
recipes.

**A whole class of documents is held out.** All 44 Instructions are outside training: 42 in
`test_zhanr` and two more in `test_zhivye`, which have live questions about them. Instructions
differ from Directives in language and in structure, and transfer to a new act and transfer to
a new genre are different things that have to be measured apart.

## Queries: four registers

| subset | count | origin | median length |
|---|---:|---|---:|
| live | 1621 collected, 325 in the set, 142 annotated | the Bank of Russia clarifications section | 370 characters |
| hand written | 11 | written by hand | 62 characters |
| unseen acts | 396 | generated over held out acts | 118 characters |
| synthetic | 150 | generated over corpus chunks | 136 characters |

A threefold and sixfold difference in length is exactly why these are never averaged into one
number.

### Live questions

The Bank of Russia publishes answers to questions from supervised organisations
(`cbr.ru/explan/`). **1621** were collected. **322** of them attach to acts in the corpus,
across **29 acts**; the rest are about federal laws, reporting forms and letters whose
documents are not in the corpus at all.

The attachment to an act is recovered from the page: first from the section heading (the Bank's
own attachment), then from the act number named inside the question. The rule was verified by
measurement: if the attachment is correct, the named clause has to be found in the text of the
named act. The hit rate is 84 percent for the section heading and 79 percent for the question
text.

**One act dominates.** 590-P (loan loss provisions) accounts for **52 percent** of the live
questions. That is a property of the source: the Bank has published the most clarifications
about it. Questions are not thrown away to balance the distribution, instead **every number
over the live subset is printed twice: over the whole subset and over the same subset without
590-P**. If the two diverge, that is a result in itself.

**Among the annotated ones the share is higher: 114 of 142 against 170 of 325 over the whole
subset.** The clause route produced 108 questions and all of them are about 590-P, because the
Bank names clauses in the topic heading only in that section. The judge pool added 34 more
questions, and the other acts came with them. Of the 142 annotated live questions, 28 are not
about 590-P. Another 183 questions of the 325 are still unannotated, and with them the share
should fall back towards the subset average.

Annotation routes:

* **by named clauses: 108 questions, 348 pairs.** The Bank names the clauses in the topic
  heading itself ("Maintaining the credit file (3.1.3, 3.1.5)"), and the chunk is located by the
  start of the clause. This is external evidence: it does not depend on what the retriever
  found. On average 3.2 gold chunks per question.
* **into the pool: 217 questions.** No clauses are named, so candidates are taken from the
  union of all configuration runs and labelled by a judge model. The earlier judge covered 34
  of the 217, the remaining 183 are unlabelled.
* **dropped: 20 questions.** A clause is named, but it does not exist in our revision of the
  act. Such a question must not be given candidates found by retrieval: they would look
  plausible and be wrong by construction.

### Hand written queries

The only part of the set where **act diversity is arranged on purpose**. The live questions do
not provide it (52 percent about one act) and the synthetic ones do not provide a third
register. The queries are therefore written against a list,
`data/queries/manual_task.tsv`: 100 rows, 100 different acts, all from held out split groups,
spread across document kinds (Regulations 33, Directives 30, Instructions 27, Methodological
Recommendations 9, Official Clarifications 1).

### Synthetic data

**The generator of the training set and the generator of the test set come from different model
families.** Otherwise the test would measure how well the student learned the style of its own
generator.

| what | model | state |
|---|---|---|
| the earlier 150 questions | `Qwen/Qwen2.5-14B-Instruct`, 4 bit | done |
| train | `Qwen/Qwen2.5-7B-Instruct-AWQ`, vLLM, fp16 | **6 112 questions** |
| dev and test | `NiGuLa/Vikhr-Nemo-12B-Instruct-R-21-09-24-awq-4bit` | **300 and 396** |
| judge | `Qwen/Qwen2.5-14B-Instruct-AWQ` | see the acceptance section below |

**The test generator was replaced along the way.** It started as
`TheBloke/Mistral-7B-Instruct-v0.2-AWQ`, picked on the single criterion of not being Qwen. That
criterion is necessary but it was not enough: on a probe Mistral produced eleven English
questions out of twenty, and among the Russian ones were malformed words and nonsense phrases.
Vikhr-Nemo is Mistral-Nemo fine tuned on Russian: still not Qwen, and the language is native.
After the replacement there are zero non Russian questions in all three files.

Throughput was measured before the volume was fixed: 1.95 questions per second on Qwen (T4,
fp16, because Turing has no bf16).

Five generation filters:

1. **copied from the chunk**: a verbatim match longer than five words reduces the task to
   substring search and hands BM25 an advantage;
2. **matches too many chunks**: by the mean inverse document frequency of the query words. A
   question like "what requirements does the Bank of Russia set?" has no right answer;
3. **the teacher does not rank the gold chunk in the better half**: the cross encoder
   `bge-reranker-v2-m3` reranks BM25 candidates, and if the source chunk does not land in the
   first half, the question is dropped. This catches what the first two filters do not: a
   question that some other chunk in the corpus answers better. The threshold is relative to
   the depth rather than a fixed number, because the gold chunk is always among the candidates,
   so a hard cut at 50 with a depth of 30 would pass everything. In the run the cut was 15 of
   30; 1.2 percent were dropped and the median rank of the gold chunk was first, so there is
   almost nothing to drop;
4. **not in Russian**: Latin letters must not outnumber Cyrillic ones, and letters that Russian
   does not have (і, ї, є, ґ) must not appear at all. This is needed because the test generator
   comes from another family and another family holds the language worse. The first two filters
   let an English question through: it is neither copied nor vague;
5. **refers to the chunk itself**: "what amount does this fragment state", "according to this
   provision". As a search query this is meaningless, because the user does not have the chunk
   in front of them. In the run this dropped 160 questions from the training set and 18 from
   dev and test.

**Rejection rate in the run:** 30 percent on the training set, 22 to 24 percent on dev and
test. The main reason everywhere is the same, copying from the chunk.

**Be careful with the earlier 150 synthetic questions:** for 106 of them the gold chunk lies in
a training act. They are fine for comparison with earlier runs, but after training that row
measures memorisation as well. The new synthetic test is generated over held out acts only.

## Annotator agreement

**The guideline changed on 2026-10-24, and the numbers below belong to the previous one.** The
annotation guideline was rewritten: the annotator no longer assigns a 0/1/2 grade directly but
answers two separate questions, and the grade is assembled from those answers by a rule
(`docs/ANNOTATION_GUIDE.md`). The reason came out of a review of twenty retrieval failures:
thirteen of the twenty are explained by the annotation rather than by the retriever. Judge
acceptance became three numbers instead of one, because the kappa of the same model on the same
150 pairs fell from 0.408 to 0.307 when the inference stack was rebuilt.

**What is comparable with what.** The 0/1/2 scale is the same and a grade assembled from two
answers reads as the old one, so metrics computed on the old and the new annotation are
comparable. The agreement numbers themselves are not: they were measured against different
guidelines.

### Human anchor, 2026-10-24: acceptance not met

A human annotated the same 150 pairs under the new guideline, and this is the first human
anchor in the project: the earlier kappa of 0.408 was measured between two models, which is
stated below and stays stated. The distribution of the human grades: 96 zeros, 44 ones,
10 twos.

The threshold was declared before the data: the kappa of the judge against the human must be at
least 0.40. Four judges were measured.

| judge | GPU | pairs | kappa vs human | weighted | exact matches | gross disagreements |
|---|---|---:|---|---:|---|---|
| `Qwen2.5-14B-Instruct-AWQ` | Tesla T4 | 150 of 150 | **0.236** [0.139; 0.334] | 0.362 | 72 of 150 | 18 of 150 |
| `Qwen2.5-32B-Instruct-AWQ` | Tesla T4 x2 | 150 of 150 | **-0.016** [-0.059; 0.030] | 0.090 | 40 of 150 | 2 of 150 |
| `Qwen2.5-7B-Instruct-AWQ` | Tesla T4 | 67 of 150 | **-0.021** [-0.068; 0.000] | -0.018 | 6 of 67 | 34 of 67 |
| `Qwen3-4B-AWQ` | RTX 3070 | 148 of 150 | **-0.013** [-0.075; 0.048] | 0.173 | 39 of 148 | 4 of 148 |

**No judge met the threshold, and that is a negative result rather than a reason to keep
shopping for judges.** The best one, `Qwen2.5-14B-Instruct-AWQ`, reaches 0.236 [0.139; 0.334]
against a threshold of 0.40, so even the upper end of the interval falls short. Its other two
acceptance numbers are met: agreement between two runs is 0.950 [0.900; 0.990] against a
threshold of 0.70, and the inference stack is recorded. The declared consequence holds: the
judge annotation is not applied, the depth 50 pool (40 134 pairs) is not labelled, and the
metrics stay on the existing annotation.

**The same sample labelled by an external model: 0.424 [0.299; 0.537].** That model is not the
project's judge and does not enter acceptance (paid interface, not open weights, cannot be run
over a pool under the autonomy requirement), but the number answers a separate question: is the
guideline infeasible, or are the judges weak? It matched the human exactly on 98 of 150 pairs
with a single gross disagreement. So the task is solvable as written, and it is the open models
the project can host for free that fail the threshold. Running it over the pool is still out of
the question: 40 134 pairs at a median prompt of 2 449 tokens is 98.3 million input tokens.

**Where the judge errs.** It is more generous than the human, and asymmetrically so: on 18 of
150 pairs it called a chunk a full answer where the human saw no answer at all, and on another
36 it called them partial answers on the same human zeros. There is not a single gross error in
the other direction.

**What this says about the 0.408.** Agreement between two models turned out to be three times
higher than agreement between a model and a human on a sample of the same size. The earlier
number therefore measured how alike two automatic annotators are, not how close either is to an
expert, and it cannot be read as annotation quality.

| | |
|---|---:|
| pairs in the check sample | 150 (stratified: 52 zeros, 50 ones, 48 twos) |
| exact matches | 91 of 150 (61%) |
| **plain kappa** | **0.408**, moderate agreement |
| **weighted kappa** (quadratic weights) | **0.611**, substantial agreement |
| gross disagreements (two steps apart) | 3 of 150 |

**What that number does not mean.** The check annotation was done by a language model of
another family, not by a human. The kappa measures agreement between two automatic annotators
and **does not demonstrate agreement with a human expert**. That is a limitation of the
dataset, not a footnote.

**How the number was obtained.** The first round gave a kappa of 0.387, below the threshold the
project had set for itself. The review showed that the disagreement was not about substance
(there was a single gross disagreement in 150 pairs) but about two boundaries of the scale the
guideline left unsaid: what counts as "related to the matter" against "unrelated", and whether
a rule written for a different kind of organisation counts as a direct answer. Both boundaries
were made explicit and **the whole sample was annotated again**, because once the guideline
changed, what was annotated before and after is not comparable. The first round enters no
number.

**Where disagreement remains.** Of the judge's 48 twos the checker confirms 20: the judge is
more generous with "direct answer". Symmetrically, among its 50 ones the checker finds 8 twos.
The figure for "queries with a direct answer" should therefore be read as an upper bound.

## Relevance grades

* **2**: the chunk answers the query directly;
* **1**: related to the matter, contains no answer;
* **0**: unrelated.

Without grades NDCG degenerates into a binary metric. A grade of 1 is assigned automatically
only where a neighbouring chunk physically contains a substantial part of the gold text
(overlap at least 0.30). That threshold is a measurement over the corpus: half of the
neighbouring pairs do not overlap at all, and among those that do the median overlap is 0.15,
that is a neighbour holds a seventh of the gold chunk. Writing those down as relevant would
pass a guess off as a measurement.

The annotation guideline is `docs/ANNOTATION_GUIDE.md` in the repository. It is the same
document for the human and for the judge, and that identity is checked by a test.

## Layout on disk

```
data/queries/
  split.json               the split by act
  explan_live.jsonl        live questions: act, clauses, annotation route
  manual_task.tsv          the list of acts for hand written queries
  manual.jsonl             hand written queries
  synthetic.jsonl          the earlier 150 synthetic questions
  queries.jsonl            all query texts for a run
  pool_candidates.tsv      the annotation sheet over the union of runs
  qrels_by_clause.tsv      annotation by named clauses, with the evidence
  qrels_judge.tsv          judge annotation
  qrels.tsv                the final annotation
  podvyborki.json          subset membership
data/benchmark/            the MTEB export, which is what is published here
```

The annotation is **append only**. A query and chunk pair, once published, has to stay: earlier
runs were computed over it, and a gold pair that quietly disappears would look like a retrieval
failure. The build checks this and refuses to write the file if a pair went missing.

## What is not here

* currently effective revisions of the acts, only the revision as published in the bulletin;
* federal laws, reporting forms and letters: for 1299 of the collected live questions there is
  no answer in the corpus at all;
* amending documents of the "On amendments to clause 3.1" kind: 2527 such documents were left
  out, since they carry no standalone text;
* the Bank's answers to the questions, only the questions themselves are collected.
