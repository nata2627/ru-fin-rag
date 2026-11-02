#!/usr/bin/env python3
"""Stage 12 live set: labeler-only qrels, clause units, dev/test split, run metrics.

Decision 26: labels come from the labeler only. The old union with clauses named
by the Bank of Russia and with the judge is dropped, so the clause conversion runs
without named clauses (long-clause pieces only). A question with no positive goes
to the "no answer" subset: it never enters a metric and is printed with its
denominator.

Label sources (one labeler build, `razmetka_yadro.py`):

* 142 old: sheet `data/queries/razmetka_yadro.tsv`;
* 183 old reserve: sheet `data/queries/razmetka_extra.tsv` plus the tail
  `data/interim/live12/razmetka_tail`;
* 677 new: `data/interim/live12/razmetka_main` and, once it exists,
  `data/interim/live12/razmetka_rr` (hybrid-rerank pairs outside the main pool);
* earlier positives the labeler never saw (named clauses, the old judge, outside the
  labeled pool core): `data/interim/live12/razmetka_unseen`, so that dropping the union
  does not drop questions whose only answer sat outside the core.

    python3 scripts/live12_build.py
    python3 scripts/live12_build.py --allow-partial   # dry run on partial labels
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import live12_candidates as C  # noqa: E402
import razmetka_yadro as Y  # noqa: E402

from rufin import gate as G  # noqa: E402
from rufin import punkt as P  # noqa: E402
from rufin.benchmark import LETTER_CHARS, read_qrels, write_qrels  # noqa: E402

ROOT = Y.ROOT
QDIR = Y.QDIR
LIVE12 = os.path.join(ROOT, "data", "interim", "live12")
QUERIES = os.path.join(LIVE12, "live_all.jsonl")
SHEETS = (os.path.join(QDIR, "razmetka_yadro.tsv"), os.path.join(QDIR, "razmetka_extra.tsv"))
WORKS = tuple(os.path.join(LIVE12, n) for n in ("razmetka_tail", "razmetka_main", "razmetka_unseen", "razmetka_rr"))
# The hybrid-rerank pair folder appears only after the Kaggle run.
OPTIONAL_WORKS = (os.path.join(LIVE12, "razmetka_rr"),)
CORE_MODEL_ONLY = os.path.join(QDIR, "qrels_live_core_model_only.tsv")
RUN_DIRS = (os.path.join(ROOT, "data", "runs"), os.path.join(LIVE12, "runs"))
CHUNKS = Y.CHUNKS

OUT_QRELS = os.path.join(QDIR, "qrels_live12.tsv")
OUT_PUNKT = os.path.join(QDIR, "qrels_live12_punkt.tsv")
OUT_QUERIES = os.path.join(QDIR, "live12.jsonl")
OUT_SUBSETS = os.path.join(QDIR, "podvyborki_live12.json")
RESULT = os.path.join(ROOT, "data", "results", "live12.json")

CONFIGS = ("bm25", "dense-user-bge-m3", "dense-rufin", "hybrid", "hybrid-rerank")
PAIRS = (("hybrid-rerank", "hybrid"), ("hybrid", "bm25"))
ROUTES = ("раздел", "текст", "тема", "форма")
DOMINANT_ACT = "590-П_28062017"
DEV_SHARE = 0.25
SEED = 12
# Rarer acts merge into one bucket: with a quarter going to dev, a stratum of
# three questions would get zero or one dev question by rounding alone.
MIN_ACT = 8
# Candidates were grouped at 0.9, and a check found pairs at 0.8 that differ by a date or a
# figure on both sides of the split (exp0356 and exp0495). The set is regrouped here at 0.8,
# so the question selection made on the 0.9 groups stays as it was.
DUP = 0.8
OTHER_ACTS = "прочие"
DEV, TEST, NO_ANSWER, UNLABELED = "dev", "test", "нет ответа", "не размечен"


def share(part: int, whole: int) -> dict:
    return {"числитель": part, "знаменатель": whole}


def sheet_labels(path: str) -> dict[tuple[str, str], int]:
    return {(r["query_id"], r["chunk_id"]): Y.grade(r["otvet1_celikom"], r["otvet2_vmeste"])
            for r in Y.read_tsv(path)}


def work_labels(work: str) -> tuple[dict[tuple[str, str], int], list[str], set[str]]:
    """Labels of a batch folder, batches without answers, questions not fully labeled."""
    Y.WORK = work
    manifest = json.load(open(os.path.join(work, "manifest.json"), encoding="utf-8"))
    todo = [f"{b:03d}" for b in range(1, manifest["партий"] + 1)
            if not os.path.exists(os.path.join(work, f"batch_{b:03d}.answers.tsv"))]
    labels = {k: Y.grade(a, b) for k, (a, b, _) in Y.collect().items()}
    pairs = [(r["query_id"], r["chunk_id"]) for r in Y.read_tsv(os.path.join(work, "pairs.tsv"))]
    return labels, todo, {q for q, c in pairs if (q, c) not in labels}


def gather(sheets=SHEETS, works=WORKS, optional=OPTIONAL_WORKS, allow_partial: bool = False):
    """All labeler grades. A pair labeled twice keeps the higher grade."""
    labels: dict[tuple[str, str], int] = {}
    report, incomplete, todo_total = {}, set(), 0

    def add(got: dict[tuple[str, str], int]) -> int:
        twice = 0
        for k, g in got.items():
            twice += k in labels
            labels[k] = max(labels.get(k, 0), g)
        return twice

    for path in sheets:
        got = sheet_labels(path)
        report[os.path.relpath(path, ROOT)] = {"пар": len(got), "позитивов": sum(g > 0 for g in got.values()),
                                               "повторов": add(got)}
    for work in works:
        name = os.path.relpath(work, ROOT)
        if not os.path.exists(os.path.join(work, "manifest.json")):
            if work in optional:
                report[name] = "папки нет"
                continue
            raise SystemExit(f"{name}: нет manifest.json")
        got, todo, partial = work_labels(work)
        todo_total += len(todo)
        incomplete |= partial
        report[name] = {"пар": len(got), "позитивов": sum(g > 0 for g in got.values()), "повторов": add(got),
                        "партий без ответа": len(todo), "вопросов размечено не полностью": len(partial)}
    if todo_total and not allow_partial:
        raise SystemExit(f"партий без ответа {todo_total}; для пробы на неполной разметке --allow-partial")
    return labels, report, incomplete, todo_total


def labeler_qrels(labels: dict[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for (q, c), g in labels.items():
        if g > 0:
            out[q][c] = g
    return dict(out)


def check_core(qrels: dict[str, dict[str, int]], labels: dict[tuple[str, str], int], core_ids: set[str],
               path: str = CORE_MODEL_ONLY, sheet: str = SHEETS[0]) -> dict:
    """Compare the 142 with the old "model only" file.

    That file also kept old positives on pairs the labeler never saw (named
    clauses, judge, gold outside the pool core). So the files must agree on
    pairs the labeler saw, and the unseen ones are reported as a count.
    """
    old = read_qrels(path)
    # Only the core sheet: pairs relabeled later (razmetka_unseen) were old positives
    # in that file by construction, so they are expected to differ.
    seen = set(sheet_labels(sheet))
    old_seen = {(q, c): s for q, r in old.items() for c, s in r.items() if s > 0 and (q, c) in seen}
    new_seen = {(q, c): s for q, r in qrels.items() if q in core_ids for c, s in r.items() if (q, c) in seen}
    if old_seen != new_seen:
        diff = sorted(set(old_seen.items()) ^ set(new_seen.items()))
        raise SystemExit(f"142: на парах разметчика расхождение с {os.path.basename(path)}: {diff[:5]}")
    unseen = [(q, c) for q, r in old.items() for c, s in r.items() if s > 0 and (q, c) not in seen]
    with_pos_old = {q for q, r in old.items() if any(s > 0 for s in r.values())}
    with_pos_new = {q for q in core_ids if qrels.get(q)}
    return {"позитивов разметчика совпало": len(new_seen),
            "прежних позитивов, которых разметчик не видел": len(unseen),
            "вопросов с позитивом в прежнем файле": len(with_pos_old),
            "вопросов с позитивом только у разметчика": len(with_pos_new),
            "вопросов, которые без невиденных позитивов остались без ответа":
                sorted(with_pos_old - with_pos_new)}


def load_chunk_layout(path: str = CHUNKS) -> list[dict]:
    """Chunk fields needed to glue long-clause pieces. Bodies are skipped: without
    named clauses the gluing does not use them, and they weigh a quarter of a gigabyte."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            out.append({"chunk_id": c["chunk_id"], "act_id": c["act_id"], "position": c["position"],
                        "section": c["section"] or "", "units": c["units"] or "",
                        "char_start": c["char_start"], "char_end": c["char_end"]})
    return out


def to_units(qrels: dict[str, dict[str, int]], chunks: list[dict]) -> dict[str, dict[str, P.Unit]]:
    # No named clauses (decision 26, item 1): a unit is a chunk or the pieces of
    # one long clause, as for judge and synthetic labels in task 11.1.
    return P.build_units(qrels, {}, {}, {}, P.piece_groups(chunks), None)


def is_letter(text: str) -> bool:
    return len(text) > LETTER_CHARS


def stratum(q: dict, common_acts: set[str]) -> tuple[str, str, str]:
    act = q.get("act_id") or "-"
    return ("прежний" if q["old"] else "новый", q.get("route") or "-", act if act in common_acts else OTHER_ACTS)


def near_groups(rows: list[dict], threshold: float = DUP) -> list[int]:
    """Union of the candidate groups and word Jaccard groups at the threshold, as row indices."""
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[object, int] = {}
    for i, (r, g) in enumerate(zip(rows, C.groups([r["text"] for r in rows], threshold))):
        for key in (("candidate", r["group"]), ("text", g)):
            parent[find(i)] = find(first.setdefault(key, i))
    return [find(i) for i in range(len(rows))]


def split_dev_test(rows: list[dict], seed: int = SEED, dev_share: float = DEV_SHARE,
                   min_act: int = MIN_ACT) -> dict[str, str]:
    """query_id -> dev or test. A near-duplicate group stays in one part.

    A group belongs to the stratum of its first question. The dev target of a
    stratum uses cumulative rounding, so the strata add up to the exact share.
    """
    acts = collections.Counter(q.get("act_id") or "-" for q in rows)
    common = {a for a, n in acts.items() if n >= min_act}
    groups: dict[object, list[dict]] = collections.defaultdict(list)
    for q in sorted(rows, key=lambda q: q["query_id"]):
        groups[q["group"]].append(q)
    strata: dict[tuple, list[list[dict]]] = collections.defaultdict(list)
    for members in groups.values():
        strata[stratum(members[0], common)].append(members)
    rng = random.Random(seed)
    out, seen, dev_total = {}, 0, 0
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda m: m[0]["query_id"])
        rng.shuffle(members)
        size = sum(len(m) for m in members)
        target = round((seen + size) * dev_share) - dev_total
        seen += size
        for m in members:
            part = DEV if len(m) <= target else TEST
            if part == DEV:
                target -= len(m)
                dev_total += len(m)
            for q in m:
                out[q["query_id"]] = part
    return out


def assign(queries: list[dict], qrels: dict[str, dict[str, int]], incomplete: set[str]) -> dict[str, str]:
    split = {q["query_id"]: UNLABELED for q in queries if q["query_id"] in incomplete}
    for q in queries:
        if q["query_id"] not in split and not qrels.get(q["query_id"]):
            split[q["query_id"]] = NO_ANSWER
    split.update(split_dev_test([q for q in queries if q["query_id"] not in split]))
    return split


def ndcg_rows(runs: dict[str, dict[str, list[str]]], units: dict[str, dict[str, P.Unit]],
              ids: list[str]) -> dict[str, np.ndarray]:
    """Per-question clause NDCG@10, the same count as `punkt_effect.py`.

    A question without positives is refused: its metric is undefined, a nan
    would silently drop out of the mean and the denominator would lie."""
    bad = [q for q in ids if not any(u.score > 0 for u in units.get(q, {}).values())]
    if bad:
        raise ValueError(f"вопросы без позитива в метрике: {bad[:5]}")
    return {cfg: np.array([P.metrics(run[q], units[q])["NDCG@10"] for q in ids]) for cfg, run in runs.items()}


def load_runs(ids: set[str]) -> tuple[dict[str, dict[str, list[str]]], dict[str, str]]:
    runs, skipped = {}, {}
    for cfg in CONFIGS:
        run: dict[str, list[str]] = {}
        for d in RUN_DIRS:
            path = os.path.join(d, f"base__{cfg}.jsonl")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    for r in map(json.loads, f):
                        if r["query_id"] in ids:
                            run.setdefault(r["query_id"], r["ranked"])
        if set(run) >= ids:
            runs[cfg] = run
        else:
            skipped[cfg] = f"выдача есть для {len(set(run) & ids)} из {len(ids)} вопросов"
    return runs, skipped


def subsets_of(queries: list[dict], split: dict[str, str]) -> dict[str, list[str]]:
    test = [q for q in queries if split[q["query_id"]] == TEST]
    out = {"живые тест": [q["query_id"] for q in test],
           "живые dev": [q["query_id"] for q in queries if split[q["query_id"]] == DEV],
           "тест без 590-П": [q["query_id"] for q in test if q["без590"]],
           "тест письма": [q["query_id"] for q in test if q["letter"]],
           "тест без писем": [q["query_id"] for q in test if not q["letter"]],
           "тест прежние 325": [q["query_id"] for q in test if q["old"]]}
    for route in ROUTES:
        out[f"тест {route}"] = [q["query_id"] for q in test if q.get("route") == route]
    out[NO_ANSWER] = [q["query_id"] for q in queries if split[q["query_id"]] == NO_ANSWER]
    unlabeled = [q["query_id"] for q in queries if split[q["query_id"]] == UNLABELED]
    if unlabeled:
        out[UNLABELED] = unlabeled
    return {k: sorted(v) for k, v in out.items()}


def describe(queries: list[dict]) -> dict:
    acts = collections.Counter(q.get("act_id") or "-" for q in queries)
    return {"вопросов": len(queries),
            "прежних": sum(q["old"] for q in queries),
            "по маршрутам": dict(collections.Counter(q.get("route") or "-" for q in queries).most_common()),
            "акты топ-10": dict(acts.most_common(10)),
            "актов": len(acts),
            "590-П": share(acts[DOMINANT_ACT], len(queries)),
            "письма": share(sum(q["letter"] for q in queries), len(queries))}


def print_no_answer(queries: list[dict], total: int) -> None:
    by_route = collections.Counter(q.get("route") or "-" for q in queries)
    print(f"нет ответа в корпусе: {len(queries)} из {total} размеченных вопросов, в метрику не входят; "
          f"прежних {sum(q['old'] for q in queries)}, по маршрутам {dict(by_route.most_common())}")
    print("  " + " ".join(q["query_id"] for q in queries))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true",
                    help="проба на неполной разметке: вопросы, размеченные не до конца, выпадают и печатаются")
    args = ap.parse_args()

    with open(QUERIES, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]
    labels, sources, incomplete, todo = gather(allow_partial=args.allow_partial)
    old_ids = {q for q, _ in sheet_labels(SHEETS[0])} | {q for q, _ in sheet_labels(SHEETS[1])}
    core_ids = {q for q, _ in sheet_labels(SHEETS[0])}
    labeled_q = {q for q, _ in labels}
    ids = {q["query_id"] for q in raw}
    stray = labeled_q - ids
    if stray:
        raise SystemExit(f"метки на вопросы вне набора: {sorted(stray)[:5]}")
    never = ids - labeled_q
    if never and not args.allow_partial:
        raise SystemExit(f"вопросов без единой размеченной пары {len(never)}: {sorted(never)[:5]}")
    incomplete |= never

    qrels = labeler_qrels(labels)
    write_qrels(OUT_QRELS, qrels)
    core = check_core(qrels, labels, core_ids)
    units = to_units(qrels, load_chunk_layout())
    P.write_units(OUT_PUNKT, units)

    queries = [{"query_id": q["query_id"], "text": q["text"], "route": q.get("route"), "act_id": q.get("act_id"),
                "group": g, "old": q["query_id"] in old_ids, "letter": is_letter(q["text"]),
                "без590": q.get("act_id") != DOMINANT_ACT} for q, g in zip(raw, near_groups(raw))]
    split = assign(queries, qrels, incomplete)
    with open(OUT_QUERIES, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({**q, "split": split[q["query_id"]]}, ensure_ascii=False) + "\n")
    subsets = subsets_of(queries, split)
    with open(OUT_SUBSETS, "w", encoding="utf-8") as f:
        json.dump(subsets, f, ensure_ascii=False, indent=1)
        f.write("\n")

    by_id = {q["query_id"]: q for q in queries}
    runs, skipped = load_runs({q for q, s in split.items() if s in (DEV, TEST)})
    table = []
    for name, qids in subsets.items():
        if name in (NO_ANSWER, UNLABELED) or not qids:
            continue
        per = ndcg_rows(runs, units, qids)
        table.append({"подвыборка": name, "вопросов": len(qids),
                      **{cfg: round(float(v.mean()), 4) for cfg, v in per.items()}})
    test_ids = subsets["живые тест"]
    per_test = ndcg_rows(runs, units, test_ids)
    resolution = []
    for cand, ref in PAIRS:
        if cand not in per_test or ref not in per_test:
            resolution.append({"пара": f"{cand} против {ref}", "нет выдачи": [c for c in (cand, ref) if c in skipped]})
            continue
        v = G.решить(per_test[cand], per_test[ref], имя=cand, имя_эталона=ref)
        resolution.append({"пара": f"{cand} против {ref}", "вопросов": v.запросов,
                           "разница": {"среднее": round(v.разница.mean, 4), "lo": round(v.разница.lo, 4),
                                       "hi": round(v.разница.hi, 4), "p": v.p},
                           "полуширина 95%": round(v.различимое, 4), "установлена": v.установлена})

    no_answer = [by_id[q] for q in subsets[NO_ANSWER]]
    labeled_total = len(queries) - len(subsets.get(UNLABELED, []))
    grades = collections.Counter(labels.values())
    result = {"разметка полная": todo == 0 and not incomplete,
              "партий без ответа": todo,
              "источники меток": sources,
              "пар размечено": len(labels),
              "оценки разметчика": {str(k): grades[k] for k in sorted(grades)},
              "позитивов": sum(len(r) for r in qrels.values()),
              "единиц по пунктам": sum(sum(u.score > 0 for u in us.values()) for us in units.values()),
              "сверка 142 с qrels_live_core_model_only": core,
              "вопросов": describe(queries),
              "не размечено до конца": share(len(subsets.get(UNLABELED, [])), len(queries)),
              "нет ответа": {**share(len(no_answer), labeled_total), "состав": describe(no_answer)},
              "dev": describe([by_id[q] for q in subsets["живые dev"]]),
              "test": describe([by_id[q] for q in test_ids]),
              "подвыборки": {k: len(v) for k, v in subsets.items()},
              "метрика": "NDCG@10 по пунктам (rufin.punkt.metrics), метки только разметчика",
              "связки пропущены": skipped,
              "таблица": table,
              "разрешающая способность на тесте": resolution,
              "сплит": {"доля dev": DEV_SHARE, "семя": SEED, "акт реже": MIN_ACT, "порог дублей": DUP,
                        "слой": "прежний или новый, маршрут, акт"}}
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print(f"разметка {'полная' if result['разметка полная'] else 'НЕПОЛНАЯ'}: партий без ответа {todo}, "
          f"вопросов размечено не до конца {len(subsets.get(UNLABELED, []))} из {len(queries)}")
    print(f"пар {len(labels)}, позитивов {result['позитивов']}, единиц по пунктам {result['единиц по пунктам']}")
    print(f"сверка 142: {json.dumps(core, ensure_ascii=False)}")
    print_no_answer(no_answer, labeled_total)
    print(f"dev {len(subsets['живые dev'])}, test {len(test_ids)}; пропущены {skipped or 'нет'}")
    for row in table:
        print(f"  {row['подвыборка']:18s} {row['вопросов']:4d}  " +
              "  ".join(f"{c} {row[c]:.3f}" for c in runs))
    for r in resolution:
        print(f"  {json.dumps(r, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
