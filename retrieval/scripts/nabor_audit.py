#!/usr/bin/env python3
"""Семь показателей набора по записанным определениям, снимок в один файл.

Определения записаны в `docs/18-nabor-do-i-posle.md` (раздел «Определения») до
первого замера, и этот скрипт считает ровно их: плакат «как было» и плакат «что
стало» читают снимки этого скрипта, а не текст. Поменять определение здесь,
не поменяв раздел определений, значит переопределить показатель задним числом.

Подвыборки не смешиваются: синтетика и живые считаются отдельно везде, где
это имеет смысл. Общая цифра по всем 991 запросу печатается только у M1,
потому что она записана в определениях до замера, и стоит рядом с подвыборками.

    python3 scripts/nabor_audit.py --tag do
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.benchmark import read_qrels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKS = os.path.join(ROOT, "data", "chunks", "base.jsonl")
OUTDIR = os.path.join(ROOT, "data", "results")

LIVE = "живые"
SYNTHETIC = ("синтетика", "dev", "невиданные акты")
LONG_QUERY = 600

# Откуда пара попала в разметку. Порядок повторяет приоритет сборки
# (`scripts/build_benchmark.py`): эталон по построению, названные пункты,
# судья, а автоматическая оценка 1 за перекрытие с эталоном остается последней.
SRC_GENERATION = "источник генерации"
SRC_CLAUSE = "пункты, названные Банком России"
SRC_JUDGE = "судья"
SRC_OVERLAP = "сосед по перекрытию с эталоном"


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_tsv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def share(part: int, whole: int) -> dict:
    return {"числитель": part, "знаменатель": whole,
            "доля": round(part / whole, 4) if whole else None}


def p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def digest(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def gold_of(q: dict) -> set[str]:
    return set(q.get("gold_chunk_ids") or ([q["gold_chunk_id"]] if q.get("gold_chunk_id") else []))


def confirmed_pairs() -> tuple[set, set]:
    """Пары с ненулевой оценкой судьи и с ненулевой оценкой человека."""
    judge = {(r["query_id"], r["chunk_id"]) for r in read_tsv(os.path.join(QDIR, "judge.tsv"))
             if r["otsenka_0_1_2"].strip() in ("1", "2")}
    human = {(r["query_id"], r["chunk_id"]) for r in read_tsv(os.path.join(QDIR, "human_sample.tsv"))
             if r["otsenka_0_1_2"].strip() in ("1", "2")}
    # Вторая ручная выборка размечена двумя вопросами: ответ во фрагменте
    # целиком или вместе с соседями. Любое «да» значит, что фрагмент по делу.
    human |= {(r["query_id"], r["chunk_id"])
              for r in read_tsv(os.path.join(QDIR, "human_sample_new.tsv"))
              if "да" in (r["otvet1_celikom"].strip(), r["otvet2_vmeste"].strip())}
    return judge, human


def load_labels(path: str) -> tuple[dict[str, dict[str, int]], dict[tuple[str, str], set[str]]]:
    """Метки и фрагменты за каждой меткой.

    С задачи 11.1 метка может стоять на пункте (`qrels_punkt.tsv`, столбец
    `chunk-ids`), и тогда за ней несколько фрагментов. У прежних файлов за
    меткой ровно один фрагмент, и показатели считаются как раньше.
    """
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    if "punkt-id" not in header:
        qrels = read_qrels(path)
        return qrels, {(q, c): {c} for q, rel in qrels.items() for c in rel}
    qrels: dict[str, dict[str, int]] = collections.defaultdict(dict)
    members = {}
    for r in read_tsv(path):
        qrels[r["query-id"]][r["punkt-id"]] = int(r["score"])
        members[(r["query-id"], r["punkt-id"])] = set(r["chunk-ids"].split(","))
    return dict(qrels), members


def density(qrels: dict[str, dict[str, int]], qids: list[str]) -> dict:
    counts = [sum(1 for s in qrels[q].values() if s > 0) for q in qids]
    return {"запросов": len(qids), "меток": sum(counts),
            "среднее": round(sum(counts) / len(counts), 2),
            "медиана": statistics.median(counts),
            "ровно одна метка": share(sum(1 for c in counts if c == 1), len(counts))}


def lengths(texts: list[str]) -> dict:
    n = [len(t) for t in texts]
    return {"запросов": len(n), "медиана": statistics.median(n), "95-я перцентиль": round(p95(n)),
            "максимум": max(n), f"длиннее {LONG_QUERY}": share(sum(1 for x in n if x > LONG_QUERY), len(n))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="имя снимка: do, 11N, posle")
    ap.add_argument("--qrels", default=os.path.join(QDIR, "qrels.tsv"))
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--train", default=os.path.join(QDIR, "synthetic_train.jsonl"))
    ap.add_argument("--copy-threshold", type=float, default=None,
                    help="порог copy_share этапа; на 29.10.2026 его нет")
    ap.add_argument("--out", help="куда писать; по умолчанию data/results/nabor-audit-<тег>.json")
    args = ap.parse_args()

    queries = {q["query_id"]: q for q in read_jsonl(args.queries)}
    qrels, members = load_labels(args.qrels)
    by_clause = read_qrels(os.path.join(QDIR, "qrels_by_clause.tsv"))
    by_judge = read_qrels(os.path.join(QDIR, "qrels_judge.tsv"))
    train = read_jsonl(args.train)
    chunks = read_jsonl(CHUNKS)
    judge_yes, human_yes = confirmed_pairs()

    subsets: dict[str, list[str]] = collections.defaultdict(list)
    for qid in sorted(qrels):
        subsets[queries[qid]["podvyborka"]].append(qid)

    m1 = {"все": density(qrels, sorted(qrels)),
          "подвыборки": {name: density(qrels, ids) for name, ids in sorted(subsets.items())}}
    m1["синтетика вместе"] = density(qrels, [q for s in SYNTHETIC for q in subsets[s]])

    m2: dict[str, dict] = {}
    groups = dict(sorted(subsets.items()))
    groups["синтетика вместе"] = [q for s in SYNTHETIC for q in subsets[s]]
    for name, ids in groups.items():
        by_source: collections.Counter = collections.Counter()
        judged = humaned = either = 0
        for qid in ids:
            gold = gold_of(queries[qid])
            for label, score in qrels[qid].items():
                if score <= 0:
                    continue
                inside = members[(qid, label)]
                if "#п" in label or inside & set(by_clause.get(qid, {})):
                    src = SRC_CLAUSE
                elif inside & gold:
                    src = SRC_GENERATION if queries[qid]["origin"] == "синтетический" else SRC_CLAUSE
                elif inside & set(by_judge.get(qid, {})):
                    src = SRC_JUDGE
                else:
                    src = SRC_OVERLAP
                by_source[src] += 1
                pairs = {(qid, c) for c in inside}
                judged += bool(pairs & judge_yes)
                humaned += bool(pairs & human_yes)
                either += bool(pairs & (judge_yes | human_yes))
        total = sum(by_source.values())
        m2[name] = {"позитивов": total,
                    "по происхождению": {k: share(v, total) for k, v in sorted(by_source.items())},
                    "подтверждено разметкой": share(either, total),
                    "из них судьей": judged, "из них человеком": humaned}

    live_ids = subsets[LIVE]
    acts = collections.Counter(cid.split("#")[0] for q in live_ids
                               for cid, s in qrels[q].items() if s > 0)
    top_act, top_n = acts.most_common(1)[0]
    m3 = {"запросов": len(live_ids), "актов": len(acts), "самый частый акт": top_act,
          "метки самого частого акта": share(top_n, sum(acts.values()))}

    train_sources = {t["gold_chunk_id"] for t in train}
    def fragments(ids) -> set[str]:
        return {c for q in ids for label, s in qrels[q].items() if s > 0 for c in members[(q, label)]}

    test_gold = fragments(qrels)
    leaked = test_gold & train_sources
    m4 = {"пересечение": share(len(leaked), len(test_gold)),
          "по подвыборкам": {name: len(fragments(ids) & train_sources)
                             for name, ids in sorted(subsets.items())},
          "фрагменты": sorted(leaked)}

    corpus_acts = {c["act_id"] for c in chunks}
    m5 = {"обучающих вопросов": len(train),
          "фрагменты": share(len(train_sources & {c["chunk_id"] for c in chunks}), len(chunks)),
          "акты": share(len({t["act_id"] for t in train} & corpus_acts), len(corpus_acts))}

    copy = [t["copy_share"] for t in train]
    m6 = {"вопросов": len(copy), "медиана": statistics.median(copy), "95-я перцентиль": round(p95(copy), 2),
          "максимум": max(copy), "порог": args.copy_threshold,
          "выше порога": (share(sum(1 for c in copy if c > args.copy_threshold), len(copy))
                          if args.copy_threshold is not None else None)}

    all_live = [q["text"] for q in queries.values() if q["podvyborka"] == LIVE]
    m7 = {"порог длины": LONG_QUERY, "живые все": lengths(all_live),
          "живые в тесте": lengths([queries[q]["text"] for q in live_ids]),
          "синтетика": lengths([queries[q]["text"] for s in SYNTHETIC for q in subsets[s]])}

    out = {"тег": args.tag, "определения": "docs/18-nabor-do-i-posle.md, раздел «Определения»",
           "аргументы": {"--qrels": os.path.relpath(args.qrels, ROOT), "--train": os.path.relpath(args.train, ROOT),
                         "--queries": os.path.relpath(args.queries, ROOT)}
           | ({"--copy-threshold": args.copy_threshold} if args.copy_threshold is not None else {}),
           "входы": {os.path.relpath(p, ROOT): digest(p)
                     for p in (args.qrels, args.queries, args.train, CHUNKS)},
           "M1 плотность меток": m1, "M2 происхождение позитива": m2, "M3 концентрация теста": m3,
           "M4 утечка": m4, "M5 покрытие корпуса": m5, "M6 копирование": m6, "M7 длина запроса": m7}
    path = args.out or os.path.join(OUTDIR, f"nabor-audit-{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"M1 одна метка: все {m1['все']['ровно одна метка']}, живые "
          f"{m1['подвыборки'][LIVE]['ровно одна метка']}")
    print(f"M2 живые подтверждено разметкой: {m2[LIVE]['подтверждено разметкой']}")
    print(f"M3 самый частый акт {top_act}: {m3['метки самого частого акта']}")
    print(f"M4: {m4['пересечение']}, по подвыборкам {m4['по подвыборкам']}")
    print(f"M5: фрагменты {m5['фрагменты']}, акты {m5['акты']}")
    print(f"M6: медиана {m6['медиана']}, 95-я перцентиль {m6['95-я перцентиль']}, максимум {m6['максимум']}")
    print(f"M7 живые все: {m7['живые все']}")
    print(f"записано: {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
