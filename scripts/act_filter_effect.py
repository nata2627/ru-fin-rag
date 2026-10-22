#!/usr/bin/env python3
"""Фильтр по номеру акта: сузить корпус до названного акта и искать внутри.

В живых вопросах автор сам называет акт, и сейчас этот номер работает только
через лексическое совпадение в BM25. Здесь он работает как фильтр по
метаданным: номер читается из текста вопроса (`rufin.references.act_numbers`),
и поиск идет только по фрагментам этого акта. Плотная ступень от такого
фильтра выигрывает больше лексической: номер в вектор почти не попадает.

Номер берется из текста вопроса, а не из темы раздела «Разъяснений»: сервис
темы не видит, и фильтр, построенный на ней, в проде был бы неповторим.

Опыт парный: обе ветки, с фильтром и без, считаются здесь же одним кодом на
одних и тех же запросах. Сохраненные выдачи этапа B для этого не годятся:
разница между реализациями смешалась бы с действием фильтра. Поэтому ветка
без фильтра сверяется с сохраненной выдачей, и доля совпадения печатается.

Запросы без номера и запросы, чей акт в корпус не попал, считаются
отдельными группами: на них фильтр не применяется по построению, и
усреднение с ними скрыло бы его действие.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.references import act_numbers  # noqa: E402
from rufin.retrieval.bm25 import BM25Index  # noqa: E402
from rufin.retrieval.dense import DenseIndex  # noqa: E402
from rufin.retrieval.hybrid import rrf  # noqa: E402
from rufin.retrieval.model_specs import MODELS  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
RUNDIR = os.path.join(ROOT, "data", "runs")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
EMBDIR = os.path.join(ROOT, "data", "embeddings")
REPORT = os.path.join(ROOT, "docs", "raw", "act_filter_effect.txt")
RESULT = os.path.join(ROOT, "data", "results", "act-filter.json")
DEPTH = 50
TOP = 10
METRICS = ("Recall@10", "NDCG@10")


def load_chunks(path: str) -> tuple[list[str], list[str], list[str]]:
    ids, texts, numbers = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ids.append(rec["chunk_id"])
            texts.append(rec["text"])
            numbers.append(rec["number"])
    return ids, texts, numbers


def load_run(path: str) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        return {r["query_id"]: r["ranked"]
                for r in (json.loads(l) for l in f if l.strip())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--model", default="user-bge-m3")
    ap.add_argument("--subset", default="живые")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    ids, texts, numbers = load_chunks(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"))
    by_number: dict[str, list[int]] = collections.defaultdict(list)
    for i, number in enumerate(numbers):
        by_number[number].append(i)
    say(f"нарезка {args.chunks}: фрагментов {len(ids)}, актов {len(by_number)}")

    qrels = read_qrels(os.path.join(QDIR, "qrels.tsv"))
    subsets = json.load(open(os.path.join(QDIR, "podvyborki.json"), encoding="utf-8"))
    queries = {}
    with open(os.path.join(QDIR, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["query_id"]] = q["text"]
    qids = sorted(q for q in subsets[args.subset] if q in qrels and q in queries)
    say(f"подвыборка «{args.subset}»: запросов с эталоном {len(qids)}")

    # Группы запросов. Фильтр применяется только там, где названный акт
    # в корпусе есть; остальные две группы это контроль.
    allowed_of: dict[str, np.ndarray] = {}
    group: dict[str, str] = {}
    for qid in qids:
        found = act_numbers(queries[qid])
        rows = [i for number in found for i in by_number.get(number, [])]
        if not found:
            group[qid] = "номер не назван"
        elif not rows:
            group[qid] = "номер назван, акта нет в корпусе"
        else:
            group[qid] = "номер назван, акт в корпусе"
            allowed_of[qid] = np.array(sorted(set(rows)), dtype=int)
    counts = collections.Counter(group.values())
    for name, n in counts.most_common():
        say(f"  {name:<34} {n}")
    sizes = [len(a) for a in allowed_of.values()]
    if sizes:
        say(f"  фрагментов в суженном корпусе: медиана {int(np.median(sizes))}, "
            f"от {min(sizes)} до {max(sizes)} (из {len(ids)})")

    t0 = time.time()
    index = BM25Index.build(ids, texts)
    say(f"индекс BM25 построен за {index.build_seconds:.0f} с, {index.size_mb:.0f} МБ")
    del texts

    emb = os.path.join(EMBDIR, args.chunks, args.model)
    dense = DenseIndex.from_files(MODELS[args.model], os.path.join(emb, "vectors.npy"),
                                 os.path.join(emb, "ids.txt"), backend="numpy")
    # Один раз во float32: иначе приведение повторялось бы на каждом запросе
    # и стоило бы дороже самого поиска.
    dense.vectors = dense.vectors.astype("float32")
    if dense.ids != ids:
        raise SystemExit("порядок фрагментов в матрице и в нарезке разошелся")
    qvec_all = np.load(os.path.join(emb, f"{args.model}-1024-queries.npy"))
    order = open(os.path.join(emb, f"{args.model}-1024-queries.ids.txt"),
                 encoding="utf-8").read().split()
    qplace = {qid: i for i, qid in enumerate(order)}
    missing = [q for q in qids if q not in qplace]
    if missing:
        raise SystemExit(f"нет векторов запросов: {len(missing)}, первый {missing[0]}")
    say(f"матрица {dense.vectors.shape}, векторы запросов {qvec_all.shape}")
    say(f"подготовка {time.time() - t0:.0f} с")

    runs: dict[str, dict[str, list[str]]] = {name: {} for name in (
        "bm25", "bm25+фильтр", "dense", "dense+фильтр", "hybrid", "hybrid+фильтр")}
    t0 = time.time()
    for qid in qids:
        text = queries[qid]
        qvec = qvec_all[qplace[qid]:qplace[qid] + 1].astype("float32")
        lex = index.search(text, DEPTH)
        den = dense.search_vectors(qvec, DEPTH)[0]
        runs["bm25"][qid] = [c for c, _ in lex[:TOP]]
        runs["dense"][qid] = [c for c, _ in den[:TOP]]
        runs["hybrid"][qid] = [c for c, _ in rrf([lex, den])[:TOP]]
        allowed = allowed_of.get(qid)
        if allowed is None:
            for name in ("bm25", "dense", "hybrid"):
                runs[f"{name}+фильтр"][qid] = list(runs[name][qid])
            continue
        lex_f = index.search(text, DEPTH, allowed=allowed)
        den_f = dense.search_vectors(qvec, DEPTH, allowed=allowed)[0]
        runs["bm25+фильтр"][qid] = [c for c, _ in lex_f[:TOP]]
        runs["dense+фильтр"][qid] = [c for c, _ in den_f[:TOP]]
        runs["hybrid+фильтр"][qid] = [c for c, _ in rrf([lex_f, den_f])[:TOP]]
    say(f"поиск по {len(qids)} запросам в шести вариантах: {time.time() - t0:.0f} с")

    # Сверка ветки без фильтра с выдачей этапа B: если реализация разошлась,
    # разница в метрике будет не про фильтр.
    say()
    for name, saved in (("bm25", "base__bm25.jsonl"),
                        ("dense", f"base__dense-{args.model}.jsonl")):
        path = os.path.join(RUNDIR, saved)
        if not os.path.exists(path):
            continue
        ref = load_run(path)
        same = [len(set(runs[name][q]) & set(ref[q][:TOP])) / TOP
                for q in qids if q in ref]
        say(f"сверка «{name}» с {saved}: совпадение первой десятки "
            f"{100 * float(np.mean(same)):.1f}% на {len(same)} запросах")

    # Почему лексический поиск выигрывает от фильтра больше плотного. Версия:
    # номер названного акта встречается в текстах других актов, они ссылаются
    # друг на друга, и BM25 за это совпадение цепляется. Проверяется прямо:
    # сколько фрагментов первой десятки взяты не из названного акта и при этом
    # содержат его номер в тексте.
    say()
    wanted = {cid for name in ("bm25", "dense") for q in allowed_of
              for cid in runs[name][q]}
    body_of: dict[str, str] = {}
    act_of_chunk: dict[str, str] = {}
    with open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["chunk_id"] in wanted:
                body_of[rec["chunk_id"]] = rec["text"].upper()
                act_of_chunk[rec["chunk_id"]] = rec["number"]
    diagnostics = {}
    for name in ("bm25", "dense"):
        alien = mentions = total = 0
        for qid in allowed_of:
            asked = set(act_numbers(queries[qid]))
            for cid in runs[name][qid]:
                total += 1
                if act_of_chunk.get(cid) in asked:
                    continue
                alien += 1
                if any(number in body_of.get(cid, "") for number in asked):
                    mentions += 1
        diagnostics[name] = {"позиций": total, "из чужого акта": alien,
                             "из чужого акта, но с номером в тексте": mentions}
        say(f"«{name}» без фильтра на {len(allowed_of)} вопросах с номером: "
            f"из {total} позиций первой десятки {alien} взяты не из названного "
            f"акта, и {mentions} из них содержат его номер в тексте")

    per_q = {name: M.per_query({q: runs[name][q] for q in qids},
                               {q: qrels[q] for q in qids})
             for name in runs}
    rows = []
    say()
    say(f"{'вариант':<16} " + " ".join(f"{m:>24}" for m in METRICS))
    for name in runs:
        say(f"{name:<16} " + " ".join(
            f"{str(M.bootstrap_ci(per_q[name][m])):>24}" for m in METRICS))

    say()
    say("прибавка фильтра, парный бутстрэп по запросам, p перестановочный:")
    for base in ("bm25", "dense", "hybrid"):
        parts = []
        row = {"связка": base}
        for m in METRICS:
            diff, p = M.paired_diff_ci(per_q[f"{base}+фильтр"][m], per_q[base][m])
            parts.append(f"{m} {diff} p={p:.3f}")
            row[m] = {"diff": round(diff.mean, 4), "lo": round(diff.lo, 4),
                      "hi": round(diff.hi, 4), "p": round(p, 4)}
        say(f"  {base:<14} " + "   ".join(parts))
        rows.append(row)

    say()
    say("то же по группам запросов:")
    group_rows = []
    for name in sorted(counts):
        part = [q for q in qids if group[q] == name]
        if len(part) < 10:
            say(f"  {name}: {len(part)} запросов, интервал шире метрики, не считаем")
            continue
        say(f"  {name} ({len(part)} запросов):")
        for base in ("bm25", "dense", "hybrid"):
            sub_now = M.per_query({q: runs[base][q] for q in part},
                                  {q: qrels[q] for q in part})
            sub_f = M.per_query({q: runs[f"{base}+фильтр"][q] for q in part},
                                {q: qrels[q] for q in part})
            parts = []
            row = {"группа": name, "запросов": len(part), "связка": base}
            for m in METRICS:
                diff, p = M.paired_diff_ci(sub_f[m], sub_now[m])
                parts.append(f"{m} {diff} p={p:.3f}")
                row[m] = {"diff": round(diff.mean, 4), "lo": round(diff.lo, 4),
                          "hi": round(diff.hi, 4), "p": round(p, 4)}
            say(f"     {base:<12} " + "   ".join(parts))
            group_rows.append(row)

    levels = {name: {m: round(M.bootstrap_ci(per_q[name][m]).mean, 4) for m in METRICS}
              for name in runs}
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump({"подвыборка": args.subset, "запросов": len(qids),
                   "модель": args.model, "глубина слияния": DEPTH,
                   "группы": dict(counts), "уровни": levels,
                   "прибавка": rows, "прибавка по группам": group_rows,
                   "чужие акты в первой десятке": diagnostics},
                  f, ensure_ascii=False, indent=2)
    say()
    say(f"файлы: {os.path.relpath(REPORT, ROOT)}, {os.path.relpath(RESULT, ROOT)}")


if __name__ == "__main__":
    main()
