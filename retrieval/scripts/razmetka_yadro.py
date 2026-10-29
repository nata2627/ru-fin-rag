#!/usr/bin/env python3
"""Разметка ядра пула: все пары приоритета 1 по 142 живым вопросам теста.

Разметчик, инструкция, вид пары и размер партии те же, что в пробе
(`scripts/razmetka_proba.py`, задача 11.4): приемку прошла ровно эта сборка, и
менять в ней что-либо значило бы размечать непроверенным разметчиком. Пары из
контрольной выборки, которые лежат в ядре, повторно не размечаются: их метка
берется из первого прогона пробы.

Состояние на диске: папка `data/interim/razmetka_yadro/` (в git не идет), в ней
партии `batch_NNN.prompt.txt`, ответы `batch_NNN.answers.tsv` и манифест.
Прерванная разметка продолжается с партий, у которых нет ответа: `status`
их печатает.

    python3 scripts/razmetka_yadro.py prepare
    python3 scripts/razmetka_yadro.py status
    python3 scripts/razmetka_yadro.py apply --usage <usage.json>
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from razmetka_proba import RUBRIC, TASK, parse, read_tsv, render  # noqa: E402

from rufin.explanations import body_of  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
WORK = os.path.join(ROOT, "data", "interim", "razmetka_yadro")
POOL = os.path.join(QDIR, "pool_live.tsv")
CHUNKS = os.path.join(ROOT, "data", "chunks", "base.jsonl")
PROBA = os.path.join(QDIR, "razmetka_proba.tsv")
SHEET = os.path.join(QDIR, "razmetka_yadro.tsv")
CORE_QRELS = os.path.join(QDIR, "qrels_live_core.tsv")
MODEL_ONLY_QRELS = os.path.join(QDIR, "qrels_live_core_model_only.tsv")
RESULT = os.path.join(ROOT, "data", "results", "pool-labeling.json")
LIVE = "живые"
LIVE_SUBSETS = ("живые", "живые без писем", "письма", "живые без 590-П")
CORE = "1"
BATCH = 25
SEED = 11


def core_pairs() -> list[dict]:
    return [r for r in read_tsv(POOL) if r["prioritet"] == CORE]


def prepare(args) -> None:
    probed = {(r["query_id"], r["chunk_id"]) for r in read_tsv(PROBA)}
    pairs = [r for r in core_pairs() if (r["query_id"], r["chunk_id"]) not in probed]
    wanted = {r["chunk_id"] for r in pairs}
    chunks = {}
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["chunk_id"] in wanted:
                chunks[c["chunk_id"]] = c
    queries = {}
    with open(os.path.join(QDIR, "queries.jsonl"), encoding="utf-8") as f:
        for q in map(json.loads, f):
            queries[q["query_id"]] = q["text"]
    pairs.sort(key=lambda r: (r["query_id"], r["chunk_id"]))
    # Перемешивание как в контрольной выборке: подряд идущие пары одного
    # вопроса подталкивали бы разметчика сравнивать фрагменты между собой.
    random.Random(SEED).shuffle(pairs)
    rows = []
    for n, r in enumerate(pairs, start=1):
        c = chunks[r["chunk_id"]]
        rows.append({"nomer": str(n), "zapros": queries[r["query_id"]],
                     "akt": f"{c.get('number', '?')} от {c.get('date', '?')}",
                     "razdel": (c.get("section") or "")[:60], "punkty": c.get("units", ""),
                     "fragment": body_of(c), "query_id": r["query_id"], "chunk_id": r["chunk_id"]})
    os.makedirs(WORK, exist_ok=True)
    head = TASK + "ИНСТРУКЦИЯ\n\n" + RUBRIC + "\n\nПАРЫ\n\n"
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    for b, batch in enumerate(batches, start=1):
        with open(os.path.join(WORK, f"batch_{b:03d}.prompt.txt"), "w", encoding="utf-8") as f:
            f.write(head + "\n".join(render(r) for r in batch))
    with open(os.path.join(WORK, "pairs.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["nomer", "batch", "query_id", "chunk_id"])
        for i, r in enumerate(rows):
            w.writerow([r["nomer"], f"{i // BATCH + 1:03d}", r["query_id"], r["chunk_id"]])
    manifest = {"пар ядра": len(core_pairs()), "из контрольной выборки": len(core_pairs()) - len(rows),
                "размечать": len(rows), "партий": len(batches), "в партии": BATCH, "семя": SEED,
                "sha256 инструкции": hashlib.sha256(RUBRIC.encode()).hexdigest()[:16],
                "sha256 задания": hashlib.sha256(TASK.encode()).hexdigest()[:16]}
    with open(os.path.join(WORK, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps(manifest, ensure_ascii=False))


def status(args) -> list[str]:
    manifest = json.load(open(os.path.join(WORK, "manifest.json"), encoding="utf-8"))
    todo = [f"{b:03d}" for b in range(1, manifest["партий"] + 1)
            if not os.path.exists(os.path.join(WORK, f"batch_{b:03d}.answers.tsv"))]
    print(f"партий {manifest['партий']}, без ответа {len(todo)}: {' '.join(todo[:40])}"
          f"{' ...' if len(todo) > 40 else ''}")
    return todo


def collect() -> dict[tuple[str, str], tuple[str, str, str]]:
    """Ответы по всем партиям: (query_id, chunk_id) -> (вопрос 1, вопрос 2, тип)."""
    by_nomer = {}
    for name in sorted(os.listdir(WORK)):
        if name.endswith(".answers.tsv"):
            got = parse(os.path.join(WORK, name))
            batch = name.split("_")[1].split(".")[0]
            expected = [r for r in read_tsv(os.path.join(WORK, "pairs.tsv")) if r["batch"] == batch]
            if set(got) != {r["nomer"] for r in expected}:
                raise SystemExit(f"{name}: номера ответов не совпадают с партией")
            by_nomer.update(got)
    return {(r["query_id"], r["chunk_id"]): by_nomer[r["nomer"]]
            for r in read_tsv(os.path.join(WORK, "pairs.tsv")) if r["nomer"] in by_nomer}


def grade(first: str, second: str) -> int:
    return 2 if first == "да" else (1 if second == "да" else 0)


def model_labels() -> dict[tuple[str, str], tuple[str, str, str]]:
    """Ответы разметчика по всему ядру: партии плюс пары контрольной выборки
    из первого прогона пробы (та же сборка, повторно не размечались)."""
    core = {(r["query_id"], r["chunk_id"]) for r in core_pairs()}
    out = collect()
    for r in read_tsv(PROBA):
        key = (r["query_id"], r["chunk_id"])
        if key in core:
            out[key] = (r["otvet1_celikom"], r["otvet2_vmeste"], r["tip"])
    missing = core - set(out)
    if missing:
        raise SystemExit(f"в ядре без ответа {len(missing)} пар, например {sorted(missing)[:3]}")
    return out


def bias() -> dict:
    """Смещение разметчика по контрольной выборке: сколько позитивов человека
    он пропускает и сколько добавляет там, где человек сказал «нет»."""
    from kappa import read_marks

    model, person = read_marks(PROBA), read_marks(os.path.join(QDIR, "human_sample_new.tsv"))
    keys = sorted(set(model) & set(person))
    pos = [k for k in keys if person[k] > 0]
    neg = [k for k in keys if person[k] == 0]
    return {"пар": len(keys),
            "пропущено позитивов человека": {"числитель": sum(1 for k in pos if model[k] == 0),
                                             "знаменатель": len(pos)},
            "добавлено позитивов": {"числитель": sum(1 for k in neg if model[k] > 0), "знаменатель": len(neg)},
            "позитивов у модели": sum(1 for k in keys if model[k] > 0),
            "позитивов у человека": len(pos)}


def apply(args) -> None:
    import numpy as np

    from rufin import metrics as M
    from rufin.benchmark import read_qrels, write_qrels

    labels = model_labels()
    pool = {(r["query_id"], r["chunk_id"]): r["istochnik"] for r in core_pairs()}
    with open(SHEET, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["query_id", "chunk_id", "otvet1_celikom", "otvet2_vmeste", "tip", "istochnik_v_pule"])
        for (q, c), (a, b, t) in sorted(labels.items()):
            w.writerow([q, c, a, b, t, pool[(q, c)]])

    old = read_qrels(os.path.join(QDIR, "qrels.tsv"))
    subsets = json.load(open(os.path.join(QDIR, "podvyborki.json"), encoding="utf-8"))
    live = subsets[LIVE]
    union: dict[str, dict[str, int]] = {q: {c: s for c, s in old[q].items() if s > 0} for q in live}
    model_only: dict[str, dict[str, int]] = {q: {c: s for c, s in old[q].items()
                                                 if s > 0 and (q, c) not in labels} for q in live}
    for (q, c), (a, b, _) in labels.items():
        g = grade(a, b)
        if g > 0:
            union[q][c] = max(union[q].get(c, 0), g)
            model_only[q][c] = g
    # Разметка только дополняется: прежний позитив, который разметчик счел
    # нерелевантным, в основном варианте остается. Вариант «только модель»
    # печатается рядом как проверка, насколько вывод от этого зависит.
    write_qrels(CORE_QRELS, union)
    write_qrels(MODEL_ONLY_QRELS, model_only)

    agree = collections.defaultdict(collections.Counter)
    for (q, c), (a, b, _) in labels.items():
        agree[pool[(q, c)]][f"прежде {'да' if old.get(q, {}).get(c, 0) > 0 else 'нет'}, "
                            f"модель {'да' if grade(a, b) > 0 else 'нет'}"] += 1

    def density(qrels: dict[str, dict[str, int]], ids: list[str]) -> dict:
        n = [sum(1 for s in qrels.get(q, {}).values() if s > 0) for q in ids]
        return {"запросов": len(ids), "меток": sum(n), "среднее": round(sum(n) / len(n), 2),
                "медиана": float(np.median(n))}

    runs = {}
    for name in sorted(os.listdir(os.path.join(ROOT, "data", "runs"))):
        if name.startswith("base__") and name.endswith(".jsonl"):
            with open(os.path.join(ROOT, "data", "runs", name), encoding="utf-8") as f:
                runs[name[6:-6]] = {r["query_id"]: r["ranked"] for r in map(json.loads, f)}
    table = []
    for sub in LIVE_SUBSETS:
        ids = [q for q in subsets.get(sub, []) if q in union]
        for cfg, run in runs.items():
            qids = [q for q in ids if q in run]
            if not qids:
                continue
            r = {q: run[q] for q in qids}
            before = M.per_query(r, {q: old[q] for q in qids})
            after = M.per_query(r, {q: union[q] for q in qids})
            only = M.per_query(r, {q: model_only[q] for q in qids if model_only[q]})
            diff, p = M.paired_diff_ci(after["NDCG@10"], before["NDCG@10"])
            table.append({"подвыборка": sub, "связка": cfg, "запросов": len(qids),
                          "NDCG@10 прежде": round(float(np.nanmean(before["NDCG@10"])), 4),
                          "NDCG@10 плотнее": round(float(np.nanmean(after["NDCG@10"])), 4),
                          "NDCG@10 только модель": round(float(np.nanmean(only["NDCG@10"])), 4),
                          "Recall@10 прежде": round(float(np.nanmean(before["Recall@10"])), 4),
                          "Recall@10 плотнее": round(float(np.nanmean(after["Recall@10"])), 4),
                          "разница": {"среднее": round(diff.mean, 4), "lo": round(diff.lo, 4),
                                      "hi": round(diff.hi, 4), "p": p}})

    def rank(sub: str, key: str) -> list[str]:
        rows = [t for t in table if t["подвыборка"] == sub]
        return [t["связка"] for t in sorted(rows, key=lambda t: -t[key])]

    usage = json.load(open(args.usage, encoding="utf-8")) if args.usage else None
    grades = collections.Counter(grade(a, b) for a, b, _ in labels.values())
    result = {"пар ядра": len(labels), "из них размечено партиями": len(collect()),
              "оценки модели": {str(k): grades[k] for k in sorted(grades)},
              "согласие с прежними метками по источнику пары": {k: dict(v) for k, v in sorted(agree.items())},
              "плотность живого теста": {"прежде": density(old, live), "плотнее": density(union, live),
                                         "только модель": density(model_only, live)},
              "смещение разметчика по контрольной выборке": bias(),
              "таблица": table,
              "порядок связок на живых": {"прежде": rank(LIVE, "NDCG@10 прежде"),
                                          "плотнее": rank(LIVE, "NDCG@10 плотнее")},
              "токены": usage}
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    write_block(result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("таблица", "токены")},
                     ensure_ascii=False, indent=1))
    for t in table:
        if t["подвыборка"] == LIVE and t["связка"] in ("bm25", "dense-user-bge-m3", "hybrid", "hybrid-rerank"):
            print(t["связка"], t["NDCG@10 прежде"], "->", t["NDCG@10 плотнее"], "| только модель",
                  t["NDCG@10 только модель"], t["разница"])


DOC = os.path.join(ROOT, "docs", "18-nabor-do-i-posle.md")
BLOCK_START = "<!-- tablica: razmetka_yadro.py -->"
BLOCK_END = "<!-- /tablica yadro -->"
DOC_RUNS = ("bm25", "dense-user-bge-m3", "hybrid", "hybrid-rerank")


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def signed(x: float) -> str:
    return ("+" if x >= 0 else "−") + num(abs(x))


def big(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def doc_block(result: dict) -> str:
    d = result["плотность живого теста"]
    out = [f"Пар в ядре {big(result['пар ядра'])}, оценки 0, 1, 2: "
           f"{', '.join(big(result['оценки модели'][k]) for k in ('0', '1', '2'))}. Меток на живой "
           f"запрос было {num(d['прежде']['среднее'], 2)} ({big(d['прежде']['меток'])} на "
           f"{d['прежде']['запросов']}), стало {num(d['плотнее']['среднее'], 2)} "
           f"({big(d['плотнее']['меток'])}), в варианте «только новые метки внутри ядра» "
           f"{num(d['только модель']['среднее'], 2)} ({big(d['только модель']['меток'])}).", "",
           "| Связка | Подвыборка | Запросов | NDCG@10 прежде | плотнее | разница [95%] | только новые |",
           "|---|---|---|---|---|---|---|"]
    for t in result["таблица"]:
        if t["связка"] in DOC_RUNS and t["подвыборка"] in ("живые", "живые без писем"):
            r = t["разница"]
            out.append(f"| {t['связка']} | {t['подвыборка']} | {t['запросов']} | {num(t['NDCG@10 прежде'])} | "
                       f"{num(t['NDCG@10 плотнее'])} | {signed(r['среднее'])} [{signed(r['lo'])}; "
                       f"{signed(r['hi'])}] | {num(t['NDCG@10 только модель'])} |")
    out += ["", "| Пары ядра по прежней метке | Разметчик сказал «да» | Разметчик сказал «нет» |",
            "|---|---|---|"]
    names = {"-": "не размечены", "пункт": "пункт назван Банком России", "судья": "прежний судья",
             "эталон": "эталон"}
    for src, c in result["согласие с прежними метками по источнику пары"].items():
        yes = sum(v for k, v in c.items() if k.endswith("модель да"))
        no = sum(v for k, v in c.items() if k.endswith("модель нет"))
        old_yes = sum(v for k, v in c.items() if k.startswith("прежде да"))
        label = names.get(src, src) + (f", прежде позитив {old_yes}" if old_yes else "")
        out.append(f"| {label} | {yes} | {no} |")
    b = result["смещение разметчика по контрольной выборке"]
    t = result["токены"]
    out += ["", f"Смещение разметчика по контрольной выборке ({b['пар']} пар): из "
                f"{b['пропущено позитивов человека']['знаменатель']} позитивов человека пропущено "
                f"{b['пропущено позитивов человека']['числитель']}, из "
                f"{b['добавлено позитивов']['знаменатель']} пар, где человек сказал «нет», позитив "
                f"поставлен в {b['добавлено позитивов']['числитель']}. Токены сняты с журналов вызовов "
                f"всех сессий разметчика (их {t['партий']}): обработано {big(t['всего обработано'])}, из них новых входных "
                f"{big(t['новые входные'])}, чтения из кэша {big(t['чтение из кэша'])}, выходных "
                f"{big(t['выходные'])}."]
    return "\n".join(out)


def write_block(result: dict) -> None:
    body = open(DOC, encoding="utf-8").read()
    pattern = re.compile(re.escape(BLOCK_START) + ".*?" + re.escape(BLOCK_END), re.S)
    if not pattern.search(body):
        raise SystemExit(f"в {DOC} нет меток {BLOCK_START} ... {BLOCK_END}")
    with open(DOC, "w", encoding="utf-8") as f:
        f.write(pattern.sub(lambda _: f"{BLOCK_START}\n{doc_block(result)}\n{BLOCK_END}", body))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("status")
    a = sub.add_parser("apply")
    a.add_argument("--usage", help="json с токенами, снятыми с журналов вызовов разметчиков")
    args = ap.parse_args()
    {"prepare": prepare, "status": status, "apply": apply}[args.cmd](args)


if __name__ == "__main__":
    main()
