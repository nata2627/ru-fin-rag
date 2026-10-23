#!/usr/bin/env python3
"""Разбор запросов, на которых поиск провалился.

Берутся запросы, где ни один размеченный фрагмент не попал в первую десятку,
и раскладываются по причинам. Признаки считаются автоматически, но причина
предлагается, а не назначается: скрипт готовит рабочий лист и читаемую
выгрузку, а причина после разбора лежит в отдельном файле (`--razbor`),
который скрипт только читает. Разделение нужно затем, что повторный запуск
не должен затирать разбор.

Признаки, предлагающие причину:
  синонимия          слова запроса и ответа не пересекаются;
  длинный документ   нужный акт найден, но выдан не тот его кусок;
  нужно несколько    ответ собирается из нескольких фрагментов;
  нарезка            ответ разорван границей чанка;
  нет в корпусе      размеченного ответа в корпусе фактически нет;
  почти нашли        ответ есть, но ниже десятого места.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import references  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.retrieval.text import tokenize  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNDIR = os.path.join(ROOT, "data", "runs")
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
OUT = os.path.join(ROOT, "docs", "raw")
RESULTDIR = os.path.join(ROOT, "data", "results")

# доля слов запроса, встречающихся в эталонном фрагменте, ниже которой
# промах объясняется расхождением терминологии
LOW_OVERLAP = 0.25
# длина акта, начиная с которой он считается длинным
LONG_ACT_CHARS = 60_000

_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch",
    "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    " ": "-", "-": "-",
})


def slug(name: str) -> str:
    return "".join(ch for ch in name.lower().translate(_TRANSLIT) if ch.isalnum() or ch == "-")


def clause_numbers(text: str) -> set[str]:
    """Номера пунктов, названные в вопросе («пп. 3.12.2.8», «п. 3.10»)."""
    out = set()
    for m in references.CLAUSE.finditer(text):
        out |= set(re.findall(r"\d+(?:\.\d+)+", m.group(2)))
    return out


def clause_chunks(chunks: dict[str, dict], act_id: str, clause: str) -> list[str]:
    """Фрагменты акта, в которых пункт начинается, а не только упомянут.

    Условие «номер стоит в начале строки или после пробела и за ним точка
    с пробелом» отсекает ссылки вида «в соответствии с пунктом 3.12.1»:
    такая ссылка встречается в десятках фрагментов и адресом не является.
    """
    pat = re.compile(r"(?:^|\n|\s)" + re.escape(clause) + r"\.\s")
    return [cid for cid, c in chunks.items()
            if c["act_id"] == act_id and pat.search(c["text"])]


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="файл выдачи из data/runs")
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--queries", default=os.path.join(QDIR, "queries.jsonl"))
    ap.add_argument("--qrels", default=os.path.join(QDIR, "qrels.tsv"))
    ap.add_argument("--subsets", default=os.path.join(QDIR, "podvyborki.json"))
    ap.add_argument("--subset", default=None, help="имя подвыборки из podvyborki.json")
    ap.add_argument("--deep-run", default=None,
                    help="выдача той же конфигурации глубже: расширяет «почти нашли»")
    ap.add_argument("--take", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None,
                    help="отбирать случайно с этим семенем; без него берутся первые по query_id")
    ap.add_argument("--razbor", default=None,
                    help="TSV с причинами после разбора: query_id, причина, помогла_бы, заметка")
    args = ap.parse_args()

    chunks = {c["chunk_id"]: c for c in
              (json.loads(l) for l in open(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"),
                                           encoding="utf-8"))}
    act_chars: dict[str, int] = collections.Counter()
    act_chunks: dict[str, int] = collections.Counter()
    for c in chunks.values():
        act_chars[c["act_id"]] += len(c["text"])
        act_chunks[c["act_id"]] += 1

    queries = {q["query_id"]: q for q in
               (json.loads(l) for l in open(args.queries, encoding="utf-8"))}
    qrels = read_qrels(args.qrels)
    run = {r["query_id"]: r["ranked"] for r in
           (json.loads(l) for l in open(args.run, encoding="utf-8"))}
    deep = {}
    if args.deep_run and os.path.exists(args.deep_run):
        deep = {r["query_id"]: r["ranked"] for r in
                (json.loads(l) for l in open(args.deep_run, encoding="utf-8"))}

    scope = set(run)
    if args.subset:
        subsets = json.load(open(args.subsets, encoding="utf-8"))
        scope &= set(subsets[args.subset])

    failures = []
    for qid in sorted(scope):
        ranked = run[qid]
        rel = {c for c, g in qrels.get(qid, {}).items() if g > 0}
        if not rel or rel & set(ranked[:10]):
            continue
        q = queries[qid]
        gold_ids = sorted(rel)
        gold = chunks.get(gold_ids[0])
        gold_act = gold["act_id"] if gold else ""
        found_acts = {chunks[c]["act_id"] for c in ranked[:10] if c in chunks}

        qt = set(tokenize(q["text"]))
        gt = set(tokenize(gold["text"])) if gold else set()
        overlap = len(qt & gt) / len(qt) if qt else 0.0
        # позиция эталона в самой выдаче (глубже десятки) и в более глубокой, если дана
        own_pos = next((i for i, c in enumerate(ranked, start=1) if c in rel), None)
        deep_pos = None
        if qid in deep:
            deep_pos = next((i for i, c in enumerate(deep[qid], start=1) if c in rel), None)
        pos = own_pos or deep_pos

        numbers = references.act_numbers(q["text"])
        gold_number = gold["number"] if gold else ""

        # Пункт, названный в вопросе, это адрес ответа. Если фрагмент с этим
        # пунктом в первой десятке есть, а в эталоне его нет, провал
        # принадлежит метрике, а не поиску, и проверяется это кодом.
        named: list[str] = []
        named_unlabelled = False
        for clause in sorted(clause_numbers(q["text"]), key=len, reverse=True):
            for cid in clause_chunks(chunks, gold_act, clause):
                if cid in ranked[:10]:
                    named.append(f"{clause}@{ranked.index(cid) + 1}")
                    named_unlabelled = named_unlabelled or cid not in rel

        if gold is None:
            cause = "нет в корпусе"
        elif pos:
            cause = "почти нашли"
        elif gold_act in found_acts:
            cause = "длинный документ" if act_chars[gold_act] > LONG_ACT_CHARS else "нарезка"
        elif len(rel) > 1:
            cause = "нужно несколько"
        elif overlap < LOW_OVERLAP:
            cause = "синонимия"
        else:
            cause = "разобрать руками"

        failures.append({
            "query_id": qid,
            "origin": q.get("origin", ""),
            "query": q["text"],
            "predpolozhennaya_prichina": cause,
            "нужный_акт_в_выдаче": "да" if gold_act in found_acts else "нет",
            "позиция_эталона": pos or "",
            "пересечение_слов": round(overlap, 2),
            "размеченных_фрагментов": len(rel),
            "акт": f"{gold['number']} от {gold['date']}" if gold else "",
            "номер_в_вопросе": ",".join(numbers),
            "номер_совпал_с_эталоном": ("да" if any(n.upper() == gold_number.upper()
                                                    for n in numbers)
                                        else ("нет" if numbers else "не назван")),
            "названный_пункт_в_топ10": ",".join(named),
            "названный_пункт_не_размечен": "да" if named_unlabelled else "",
            "размер_акта_симв": act_chars.get(gold_act, 0),
            "чанков_в_акте": act_chunks.get(gold_act, 0),
            "эталон_начало": (gold["text"].split("\n\n", 1)[-1][:180].replace("\n", " ")
                              if gold else ""),
            "найдено_первым": (chunks[ranked[0]]["text"].split("\n\n", 1)[-1][:180].replace("\n", " ")
                               if ranked and ranked[0] in chunks else ""),
            "_gold_ids": gold_ids,
            "_ranked": ranked,
        })

    if args.seed is None:
        take = failures[:args.take]
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(failures), size=min(args.take, len(failures)), replace=False)
        take = [failures[i] for i in sorted(idx)]

    os.makedirs(OUT, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.run))[0]
    if args.subset:
        base = f"{base}_{slug(args.subset)}"
    fields = [k for k in (take[0].keys() if take else ["query_id"]) if not k.startswith("_")]
    tsv = os.path.join(OUT, f"errors_{base}.tsv")
    with open(tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(take)

    # читаемая выгрузка: по ней разбирают провал глазами, и по ней же
    # проверяют чужой разбор, не запуская поиск заново
    dump = os.path.join(OUT, f"errors_{base}.txt")
    with open(dump, "w", encoding="utf-8") as f:
        for n, fl in enumerate(take, start=1):
            f.write(f"=== {n}. {fl['query_id']} ===\n")
            f.write(f"вопрос: {fl['query']}\n")
            f.write(f"номер в вопросе: {fl['номер_в_вопросе'] or 'нет'}   "
                    f"пересечение слов: {fl['пересечение_слов']}   "
                    f"позиция эталона: {fl['позиция_эталона'] or 'вне выдачи'}\n")
            for cid in fl["_gold_ids"]:
                c = chunks.get(cid)
                body = c["text"].split("\n\n", 1)[-1] if c else "(фрагмента нет в корпусе)"
                f.write(f"-- эталон {cid}\n   {body[:900]}\n")
            f.write("-- первая десятка выдачи\n")
            for i, cid in enumerate(fl["_ranked"][:10], start=1):
                c = chunks.get(cid)
                body = c["text"].split("\n\n", 1)[-1].replace("\n", " ") if c else "?"
                head = f"{c['number']} {c['section'][:40]}" if c else cid
                f.write(f"   {i:>2}. {cid}  {head}\n       {body[:220]}\n")
            f.write("\n")

    # маршрут разметки у провала: эталон по названным пунктам или из пула
    # с судьей. Разница важна для чтения метрики, потому что пул собран из
    # первых десяток самих конфигураций, и его эталон почти всегда кем-то
    # найден: провал на таком вопросе возможен, но структурно реже.
    def route(path: str) -> set[str]:
        if not os.path.exists(path):
            return set()
        with open(path, encoding="utf-8") as f:
            return {line.split("\t")[0] for line in f if not line.startswith("query-id")}

    routes = {"по названным пунктам": route(os.path.join(QDIR, "qrels_by_clause.tsv")),
              "пулом с судьей": route(os.path.join(QDIR, "qrels_judge.tsv"))}
    failed_ids = {f["query_id"] for f in failures}

    report = {
        "выдача": os.path.relpath(args.run, ROOT),
        "md5 выдачи": md5(args.run),
        "md5 эталона": md5(args.qrels),
        "md5 вопросов": md5(args.queries),
        "подвыборка": args.subset or "вся выдача",
        "запросов в подвыборке": len(scope),
        "провалов": len(failures),
        "разобрано": len(take),
        "семя отбора": args.seed,
        "провалов по маршруту разметки": {
            name: {"запросов": len(scope & ids), "провалов": len(failed_ids & ids)}
            for name, ids in routes.items() if ids},
        "акты эталона у провалов": dict(collections.Counter(
            f["акт"] for f in failures).most_common()),
        "названный в вопросе пункт найден в первой десятке и не размечен": {
            "в разборе": sum(1 for f in take if f["названный_пункт_не_размечен"]),
            "во всех провалах": sum(1 for f in failures if f["названный_пункт_не_размечен"]),
        },
        "предположенные причины": dict(
            collections.Counter(f["predpolozhennaya_prichina"] for f in take).most_common()),
    }

    razbor = {}
    if args.razbor and os.path.exists(args.razbor):
        with open(args.razbor, encoding="utf-8") as f:
            razbor = {r["query_id"]: r for r in csv.DictReader(f, delimiter="\t")}
        picked = {f["query_id"] for f in take}
        missing = picked - set(razbor)
        extra = set(razbor) - picked
        if missing or extra:
            raise SystemExit(f"разбор не сходится с отбором: нет разбора у {sorted(missing)}, "
                             f"лишние {sorted(extra)}")
        report["разбор"] = {
            "файл": os.path.relpath(args.razbor, ROOT),
            "кем сделан": "разобрано моделью, человек не проверял",
            "причины": dict(collections.Counter(
                razbor[q]["причина"] for q in sorted(picked)).most_common()),
            "помогла бы человеческая манера вопроса": dict(collections.Counter(
                razbor[q]["помогла_бы"] for q in sorted(picked)).most_common()),
        }

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    os.makedirs(RESULTDIR, exist_ok=True)
    result = os.path.join(RESULTDIR, f"errors-{slug(args.subset) if args.subset else 'all'}.json")
    with open(result, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    say(f"запросов в разборе: {len(scope)}")
    say(f"провалов (ни один размеченный фрагмент не попал в топ-10): {len(failures)} "
        f"из {len(scope)} = {100 * len(failures) / max(1, len(scope)):.0f}%")
    say(f"разобрано: {len(take)}, семя отбора: {args.seed}")
    say()
    for name, counts in report["провалов по маршруту разметки"].items():
        say(f"маршрут разметки «{name}»: провалов {counts['провалов']} "
            f"из {counts['запросов']}")
    for act, n in report["акты эталона у провалов"].items():
        say(f"акт эталона у провалов: {act}, {n} из {len(failures)}")
    named = report["названный в вопросе пункт найден в первой десятке и не размечен"]
    say(f"названный в вопросе пункт найден в первой десятке и не размечен: "
        f"{named['в разборе']} из {len(take)} разобранных, "
        f"{named['во всех провалах']} из {len(failures)} провалов")
    say()
    say("предположенные причины:")
    for cause, n in report["предположенные причины"].items():
        say(f"   {cause:<20} {n:>3}")
    if razbor:
        say("\nпричины после разбора (разобрано моделью, человек не проверял):")
        for cause, n in report["разбор"]["причины"].items():
            say(f"   {cause:<28} {n:>3} из {len(take)}")
        say("\nпомогла бы человеческая манера вопроса:")
        for group, n in report["разбор"]["помогла бы человеческая манера вопроса"].items():
            say(f"   {group:<28} {n:>3} из {len(take)}")
    say(f"\nфайлы: {os.path.relpath(tsv, ROOT)}, {os.path.relpath(dump, ROOT)}, "
        f"{os.path.relpath(result, ROOT)}")

    # data/results в этом репозитории не версионируется (reproducible),
    # поэтому сводка числами ложится рядом с выгрузками в docs/raw
    svod = os.path.join(OUT, f"errors_{base}_svod.txt")
    with open(svod, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"сводка: {os.path.relpath(svod, ROOT)}")


if __name__ == "__main__":
    main()
