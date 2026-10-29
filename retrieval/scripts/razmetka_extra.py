#!/usr/bin/env python3
"""Второй набор живых: разметка ядра пула по 183 вопросам, которые не размечены вовсе.

Эти вопросы в тест не идут, это запас под будущее обучение. Поэтому здесь нет
сравнения связок: печатается, сколько позитивов нашлось, про какие акты эти
вопросы и насколько они пересекаются с тестом, и сколько их позитивов лежит в
эталоне теста (утечка, если набор пойдет в обучение).

Сборка разметчика та же, что в пробе и в ядре теста (`razmetka_yadro.py`),
партии готовит его же `prepare` в своей папке:

    python3 scripts/pool_completeness.py --unlabeled --out data/queries/pool_live_extra.tsv
    python3 scripts/razmetka_yadro.py prepare --pool data/queries/pool_live_extra.tsv --work data/interim/razmetka_extra
    python3 scripts/razmetka_extra.py --usage <usage.json>
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import razmetka_yadro as Y  # noqa: E402

from rufin import punkt as P  # noqa: E402
from rufin.benchmark import read_qrels, write_qrels  # noqa: E402

ROOT = Y.ROOT
QDIR = Y.QDIR
WORK = os.path.join(ROOT, "data", "interim", "razmetka_extra")
POOL = os.path.join(QDIR, "pool_live_extra.tsv")
SHEET = os.path.join(QDIR, "razmetka_extra.tsv")
QRELS = os.path.join(QDIR, "qrels_live_extra.tsv")
RESULT = os.path.join(ROOT, "data", "results", "pool-labeling-extra.json")
DOC = Y.DOC
BLOCK_START = "<!-- tablica: razmetka_extra.py -->"
BLOCK_END = "<!-- /tablica extra -->"
DOMINANT_ACT = "590-П"


def act_of(chunk_id: str) -> str:
    return chunk_id.split("_", 1)[0]


def test_gold() -> set[str]:
    """Эталон теста по фрагментам, как в `drop_leak.py`, плюс все позитивы qrels.tsv."""
    gold = {c for us in P.read_units(os.path.join(QDIR, "qrels_punkt.tsv")).values()
            for u in us.values() if u.score > 0 for c in u.chunks}
    for name in ("qrels.tsv", "qrels_live_core.tsv"):
        gold |= {c for rel in read_qrels(os.path.join(QDIR, name)).values() for c, s in rel.items() if s > 0}
    return gold


def share(part: int, whole: int) -> dict:
    return {"числитель": part, "знаменатель": whole}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", help="json с токенами, снятыми с журналов вызовов разметчиков")
    args = ap.parse_args()

    Y.WORK = WORK
    labels = Y.collect()
    pool = {(r["query_id"], r["chunk_id"]): r["istochnik"] for r in Y.core_pairs(POOL)}
    missing = set(pool) - set(labels)
    if missing:
        raise SystemExit(f"в ядре без ответа {len(missing)} пар, например {sorted(missing)[:3]}")
    with open(SHEET, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["query_id", "chunk_id", "otvet1_celikom", "otvet2_vmeste", "tip", "istochnik_v_pule"])
        for (q, c), (a, b, t) in sorted(labels.items()):
            w.writerow([q, c, a, b, t, pool[(q, c)]])

    queries = {}
    with open(os.path.join(QDIR, "queries.jsonl"), encoding="utf-8") as f:
        for q in map(json.loads, f):
            queries[q["query_id"]] = q
    extra = sorted({q for q, _ in pool})
    qrels: dict[str, dict[str, int]] = {q: {} for q in extra}
    for (q, c), (a, b, _) in labels.items():
        if Y.grade(a, b) > 0:
            qrels[q][c] = Y.grade(a, b)
    # Метки только разметчика: у этих вопросов прежней разметки нет, а старые
    # оценки судьи сравниваются ниже и в файл меток не идут.
    write_qrels(QRELS, {q: r for q, r in qrels.items() if r})

    grades = collections.Counter(Y.grade(a, b) for a, b, _ in labels.values())
    per_query = [len(qrels[q]) for q in extra]

    subsets = json.load(open(os.path.join(QDIR, "podvyborki.json"), encoding="utf-8"))
    test_ids = {q for name, ids in subsets.items() for q in ids}
    leaked_queries = sorted(test_ids & set(extra))
    if leaked_queries:
        raise SystemExit(f"вопросы второго набора лежат в тесте: {leaked_queries[:5]}")
    test_texts = {queries[q]["text"].strip() for q in test_ids}
    same_text = [q for q in extra if queries[q]["text"].strip() in test_texts]

    live_test = subsets[Y.LIVE]
    acts_extra = collections.Counter(queries[q].get("act") or "-" for q in extra)
    acts_test = collections.Counter(queries[q].get("act") or "-" for q in live_test)
    shared_acts = sorted(a for a in acts_extra if a != "-" and a in acts_test)

    test_by_act = {act_of(c) for rel in read_qrels(os.path.join(QDIR, "qrels.tsv")).values()
                   for c, s in rel.items() if s > 0}
    positives = [(q, c) for q, rel in qrels.items() for c in rel]
    gold = test_gold()
    leak_pairs = [(q, c) for q, c in positives if c in gold]

    judge = {}
    with open(os.path.join(QDIR, "judge.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            judge[(r["query_id"], r["chunk_id"])] = int(r["otsenka_0_1_2"])
    judged = [k for k in labels if k in judge]
    vs_judge = collections.Counter(
        f"судья {'да' if judge[k] > 0 else 'нет'}, модель {'да' if Y.grade(*labels[k][:2]) > 0 else 'нет'}"
        for k in judged)
    judged_pool = sum(1 for r in Y.read_tsv(POOL) if r["istochnik"] == "судья")

    usage = json.load(open(args.usage, encoding="utf-8")) if args.usage else None
    result = {"вопросов": len(extra), "пар ядра": len(labels),
              "оценки модели": {str(k): grades[k] for k in sorted(grades)},
              "позитивов": len(positives),
              "позитивов на вопрос": {"среднее": round(sum(per_query) / len(per_query), 2),
                                      "без позитива": share(sum(1 for n in per_query if n == 0), len(extra))},
              "акты вопросов": dict(acts_extra.most_common()),
              "акты живого теста": dict(acts_test.most_common()),
              "вопросы про акт теста": share(sum(n for a, n in acts_extra.items() if a in shared_acts),
                                             len(extra)),
              f"вопросы про {DOMINANT_ACT}": share(acts_extra[DOMINANT_ACT], len(extra)),
              "позитивы в актах эталона теста": share(sum(1 for _, c in positives if act_of(c) in test_by_act),
                                                      len(positives)),
              "утечка": {"вопросов второго набора в тесте": 0, "совпадений текста с тестом": len(same_text),
                         "позитивов в эталоне теста": share(len(leak_pairs), len(positives)),
                         "вопросов с такими позитивами": share(len({q for q, _ in leak_pairs}), len(extra)),
                         "фрагментов эталона теста": len(gold)},
              "сравнение с прежним судьей": {"пар судьи в пуле": judged_pool, "из них в ядре": len(judged),
                                             **dict(sorted(vs_judge.items()))},
              "токены": usage}
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    write_block(result)
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("акты")},
                     ensure_ascii=False, indent=1))


def doc_block(r: dict) -> str:
    big, num = Y.big, Y.num
    g, d, u, j, t = r["оценки модели"], r["позитивов на вопрос"], r["утечка"], r["сравнение с прежним судьей"], \
        r["токены"]
    top = ", ".join(f"{a} {n}" for a, n in list(r["акты вопросов"].items())[:5])

    def frac(x: dict) -> str:
        return f"{big(x['числитель'])} из {big(x['знаменатель'])}"

    agree = j.get("судья да, модель да", 0) + j.get("судья нет, модель нет", 0)
    out = [f"Вопросов {r['вопросов']}, пар в ядре {big(r['пар ядра'])}, оценки 0, 1, 2: "
           f"{', '.join(big(g.get(k, 0)) for k in ('0', '1', '2'))}. Позитивов {big(r['позитивов'])}, "
           f"в среднем {num(d['среднее'], 2)} на вопрос, без единого позитива {frac(d['без позитива'])} "
           f"вопросов.", "",
           f"Состав по актам (акт назван в самом вопросе, у кого не назван, стоит «-»): {top}. Про "
           f"{DOMINANT_ACT} {frac(r[f'вопросы про {DOMINANT_ACT}'])} вопросов, про акты, которые есть и "
           f"в живом тесте, {frac(r['вопросы про акт теста'])}. Позитивы лежат в актах эталона теста в "
           f"{frac(r['позитивы в актах эталона теста'])} случаев.", "",
           f"Утечка. Вопросов второго набора в тесте нет, совпадений текста с вопросами теста "
           f"{u['совпадений текста с тестом']}. Позитивов, которые лежат в эталоне теста "
           f"({big(u['фрагментов эталона теста'])} фрагментов), {frac(u['позитивов в эталоне теста'])}, они "
           f"у {frac(u['вопросов с такими позитивами'])} вопросов. Если набор пойдет в обучение, эти "
           f"пары убираются так же, как в задаче 11.2.", "",
           f"Прежний судья оценил {big(j['пар судьи в пуле'])} пар этого пула, в ядро попали "
           f"{big(j['из них в ядре'])}, на них разметчик согласен с судьей в {big(agree)}: судья «да», "
           f"модель «да» {big(j.get('судья да, модель да', 0))}, судья «да», модель «нет» "
           f"{big(j.get('судья да, модель нет', 0))}, судья «нет», модель «да» {big(j.get('судья нет, модель да', 0))}, "
           f"оба «нет» {big(j.get('судья нет, модель нет', 0))}."]
    if t:
        out += ["", f"Токены сняты с журналов вызовов всех сессий разметчика (их {t['партий']}): обработано "
                    f"{big(t['всего обработано'])}, из них новых входных {big(t['новые входные'])}, чтения из "
                    f"кэша {big(t['чтение из кэша'])}, выходных {big(t['выходные'])}."]
    return "\n".join(out)


def write_block(result: dict) -> None:
    body = open(DOC, encoding="utf-8").read()
    pattern = re.compile(re.escape(BLOCK_START) + ".*?" + re.escape(BLOCK_END), re.S)
    if not pattern.search(body):
        raise SystemExit(f"в {DOC} нет меток {BLOCK_START} ... {BLOCK_END}")
    with open(DOC, "w", encoding="utf-8") as f:
        f.write(pattern.sub(lambda _: f"{BLOCK_START}\n{doc_block(result)}\n{BLOCK_END}", body))


if __name__ == "__main__":
    main()
