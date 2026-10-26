#!/usr/bin/env python3
"""Слабая метка на уровне акта: «этот вопрос про этот акт».

Зачем она нужна. Разметка по фрагментам у живых вопросов неполна: у связки
`hybrid-rerank` 7,64 позиции первой десятки из 10 не размечены никем
(`docs/11-polnota-razmetki.md`), и метрика читает неразмеченный фрагмент как
промах. Разметить пул нечем, пока судья не взял приёмку. А метка на уровне
акта лежит готовой и не стоит ничего: Банк России публикует вопросы
поднадзорных организаций разделами, и номер акта стоит в заголовке раздела,
написанном самим Банком России.

Чем она слабая и почему это написано здесь, а не в примечании. Метка говорит
«ответ на этот вопрос Банк России опубликовал под этим актом», а не
«ответ лежит в этом акте». Вопрос из раздела про 590-П может отвечаться
пунктом другого акта, и такой случай метка считает промахом. Поэтому она
лежит рядом с разметкой по фрагментам, а не вместо неё, и в одно число
с ней не складывается.

Что она даёт взамен. Охват: вопросов с меткой на уровне акта 301 против 142
с разметкой по фрагментам, и 162 из них не размечены по фрагментам вовсе.
И независимость: метка составлена до всякого прогона поиска и ни одной
связке не подыгрывает.

Источник метки назван точно: `data/queries/explan_live.jsonl`, поле `act`,
заполненное `rufin.explanations.act_of` из заголовка раздела `cbr.ru/explan`.
Разметку эту делали люди Банка России, а не модель, и писать «проверено
человеком» про неё можно: человек её и написал, просто не наш.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
RUNDIR = os.path.join(ROOT, "data", "runs")
RESULTS = os.path.join(ROOT, "data", "results")
REPORT = os.path.join(ROOT, "docs", "raw", "act_level_labels.txt")

CORE = ["bm25", "hybrid", "hybrid-rerank", "dense-user-bge-m3", "dense-rufin"]


def load_runs(config: str) -> dict[str, dict[str, list[str]]]:
    runs: dict[str, dict[str, list[str]]] = {}
    for name in sorted(os.listdir(RUNDIR)):
        m = re.fullmatch(rf"{re.escape(config)}__(.+)\.jsonl", name)
        if not m:
            continue
        run: dict[str, list[str]] = {}
        with open(os.path.join(RUNDIR, name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                run[rec["query_id"]] = rec["ranked"]
        runs[m.group(1)] = run
    return runs


def act_numbers(chunks_path: str) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Номер акта в его `act_id` и все `act_id` этого номера.

    Второе нужно затем, что номер не уникален: у методических рекомендаций
    один номер встречается с разными датами. Вопрос с таким номером из метки
    выбрасывается, а не привязывается к первому попавшемуся: угадывание
    редакции это выдумка провенанса.
    """
    first: dict[str, str] = {}
    all_ids: dict[str, set[str]] = collections.defaultdict(set)
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            first.setdefault(c["number"], c["act_id"])
            all_ids[c["number"]].add(c["act_id"])
    return first, all_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="base")
    ap.add_argument("--config", default="base")
    ap.add_argument("--live", default=os.path.join(QDIR, "explan_live.jsonl"))
    ap.add_argument("--out", default=os.path.join(QDIR, "qrels_act_weak.tsv"))
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    всего_собрано = sum(
        1 for _ in open(os.path.join(QDIR, "explan_questions.jsonl"),
                        encoding="utf-8"))
    записи = [json.loads(l) for l in open(args.live, encoding="utf-8") if l.strip()]
    годные = [r for r in записи if not r["route"].startswith("отброшен")]
    первый, все_id = act_numbers(os.path.join(CHUNKDIR, f"{args.chunks}.jsonl"))

    метки: dict[str, str] = {}
    неоднозначных = 0
    for r in годные:
        номер = r.get("act")
        if not номер or номер not in первый:
            continue
        if len(все_id[номер]) > 1:
            неоднозначных += 1
            continue
        метки[r["query_id"]] = первый[номер]

    qrels_q = set()
    with open(os.path.join(QDIR, "qrels.tsv"), encoding="utf-8") as f:
        for line in f:
            qid = line.split("\t")[0]
            if qid != "query-id":
                qrels_q.add(qid)

    say(f"собрано живых вопросов с cbr.ru/explan: {всего_собрано}")
    say(f"из них годных после отбора: {len(годные)}")
    say(f"с номером акта, который есть в корпусе: {len(метки)}")
    say(f"выброшено из-за неоднозначного номера (одно число, разные даты): "
        f"{неоднозначных}")
    say(f"актов различных: {len(set(метки.values()))}")
    say()
    say("охват: вопросов с меткой на уровне акта против вопросов с разметкой "
        "по фрагментам")
    say(f"   по фрагментам (qrels.tsv, подвыборка живых): {len(qrels_q & set(метки)) + 0} "
        f"из {len(метки)} общих")
    say(f"   только метка акта, по фрагментам не размечены: "
        f"{len([q for q in метки if q not in qrels_q])} из {len(метки)}")
    по_актам = collections.Counter(м for м in метки.values())
    say("   самые частые акты: " + ", ".join(
        f"{a.split('_')[0]} {n}" for a, n in по_актам.most_common(6)))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("query-id\tact-id\tscore\tosnovanie\n")
        for qid in sorted(метки):
            f.write(f"{qid}\t{метки[qid]}\t1\tакт назван Банком России "
                    f"в заголовке раздела cbr.ru/explan\n")
    say()
    say(f"файл метки: {os.path.relpath(args.out, ROOT)}")

    runs = load_runs(args.config)
    отсутствует = sorted(set(CORE) - set(runs))
    if отсутствует:
        raise SystemExit(f"в выдачах нет связок ядра: {', '.join(отсутствует)}")
    покрыты = sorted(q for q in метки
                     if all(q in run for run in runs.values()))
    say(f"вопросов с выдачей у всех {len(runs)} связок: {len(покрыты)} "
        f"из {len(метки)}")

    итог = {"собрано вопросов": всего_собрано, "годных": len(годные),
            "меток на уровне акта": len(метки),
            "выброшено неоднозначных номеров": неоднозначных,
            "актов": len(set(метки.values())),
            "размечены и по фрагментам": len(qrels_q & set(метки)),
            "только метка акта": len([q for q in метки if q not in qrels_q]),
            "с выдачей у всех связок": len(покрыты), "связки": {}}

    say()
    say("нашёл ли поиск нужный акт (доля вопросов, у которых акт метки есть "
        f"среди первых k позиций; знаменатель {len(покрыты)})")
    say(f"{'связка':<22}{'акт@1':>9}{'акт@10':>9}{'акт@50':>9}")
    for cfg in sorted(runs):
        run = runs[cfg]
        попал = collections.Counter()
        for q in покрыты:
            акты = [c.split("#")[0] for c in run[q][:50]]
            for k in (1, 10, 50):
                if метки[q] in акты[:k]:
                    попал[k] += 1
        n = len(покрыты)
        доли = {k: round(попал[k] / n, 3) for k in (1, 10, 50)}
        итог["связки"][cfg] = доли
        if cfg in CORE:
            say(f"{cfg:<22}{доли[1]:>9.3f}{доли[10]:>9.3f}{доли[50]:>9.3f}")
    say()
    say("метка слабая: она говорит «ответ опубликован под этим актом», а не "
        "«ответ лежит в этом акте». В одно число с разметкой по фрагментам "
        "не складывается.")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "act-level-labels.json"), "w",
              encoding="utf-8") as f:
        json.dump(итог, f, ensure_ascii=False, indent=2)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
