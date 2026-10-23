#!/usr/bin/env python3
"""Мелкая нарезка с выдачей родительского куска и цена шапки в тексте фрагмента.

Два вопроса, один стенд. Первый: помогает ли искать по мелким фрагментам,
а в выдачу отдавать родительский кусок базовой нарезки. Второй: сколько
стоит шапка с номером, названием акта и разделом, приписанная к тексту
каждого фрагмента.

Почему не годится готовый `chunking_effect.py`: он читает только изменение
разрыва между плотным поиском и BM25 и прямо запрещает сравнивать уровни,
потому что при дроблении цель уменьшается. Подъем к родителю это снимает:
в выдаче у всех ветвей фрагменты базовой нарезки, разметка одна и та же,
и уровни сравнимы напрямую (`rufin.parents`).

Ветви считаются здесь же одним кодом на одних и тех же запросах, сохраненные
выдачи этапа B не используются: разница между реализациями смешалась бы
с действием нарезки. Для сверки ветвь `base` сравнивается с сохраненной
выдачей, и доля совпадения печатается.

Абляция шапки идет против своей же локальной базовой нарезки, а не против
канонической: версия токенизатора локально (4.41.2) и на Kaggle (5.0.0)
разная, границы фрагментов от нее сдвигаются, и сравнение с канонической
мерило бы шапку вместе с версией библиотеки. Расхождение двух базовых
нарезок печатается отдельной строкой как контроль.

Видеокарта не нужна: BM25 считается на маке, подъем к родителю это арифметика
по символьным границам. Плотная ступень в этот опыт не входит, потому что
векторов мелкой нарезки на диске нет (TASK, «Открытые вопросы»).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin import metrics as M  # noqa: E402
from rufin.benchmark import read_qrels  # noqa: E402
from rufin.parents import ParentIndex  # noqa: E402
from rufin.retrieval.bm25 import BM25Index  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
RUNDIR = os.path.join(ROOT, "data", "runs")
CHUNKDIR = os.path.join(ROOT, "data", "chunks")
ACTS = os.path.join(ROOT, "data", "corpus", "acts.jsonl")
REPORT = os.path.join(ROOT, "docs", "raw", "chunk_parent_effect.txt")
RESULT = os.path.join(ROOT, "data", "results", "chunk-parent.json")

DEEP = 200          # до какой глубины мелкой выдачи читать, набирая родителей
SHALLOW = 50        # глубина ступени, на которой считается чувствительность
TOP = 10            # родителей в выдаче
METRICS = ("Recall@10", "MRR@10", "NDCG@10")

# Ветви: имя -> файл нарезки. Базовая каноническая идет первой и служит
# и единицей оценки, и точкой отсчета.
BRANCHES = {
    "base": os.path.join(CHUNKDIR, "base.jsonl"),
    "size-256": os.path.join(CHUNKDIR, "size-256.jsonl"),
    "base-local": os.path.join(CHUNKDIR, "local", "base.jsonl"),
    "no-heading": os.path.join(CHUNKDIR, "local", "no-heading.jsonl"),
    "fine-local": os.path.join(CHUNKDIR, "local", "size-256.jsonl"),
    "fine-no-heading": os.path.join(CHUNKDIR, "local", "size-256-no-heading.jsonl"),
    "no-section": os.path.join(CHUNKDIR, "local", "no-section.jsonl"),
}
# Что с чем сравнивается и что из этого читается. Локальные ветви сравниваются
# между собой: версия токенизатора локально и на Kaggle разная, границы
# фрагментов от нее сдвигаются, и смешивать сборки в одной паре нельзя.
PAIRS = (
    ("size-256", "base", "мелкая нарезка с подъемом к родителю, кагловская сборка"),
    ("fine-local", "base-local", "то же в локальной сборке"),
    ("no-heading", "base-local", "цена шапки при размере 512"),
    ("no-section", "base-local", "цена названия раздела в шапке при размере 512"),
    ("fine-no-heading", "fine-local", "цена шапки при размере 256"),
    ("fine-no-heading", "no-heading", "мелкая нарезка там, где шапки нет вовсе"),
    ("base-local", "base", "контроль: локальная сборка против кагловской"),
)
# Взаимодействие: размер и шапка не независимы, шапке отводится доля чанка.
# Разность разностей считается по тем же запросам, поэтому парным бутстрэпом,
# а не вычитанием двух средних.
INTERACTION = (("fine-no-heading", "fine-local"), ("no-heading", "base-local"))


def read_chunks(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def heading_share(rec: dict) -> float | None:
    """Какую долю символов фрагмента занимает шапка.

    Шапка опознается по своей дословной формуле «... Банка России № номер
    от дата», а не по первому пустому абзацу: у фрагмента без шапки первый
    абзац это уже текст акта, и доля получилась бы выдуманной. Одного номера
    тоже мало, акт называет сам себя в тексте: по номеру в первом абзаце
    «шапка» находилась у 2535 фрагментов нарезки, собранной без шапок вовсе.

    Нужно для механизма: шапке отводится доля чанка, и при 256 токенах она
    съедает вдвое больше места, чем при 512.
    """
    head = rec["text"].split("\n\n", 1)[0]
    if f"Банка России № {rec['number']} от {rec['date']}" not in head:
        return None
    return len(head) / len(rec["text"])


def body_of(rec: dict) -> str:
    """Текст фрагмента без шапки."""
    head, _, tail = rec["text"].partition("\n\n")
    formula = f"Банка России № {rec['number']} от {rec['date']}"
    return tail if formula in head else rec["text"]


def span_is_solid(rec: dict, act_text: str) -> bool:
    """Лежит ли текст фрагмента непрерывно внутри своего диапазона.

    Проверка нужна потому, что подъем к родителю считается по диапазону.
    У редких фрагментов он шире текста: перекрытие переносит пункт в
    следующий фрагмент через длинный пункт, который нарезан отдельно, и
    тогда между первым и последним пунктом фрагмента остается пропуск.
    Для таких фрагментов родитель может быть выбран по тексту, которого
    во фрагменте нет, и доля таких случаев должна быть напечатана, а не
    предполагаться малой.
    """
    body = body_of(rec).strip()
    if not body:
        return True
    return body[:60] in act_text[rec["char_start"]:rec["char_end"]]


def load_run(path: str) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        return {r["query_id"]: r["ranked"]
                for r in (json.loads(l) for l in f if l.strip())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="считать только названные ветви")
    ap.add_argument("--subsets", nargs="*", default=["заголовочная", "живые"])
    ap.add_argument("--span-stride", type=int, default=97,
                    help="шаг проверки символьных границ по тексту актов")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    branches = {k: v for k, v in BRANCHES.items()
                if not args.only or k in args.only}
    missing = [k for k, v in branches.items() if not os.path.exists(v)]
    if missing:
        raise SystemExit("нет нарезок: " + ", ".join(
            f"{k} ({os.path.relpath(branches[k], ROOT)})" for k in missing)
            + "\nЛокальные собираются командой "
              "`make chunks CONFIGS=\"base no-heading\" OUTDIR=data/chunks/local`")

    qrels = read_qrels(os.path.join(QDIR, "qrels.tsv"))
    queries = {q["query_id"]: q["text"]
               for q in read_chunks(os.path.join(QDIR, "queries.jsonl"))}
    qids = sorted(q for q in qrels if q in queries)
    subsets = json.load(open(os.path.join(QDIR, "podvyborki.json"), encoding="utf-8"))
    parts = {"все": qids}
    for name in args.subsets:
        parts[name] = sorted(q for q in subsets[name] if q in qrels and q in queries)

    say("мелкая нарезка с выдачей родительского куска и цена шапки")
    say(f"разметка: data/queries/qrels.tsv, запросов с эталоном и текстом {len(qids)}")
    for name, part in parts.items():
        if name != "все":
            say(f"  подвыборка «{name}»: {len(part)}")
    say(f"глубина чтения мелкой выдачи {DEEP}, родителей в выдаче {TOP}")
    say()

    t0 = time.time()
    parents = ParentIndex.from_records(read_chunks(BRANCHES["base"]))
    base_ids = {s.chunk_id for spans in parents.by_act.values() for s in spans}
    say(f"базовая нарезка: фрагментов {len(base_ids)}, актов {len(parents.by_act)}, "
        f"индекс родителей за {time.time() - t0:.0f} с")

    acts: dict[str, str] = {}
    if args.span_stride:
        acts = {a["act_id"]: a["text"] for a in read_chunks(ACTS)}
        say(f"тексты актов прочитаны: {len(acts)}")

    runs: dict[str, dict[str, list[str]]] = {}
    shallow: dict[str, dict[str, list[str]]] = {}
    report: dict[str, dict] = {}

    for name, path in branches.items():
        t0 = time.time()
        ids, texts, meta = [], [], {}
        outside = 0
        heads: list[float] = []
        covered: list[float] = []
        solid = checked = 0
        for rec in read_chunks(path):
            ids.append(rec["chunk_id"])
            texts.append(rec["text"])
            meta[rec["chunk_id"]] = (rec["act_id"], rec["char_start"], rec["char_end"])
            if rec["act_id"] not in parents.by_act:
                outside += 1
            if acts and len(ids) % args.span_stride == 1:
                checked += 1
                solid += span_is_solid(rec, acts[rec["act_id"]])
            share = heading_share(rec)
            if share is not None:
                heads.append(share)
            # Покрытие: какую долю мелкого фрагмента родитель действительно
            # содержит. Если оно низкое, подъем уводит выдачу в чужой текст,
            # и сравнение уровней мерило бы ошибку сопоставления.
            parent = parents.best_parent(rec["act_id"], rec["char_start"],
                                         rec["char_end"])
            if parent is not None and rec["char_end"] > rec["char_start"]:
                overlap = (min(rec["char_end"], parent.end)
                           - max(rec["char_start"], parent.start))
                covered.append(max(0.0, overlap)
                               / (rec["char_end"] - rec["char_start"]))
        index = BM25Index.build(ids, texts)
        del texts
        head_share = statistics.median(heads) if heads else 0.0
        say(f"{name:<11} фрагментов {len(ids):>7}, индекс {index.size_mb:>4.0f} МБ "
            f"за {index.build_seconds:.0f} с"
            + (f", актов вне базовой нарезки: {outside}" if outside else ""))
        say(f"{'':<11} символов фрагмента под шапкой: медиана "
            f"{100 * head_share:.0f}% (фрагментов с шапкой {len(heads)} из {len(ids)})")
        if checked:
            say(f"{'':<11} текст лежит в своем диапазоне непрерывно у {solid} "
                f"из {checked} проверенных фрагментов")
        poor = sum(1 for c in covered if c < 0.5)
        say(f"{'':<11} родитель покрывает фрагмент: медиана "
            f"{100 * statistics.median(covered):.0f}%, "
            f"меньше половины у {poor} фрагментов из {len(covered)}")

        deep_run: dict[str, list[str]] = {}
        for qid in qids:
            hits = [c for c, _ in index.search(queries[qid], DEEP)]
            deep_run[qid] = hits
        runs[name] = {q: parents.lift(h, meta, TOP) for q, h in deep_run.items()}
        shallow[name] = {q: parents.lift(h[:SHALLOW], meta, TOP)
                         for q, h in deep_run.items()}

        # Схлопывание: сколько разных родителей дают первые SHALLOW позиций.
        # Число нужно, чтобы читать сравнение: если их меньше TOP, мелкая
        # ветвь на этой глубине физически не может заполнить выдачу.
        distinct = [len({parents.parent_of(*meta[c]) for c in h[:SHALLOW]})
                    for h in deep_run.values()]
        short = sum(1 for r in shallow[name].values() if len(r) < TOP)
        say(f"{'':<11} разных родителей в первых {SHALLOW}: медиана "
            f"{statistics.median(distinct):.0f}, от {min(distinct)} до {max(distinct)}; "
            f"выдач короче {TOP} на этой глубине: {short}")
        report[name] = {
            "фрагментов": len(ids),
            "доля символов под шапкой, медиана": round(head_share, 4),
            "фрагментов с шапкой": len(heads),
            "покрытие родителем, медиана": round(statistics.median(covered), 4),
            "покрытие меньше половины": poor,
            "границы проверены": checked,
            "границы непрерывны": solid,
            "индекс МБ": round(index.size_mb, 1),
            "разных родителей в первых 50": {
                "медиана": float(statistics.median(distinct)),
                "минимум": min(distinct), "максимум": max(distinct)},
            "выдач короче 10 на глубине 50": short,
            "секунд": round(time.time() - t0, 1),
        }
        del index, meta, deep_run

    # Сверка ветви base с сохраненной выдачей этапа B: если реализация
    # разошлась, разница в метрике будет не про нарезку.
    say()
    saved = os.path.join(RUNDIR, "base__bm25.jsonl")
    if "base" in runs and os.path.exists(saved):
        ref = load_run(saved)
        same = [len(set(runs["base"][q]) & set(ref[q][:TOP])) / TOP
                for q in qids if q in ref]
        report["сверка base с base__bm25"] = round(100 * float(np.mean(same)), 1)
        say(f"сверка ветви base с data/runs/base__bm25.jsonl: совпадение первой "
            f"десятки {100 * float(np.mean(same)):.1f}% на {len(same)} запросах")

    # Доля позиций без разметки. Пул собирался по выдачам базовой нарезки,
    # и ветвь, которая поднимает наверх другие фрагменты, может чаще попадать
    # в неразмеченное. Если доли разойдутся, сравнение уровней смещено.
    say()
    for name in runs:
        blank = [sum(1 for c in runs[name][q] if c not in qrels[q]) for q in qids]
        report[name]["позиций без разметки из 10"] = round(float(np.mean(blank)), 2)
        say(f"{name:<11} позиций первой десятки без разметки: "
            f"{float(np.mean(blank)):.2f} из {TOP}")

    per_q: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for part_name, part in parts.items():
        per_q[part_name] = {
            name: M.per_query({q: runs[name][q] for q in part},
                              {q: qrels[q] for q in part})
            for name in runs}

    say()
    levels: dict[str, dict[str, dict[str, float]]] = {}
    for part_name, part in parts.items():
        say(f"уровни, подвыборка «{part_name}» ({len(part)} запросов):")
        say(f"{'ветвь':<12} " + " ".join(f"{m:>24}" for m in METRICS))
        levels[part_name] = {}
        for name in runs:
            cis = {m: M.bootstrap_ci(per_q[part_name][name][m]) for m in METRICS}
            say(f"{name:<12} " + " ".join(f"{str(cis[m]):>24}" for m in METRICS))
            levels[part_name][name] = {
                m: {"mean": round(ci.mean, 4), "lo": round(ci.lo, 4),
                    "hi": round(ci.hi, 4)} for m, ci in cis.items()}
        say()

    diffs = []
    say("прибавка, парный бутстрэп по запросам, p перестановочный:")
    for cand, base, note in PAIRS:
        if cand not in runs or base not in runs:
            continue
        say(f"  {cand} против {base}: {note}")
        for part_name, part in parts.items():
            row = {"ветвь": cand, "против": base, "что читается": note,
                   "подвыборка": part_name, "запросов": len(part)}
            out = []
            for m in METRICS:
                diff, p = M.paired_diff_ci(per_q[part_name][cand][m],
                                           per_q[part_name][base][m])
                out.append(f"{m} {diff} p={p:.3f}"
                           + (" *" if diff.lo * diff.hi > 0 else ""))
                row[m] = {"diff": round(diff.mean, 4), "lo": round(diff.lo, 4),
                          "hi": round(diff.hi, 4), "p": round(p, 4)}
            say(f"    {part_name:<14} ({len(part):>3}) " + "   ".join(out))
            diffs.append(row)
        say()

    # Чувствительность к глубине ступени. На глубине 50 мелкая ветвь берет
    # из своего индекса столько же позиций, сколько базовая, но родителей
    # у нее выходит меньше. Если первая десятка родителей от глубины чтения
    # не зависит, печатается одна строка: таблица, повторяющая предыдущую
    # цифра в цифру, читалась бы как ошибка, а не как проверка.
    say(f"чувствительность к глубине чтения ({SHALLOW} против {DEEP}): "
        "запросов, где первая десятка родителей разошлась")
    depth_rows = {}
    for name in runs:
        differ = sum(1 for q in qids if runs[name][q] != shallow[name][q])
        depth_rows[name] = differ
        say(f"    {name:<16} {differ} из {len(qids)}")
    say()
    say()
    (a_cand, a_base), (b_cand, b_base) = INTERACTION
    interaction = []
    if all(n in runs for n in (a_cand, a_base, b_cand, b_base)):
        say("взаимодействие размера и шапки: насколько снятие шапки при 256 "
            "дороже, чем при 512")
        for part_name, part in parts.items():
            row = {"подвыборка": part_name, "запросов": len(part)}
            out = []
            for m in METRICS:
                diff, p = M.paired_diff_ci(
                    per_q[part_name][a_cand][m] - per_q[part_name][a_base][m],
                    per_q[part_name][b_cand][m] - per_q[part_name][b_base][m])
                out.append(f"{m} {diff} p={p:.3f}"
                           + (" *" if diff.lo * diff.hi > 0 else ""))
                row[m] = {"diff": round(diff.mean, 4), "lo": round(diff.lo, 4),
                          "hi": round(diff.hi, 4), "p": round(p, 4)}
            say(f"    {part_name:<14} ({len(part):>3}) " + "   ".join(out))
            interaction.append(row)
        say()
    say("* интервал не накрывает ноль — разницу можно считать установленной")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump({"запросов": {k: len(v) for k, v in parts.items()},
                   "глубина чтения": DEEP, "глубина ступени": SHALLOW,
                   "родителей в выдаче": TOP,
                   "ветви": report, "уровни": levels,
                   "прибавка": diffs, "разошлось при глубине 50": depth_rows,
                   "взаимодействие размера и шапки": interaction},
                  f, ensure_ascii=False, indent=2)
    say()
    say(f"файлы: {os.path.relpath(REPORT, ROOT)}, {os.path.relpath(RESULT, ROOT)}")


if __name__ == "__main__":
    main()
