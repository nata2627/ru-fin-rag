#!/usr/bin/env python3
"""Голосование двух судей: в метрику идет только то, в чем они сошлись.

Один судья неотличим от собственной погрешности. 23.10.2026 одна и та же
модель на тех же 150 парах дала каппу 0,307 вместо 0,408, и причиной была
смена сборки движка, а не модель. Поэтому пул размечают двое, и правило
объявлено до разметки (`EXPERIMENTS.md` части Б, запись S17): пара, где два
судьи разошлись, в метрику не идет, а число таких пар печатается со
знаменателем.

Что считается несогласием. Оценки сравниваются дословно: 1 против 2 это
такое же несогласие, как 2 против 0. Мягкое правило («сошлись в том, что
фрагмент относится к делу») собрало бы эталон из пар, по которым судьи
спорят о главном, самодостаточности фрагмента, и метрика читала бы этот
спор как факт. Насколько несогласия грубые, видно отдельно: рядом печатается
матрица 3 на 3 и число расхождений на две ступени.

Пара, которую разметил только один судья, в метрику тоже не идет: голоса
двоих там нет. Это не то же самое, что несогласие, и считается отдельно.

Знаменатель доли несогласий это число пар, размеченных обоими. Пары, где
ответ судьи не разобрался, разметкой не считаются (`gpu_judge` такие в лист
вообще не пишет, а в листе человека на их месте стоят пустые клетки), и в
знаменатель они не попадают.

Эталон на выходе в формате qrels проекта: туда идут согласованные пары
с оценкой больше нуля. Согласованные нули это такая же разметка, но в qrels
нерелевантные пары не перечисляются, поэтому их число печатается в отчете.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.annotation import TYPES, scale_of  # noqa: E402
from rufin.benchmark import write_qrels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDIR = os.path.join(ROOT, "data", "queries")
REPORT = os.path.join(ROOT, "docs", "raw", "judge_vote.txt")
RESULT = os.path.join(ROOT, "data", "results", "judge-vote.json")
DISAGREED = os.path.join(ROOT, "docs", "raw", "judge_vote_disagreements.tsv")

SCORES = (0, 1, 2)
YES_NO = ("да", "нет")
DISAGREED_FIELDS = ["query_id", "chunk_id", "otsenka_a", "otsenka_b",
                    "stupeney", "tip_a", "tip_b"]


def short(path: str) -> str:
    """Путь для отчета: от корня проекта, если файл лежит в нем."""
    rel = os.path.relpath(path, ROOT)
    return path if rel.startswith("..") else rel


@dataclass
class Sheet:
    """Прочитанный лист судьи: оценки, пометки типа и счет пропущенных строк."""

    path: str
    marks: dict[tuple[str, str], int] = field(default_factory=dict)
    types: dict[tuple[str, str], str] = field(default_factory=dict)
    skipped: collections.Counter = field(default_factory=collections.Counter)


def answer(row: dict, *names: str) -> str:
    """Ответ из первой заполненной колонки с одним из имен.

    Колонки ответов называются по-разному: судья пишет `otvet_celikom`
    и `otvet_vmeste`, а в листе человека те же ответы стоят с номером
    вопроса, `otvet1_celikom` и `otvet2_vmeste`.
    """
    for name in names:
        value = (row.get(name) or "").strip().lower()
        if value:
            return value
    return ""


def score_of(row: dict) -> int | None:
    """Оценка строки. None, если строка разметкой не стала.

    Оценка не ставится, а собирается из двух ответов правилом `scale_of`
    из инструкции. У судьи колонка оценки заполнена кодом, у человека ее нет
    вовсе, поэтому читается то, что есть.
    """
    mark = (row.get("otsenka_0_1_2") or "").strip()
    if mark in ("0", "1", "2"):
        return int(mark)
    first = answer(row, "otvet_celikom", "otvet1_celikom")
    second = answer(row, "otvet_vmeste", "otvet2_vmeste")
    if first == "да":
        return scale_of(True, False)
    if first == "нет" and second in YES_NO:
        return scale_of(False, second == "да")
    return None


def read_sheet(path: str) -> Sheet:
    sheet = Sheet(path)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sheet.skipped["строк"] += 1
            key = ((row.get("query_id") or "").strip(),
                   (row.get("chunk_id") or "").strip())
            if not key[0] or not key[1]:
                sheet.skipped["без ключа"] += 1
                continue
            score = score_of(row)
            if score is None:
                sheet.skipped["ответ не разобран"] += 1
                continue
            if key in sheet.marks:
                # Прогон продолжается с места обрыва и уже размеченную пару
                # не переспрашивает, поэтому вторая строка на ту же пару это
                # след двух прогонов в одном файле, а не уточнение. Остается
                # первая, число таких строк идет в отчет.
                sheet.skipped["пара повторена"] += 1
                continue
            sheet.marks[key] = score
            kind = (row.get("tip") or "").strip().lower()
            if kind in TYPES:
                sheet.types[key] = kind
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-a", default=os.path.join(QDIR, "judge_14b.tsv"),
                    help="лист первого судьи; имя файла задает --judge-out "
                         "прогона на Kaggle")
    ap.add_argument("--judge-b", default=os.path.join(QDIR, "judge_32b.tsv"),
                    help="лист второго судьи")
    ap.add_argument("--name-a", default="",
                    help="имя первого судьи для отчета; по умолчанию имя файла")
    ap.add_argument("--name-b", default="")
    ap.add_argument("--out", default=os.path.join(QDIR, "qrels_vote.tsv"),
                    help="согласованная разметка в формате qrels")
    ap.add_argument("--disagreed", default=DISAGREED,
                    help="список пар, по которым судьи разошлись")
    ap.add_argument("--report", default=REPORT)
    ap.add_argument("--result", default=RESULT)
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    sheets = []
    for path, name in ((args.judge_a, args.name_a), (args.judge_b, args.name_b)):
        if not os.path.exists(path):
            raise SystemExit(f"нет {short(path)}: лист судьи "
                             f"считается на Kaggle (run_phase_c.py --judge)")
        sheet = read_sheet(path)
        sheets.append((name or os.path.basename(path), sheet))
    (name_a, a), (name_b, b) = sheets

    for name, sheet in sheets:
        say(f"судья {name}: {short(sheet.path)}, "
            f"строк {sheet.skipped['строк']}, разметкой стали "
            f"{len(sheet.marks)}")
        for reason in ("ответ не разобран", "пара повторена", "без ключа"):
            if sheet.skipped[reason]:
                say(f"   строк пропущено ({reason}): {sheet.skipped[reason]}")

    both = sorted(set(a.marks) & set(b.marks))
    if not both:
        raise SystemExit(
            f"два судьи не пересекаются ни на одной паре: у "
            f"{short(a.path)} пар {len(a.marks)}, у "
            f"{short(b.path)} пар {len(b.marks)}. Голосовать "
            f"нечем: судьям нужен один и тот же лист пула")

    agreed = [k for k in both if a.marks[k] == b.marks[k]]
    disagreed = [k for k in both if a.marks[k] != b.marks[k]]
    rough = [k for k in disagreed if abs(a.marks[k] - b.marks[k]) == 2]
    only_a = sorted(set(a.marks) - set(b.marks))
    only_b = sorted(set(b.marks) - set(a.marks))

    say()
    say(f"размечено обоими: {len(both)} пар, это знаменатель всех долей ниже")
    say(f"совпало дословно: {len(agreed)} из {len(both)} "
        f"({100 * len(agreed) / len(both):.1f}%)")
    say(f"разошлось: {len(disagreed)} из {len(both)} "
        f"({100 * len(disagreed) / len(both):.1f}%), в метрику не идут")
    say(f"из них на две ступени: {len(rough)} из {len(both)}")
    say(f"размечено только одним: {len(only_a)} у {name_a}, {len(only_b)} у "
        f"{name_b}; голоса двоих там нет, в метрику тоже не идут")

    matrix: collections.Counter = collections.Counter(
        (a.marks[k], b.marks[k]) for k in both)
    say()
    say(f"матрица оценок (строки {name_a}, столбцы {name_b}):")
    say("        " + "".join(f"{j:>8}" for j in SCORES))
    for i in SCORES:
        say(f"   {i:<4}" + "".join(f"{matrix[(i, j)]:>8}" for j in SCORES))

    os.makedirs(os.path.dirname(args.disagreed) or ".", exist_ok=True)
    with open(args.disagreed, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DISAGREED_FIELDS, delimiter="\t")
        w.writeheader()
        for key in sorted(disagreed, key=lambda k: (-abs(a.marks[k] - b.marks[k]), k)):
            w.writerow({"query_id": key[0], "chunk_id": key[1],
                        "otsenka_a": a.marks[key], "otsenka_b": b.marks[key],
                        "stupeney": abs(a.marks[key] - b.marks[key]),
                        "tip_a": a.types.get(key, "-"),
                        "tip_b": b.types.get(key, "-")})

    qrels: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for key in agreed:
        qrels[key[0]][key[1]] = a.marks[key]
    write_qrels(args.out, qrels)

    relevant = {s: sum(1 for k in agreed if a.marks[k] == s) for s in SCORES}
    say()
    say(f"согласованная разметка: {len(agreed)} пар по {len(qrels)} запросам")
    say(f"   оценка 2: {relevant[2]}, оценка 1: {relevant[1]}, "
        f"оценка 0: {relevant[0]}")
    with_relevant = sum(1 for rel in qrels.values()
                        if any(s > 0 for s in rel.values()))
    say(f"   в qrels идут оценки больше нуля: пар {relevant[2] + relevant[1]}, "
        f"запросов {with_relevant}; нерелевантные пары в qrels "
        f"не перечисляются")
    say(f"файлы: {short(args.out)}, {short(args.disagreed)}")

    # Согласие по пометке типа считается отдельно: в оценку тип не входит,
    # и в метрику он не идет вовсе. Им мерится, одинаково ли понята та часть
    # инструкции, которая делит единицы на общую норму и часть ответа.
    both_typed = sorted(set(a.types) & set(b.types))
    type_same = sum(1 for k in both_typed if a.types[k] == b.types[k])
    if both_typed:
        say()
        say(f"пометка типа: оба поставили на {len(both_typed)} парах, совпала "
            f"{type_same} ({100 * type_same / len(both_typed):.0f}%)")

    result = {
        "судья A": {"имя": name_a, "лист": short(a.path),
                    "строк": a.skipped["строк"], "пар размечено": len(a.marks),
                    "строк пропущено": dict(a.skipped)},
        "судья B": {"имя": name_b, "лист": short(b.path),
                    "строк": b.skipped["строк"], "пар размечено": len(b.marks),
                    "строк пропущено": dict(b.skipped)},
        "размечено обоими": len(both),
        "совпало": len(agreed),
        "разошлось": len(disagreed),
        "разошлось на две ступени": len(rough),
        "размечено только A": len(only_a),
        "размечено только B": len(only_b),
        "матрица": {f"{i}-{j}": matrix[(i, j)] for i in SCORES for j in SCORES},
        "согласованных по оценкам": {str(s): relevant[s] for s in SCORES},
        "в qrels": relevant[2] + relevant[1],
        "запросов в согласованной разметке": len(qrels),
        "запросов с релевантными парами": with_relevant,
        "пометка типа: оба поставили": len(both_typed),
        "пометка типа: совпала": type_same,
    }

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(args.result), exist_ok=True)
    with open(args.result, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"отчет: {short(args.report)}, {short(args.result)}")


if __name__ == "__main__":
    main()
