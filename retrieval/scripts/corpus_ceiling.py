#!/usr/bin/env python3
"""Потолок корпуса: на сколько живых вопросов ответа в нем нет в принципе.

Потолок первой ступени (`scripts/depth_ceiling.py`) отвечает на вопрос «нашел
ли поиск эталон». Этот замер про другое: существует ли эталон вообще. Разница
практическая. Если ответа нет в корпусе, то ни реранкер, ни другой энкодер, ни
генеративный слой не помогут, и честный сервис обязан сказать «не знаю».

Считается по собранным вопросам поднадзорных организаций с cbr.ru/explan: это
настоящие вопросы практиков, а не наши формулировки. Причины делятся на две
группы. Первая, документ вне состава корпуса: федеральный закон, форма
отчетности, письмо, методические материалы. Вторая, акт в корпусе есть, но
редакция другая: «Вестник» печатает акт на дату принятия, а разъяснение
относится к действующей редакции, и названного пункта у нас физически нет.

Запуск: python3 scripts/corpus_ceiling.py
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rufin.explanations import act_of, starts_clause  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QUESTIONS = os.path.join(ROOT, "data/queries/explan_questions.jsonl")
LIVE = os.path.join(ROOT, "data/queries/explan_live.jsonl")
CHUNKS = os.path.join(ROOT, "data/chunks/base.jsonl")

# По чему виден вид документа, когда акта корпуса в вопросе нет. Порядок важен:
# форма отчетности часто упоминается вместе с положением о ней.
KINDS = (
    ("форма отчетности", re.compile(r"\bформ[аеы]\s+0409|\bотчетност", re.I)),
    ("федеральный закон", re.compile(r"\d+-ФЗ|федеральн\w+ закон", re.I)),
    ("письмо или информационное сообщение", re.compile(r"письм|информационн\w+ сообщени", re.I)),
    ("мера поддержки или разъяснение без акта", re.compile(r"мер[аы] (?:по )?поддержк|пандеми", re.I)),
)


def as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.startswith("["):
        return [str(v) for v in ast.literal_eval(value)]
    return []


def kind_of(question: dict) -> str:
    haystack = f"{question.get('section', '')} {' '.join(as_list(question.get('acts')))}"
    for name, pattern in KINDS:
        if pattern.search(haystack):
            return name
    # Пусто и в разделе, и в тексте: Банк России отвечает по теме, а не по
    # акту («Национальная платежная система», «Допуск кредитных организаций»).
    # Часть таких вопросов корпус в принципе покрывает, но установить это можно
    # только руками, поэтому они идут отдельной строкой, а не записываются в
    # недоступные.
    if not as_list(question.get("acts")):
        return "акт не назван ни в разделе, ни в вопросе"
    return "назван акт, которого нет в корпусе"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data/results/corpus-ceiling.json"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "docs/raw/corpus_ceiling.txt"))
    args = ap.parse_args()

    questions = [json.loads(line) for line in open(QUESTIONS, encoding="utf-8")]
    live = [json.loads(line) for line in open(LIVE, encoding="utf-8")]
    chunks = [json.loads(line) for line in open(CHUNKS, encoding="utf-8")]

    numbers = {c["number"] for c in chunks}
    bodies: dict[str, list[str]] = collections.defaultdict(list)
    for chunk in chunks:
        bodies[chunk["number"]].append(chunk["text"])

    outside: collections.Counter = collections.Counter()
    bound = 0
    for question in questions:
        if act_of(question, numbers) is None:
            outside[kind_of(question)] += 1
        else:
            bound += 1

    # Вторая причина: акт есть, а пункта в нашей редакции нет.
    named = missing_clause = 0
    missing_by_act: collections.Counter = collections.Counter()
    for question in live:
        clauses = as_list(question.get("clauses"))
        if not clauses:
            continue
        named += 1
        act = question.get("act")
        texts = bodies.get(act, [])
        if not any(any(starts_clause(text, clause) for text in texts) for clause in clauses):
            missing_clause += 1
            missing_by_act[act] += 1

    total = len(questions)
    unknown = outside.get("акт не назван ни в разделе, ни в вопросе", 0)
    report = {
        "questions_collected": total,
        "answerable_in_principle": bound,
        "answerable_upper_bound": bound + unknown,
        "outside_corpus": total - bound,
        "outside_share": round((total - bound) / total, 4),
        "unattributed": unknown,
        "outside_by_kind": dict(outside.most_common()),
        "live_with_named_clauses": named,
        "live_clause_missing_in_our_edition": missing_clause,
        "live_clause_missing_by_act": dict(missing_by_act.most_common()),
        "acts_in_corpus": len(numbers),
    }

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    say(f"собрано вопросов практиков: {total}, актов в корпусе: {len(numbers)}")
    say(f"ответ в корпусе точно возможен: {bound} из {total} = {100 * bound / total:.0f}%")
    say(f"ответа точно нет: {total - bound - unknown} из {total} "
        f"= {100 * (total - bound - unknown) / total:.0f}%")
    say(f"не установлено (акт не назван): {unknown} из {total}, то есть верхняя граница "
        f"доступного {bound + unknown} из {total} = {100 * (bound + unknown) / total:.0f}%")
    for name, count in outside.most_common():
        say(f"   {name}: {count}")
    say()
    say(f"вопросов с названными пунктами: {named}")
    say(f"   пункта нет в нашей редакции: {missing_clause} из {named} "
        f"= {100 * missing_clause / named:.0f}%")
    for act, count in missing_by_act.most_common():
        say(f"   по актам: {act} {count}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.makedirs(os.path.dirname(args.raw), exist_ok=True)
    with open(args.raw, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    say(f"\nфайлы: {args.out}, {args.raw}")


if __name__ == "__main__":
    main()
