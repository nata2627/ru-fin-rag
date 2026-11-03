"""Stage 12 synthetic questions (decision 28): prepare generator batches, collect answers.

Style examples come from collected questions that have no route to the corpus: they are
real questions of practitioners and cannot leak into the live dev or test, which hold only
questions with a route. Copying is capped at the 90th percentile of the live set (0.30),
measured on live questions against their labeled positives, instead of an invented bar.
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufin.explanations import body_of  # noqa: E402
from rufin.leak import forbidden_chunks  # noqa: E402
from rufin.queryfilter import copy_score, po_russki  # noqa: E402

LIVE12 = ROOT / "data" / "interim" / "live12"
MAX_COPY = 0.30
BATCH = 10
SHOTS = 4
STYLES = {
    "ситуация": (0.60, """Опиши рабочую ситуацию сотрудника банка или другой поднадзорной организации в двух-четырех
предложениях и закончи вопросом в Банк России, ответ на который дает этот фрагмент. Норму, пункт
и акт не называй, пиши обстоятельствами дела, как пишут живые люди."""),
    "письмо": (0.25, """Напиши обращение в Банк России длиной 700-1200 знаков: кто пишет, какая у организации
ситуация, какие цифры или обстоятельства, что уже сделали, и в конце вопрос, ответ на который
дает этот фрагмент. Пункт не называй."""),
    "две нормы": (0.15, """Ниже два фрагмента. Сформулируй вопрос практика в двух-четырех предложениях, для ответа
на который нужны ОБА фрагмента: по одному из них ответить нельзя. Пункты не называй."""),
}
# The first probe (no terms, no act numbers) shared 0.423 of its words with the best
# single positive against 0.597 for live questions (synth12_gate.py): practitioners write
# the act's terms as they are, and 302 of 1 002 live questions name the act. So terms are allowed, whole phrases
# still are not, and a third of the tasks may name the act.
COMMON = """Требования: по-русски; термины акта (названия счетов, показателей, форм, видов операций)
употребляй как есть, но целые фразы из фрагмента не переписывай; конкретно, чтобы ответ находился
именно здесь, а не в любом документе Банка России; одной строкой, без переносов."""
NAME_ACT = "Можно назвать номер акта ({number}), как это делают в обращениях."
NAME_ACT_SHARE = 0.3
TASK = """Ниже пронумерованные задания. Для каждого напиши один вопрос по правилам задания. Перед
заданиями примеры настоящих вопросов в Банк России: смотри на манеру, содержание чужое.
Ответ: по одной строке на задание, номер и вопрос через табуляцию, без заголовка и пояснений.

"""


def style_examples(seed: int) -> list[str]:
    rows = [json.loads(l) for l in open(LIVE12 / "candidates.jsonl", encoding="utf-8")]
    out = [r["text"] for r in rows if not r["route"] and 80 <= len(r["text"]) <= 1500]
    random.Random(seed).shuffle(out)
    return out


def pick_second(first: dict, chunks: dict, by_act: dict, forbidden: frozenset, rng: random.Random) -> str | None:
    """A far fragment of the same act; the forbidden check matters here as much as for the first."""
    others = [c for c in by_act[first["act_id"]] if abs(chunks[c]["position"] - first["position"]) > 2
              and len(chunks[c]["text"]) >= 300 and c not in forbidden]
    return rng.choice(others) if others else None


def prepare(a) -> None:
    frags = [json.loads(l) for l in open(a.fragments, encoding="utf-8")]
    chunks = {}
    by_act = {}
    for line in open(ROOT / "data" / "chunks" / "base.jsonl", encoding="utf-8"):
        c = json.loads(line)
        chunks[c["chunk_id"]] = c
        by_act.setdefault(c["act_id"], []).append(c["chunk_id"])
    rng = random.Random(a.seed)
    names = list(STYLES)
    weights = [STYLES[n][0] for n in names]
    examples = style_examples(a.seed)
    forbidden = forbidden_chunks()
    tasks = []
    for n, f in enumerate(frags, start=1):
        style = rng.choices(names, weights)[0]
        second = None
        if style == "две нормы":
            second = pick_second(chunks[f["chunk_id"]], chunks, by_act, forbidden, rng)
            if second is None:
                style = "ситуация"
        tasks.append({"nomer": n, "style": style, "chunk_id": f["chunk_id"], "chunk_id_2": second,
                      "name_act": rng.random() < NAME_ACT_SHARE})
    os.makedirs(a.work, exist_ok=True)
    for b in range(0, len(tasks), BATCH):
        part = tasks[b:b + BATCH]
        shots = examples[(b // BATCH) * SHOTS % len(examples):][:SHOTS]
        text = [TASK, "ПРИМЕРЫ\n"] + [f"- {s}" for s in shots] + ["\nЗАДАНИЯ\n"]
        for t in part:
            c = chunks[t["chunk_id"]]
            act = (NAME_ACT.format(number=c["number"]) if t["name_act"] else "Номер акта не называй.") + "\n"
            text.append(f"=== Задание {t['nomer']} ({t['style']}) ===\n{STYLES[t['style']][1]}\n{COMMON}\n{act}"
                        f"Фрагмент:\n{body_of(c)}\n")
            if t["chunk_id_2"]:
                text.append(f"Второй фрагмент:\n{body_of(chunks[t['chunk_id_2']])}\n")
        with open(os.path.join(a.work, f"batch_{b // BATCH + 1:03d}.prompt.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(text))
    with open(os.path.join(a.work, "tasks.jsonl"), "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"tasks {len(tasks)}, batches {(len(tasks) + BATCH - 1) // BATCH}")


def collect(a) -> None:
    tasks = {t["nomer"]: t for t in map(json.loads, open(os.path.join(a.work, "tasks.jsonl"), encoding="utf-8"))}
    chunks = {}
    for line in open(ROOT / "data" / "chunks" / "base.jsonl", encoding="utf-8"):
        c = json.loads(line)
        chunks[c["chunk_id"]] = c
    got, kept, why = 0, [], {}
    for name in sorted(os.listdir(a.work)):
        if not name.endswith(".answers.tsv"):
            continue
        for row in csv.reader(open(os.path.join(a.work, name), encoding="utf-8"), delimiter="\t"):
            if len(row) < 2 or not row[0].strip().isdigit() or int(row[0]) not in tasks:
                continue
            got += 1
            t, text = tasks[int(row[0])], row[1].strip()
            ok, reason = po_russki(text)
            share = max(copy_score(text, chunks[c]["text"])[1] for c in (t["chunk_id"], t["chunk_id_2"]) if c)
            if not ok:
                why[reason] = why.get(reason, 0) + 1
                continue
            if share > MAX_COPY:
                why["списывание"] = why.get("списывание", 0) + 1
                continue
            kept.append({"query_id": f"s12{t['nomer']:05d}", "text": text, "style": t["style"],
                         "source": [c for c in (t["chunk_id"], t["chunk_id_2"]) if c], "copy_share": round(share, 3),
                         "new": True})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"tasks {len(tasks)}, answered {got}, kept {len(kept)}, dropped {why}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--fragments", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--seed", type=int, default=28)
    c = sub.add_parser("collect")
    c.add_argument("--work", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"prepare": prepare, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    main()
