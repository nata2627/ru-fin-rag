"""Candidates for the stage 12 live set from the full collection of Bank of Russia answers.

Every collected question gets a route to an act of the corpus, in order of trust:
the section title names the act (the Bank's own attribution), the question names it,
the topic of the block names it (checked: 63 of 65 agree with the known act), or the
question names a reporting form whose filling procedure is in the corpus. A question
with no route is outside the corpus. Near duplicates (word Jaccard 0.9) share a group,
so a group never ends up on both sides of the dev/test split.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufin.explanations import ACT_NUMBER, act_of, load_questions  # noqa: E402

FORM = re.compile(r"(?<!\d)(0[34]\d{5})(?!\d)")
FORM_DEF = re.compile(
    r"(?:Порядок составления и представления отчетности по форме|Код формы по ОКУД)\s*(0[34]\d{5})", re.I)
WORD = re.compile(r"\w+")


def words(text: str) -> frozenset[str]:
    return frozenset(w for w in WORD.findall(text.lower().replace("ё", "е")) if len(w) > 2)


def groups(texts: list[str], threshold: float) -> list[int]:
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sets = [words(t) for t in texts]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if not a or not b or min(len(a), len(b)) / max(len(a), len(b)) < threshold:
                continue
            if len(a & b) / len(a | b) >= threshold:
                parent[find(j)] = find(i)
    return [find(i) for i in range(len(texts))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dup", type=float, default=0.9)
    a = ap.parse_args()

    acts_of_number = collections.defaultdict(set)
    act_date = {}
    form_def = collections.defaultdict(set)
    for line in open(ROOT / "data" / "chunks" / "base.jsonl", encoding="utf-8"):
        c = json.loads(line)
        acts_of_number[c["number"]].add(c["act_id"])
        act_date[c["act_id"]] = c["date"]
        for f in FORM_DEF.findall(c["text"]):
            form_def[f].add(c["act_id"])
    numbers = set(acts_of_number)

    def latest(act_ids) -> str:
        return max(act_ids, key=lambda x: act_date[x][-4:] + act_date[x][3:5] + act_date[x][:2])

    live = {json.loads(l)["text"]: json.loads(l)["query_id"]
            for l in open(ROOT / "data" / "queries" / "explan_live.jsonl", encoding="utf-8")}

    seen = {}
    rows = []
    for q in load_questions(str(a.collected)):
        if q["text"] in seen:
            continue
        seen[q["text"]] = True
        route, act = None, act_of(q, numbers)
        if act:
            route = "раздел" if act in q.get("acts", ()) else "текст"
        else:
            topic = [n for n in ACT_NUMBER.findall(q.get("topic", "")) if n in numbers]
            forms = [f for f in FORM.findall(" ".join((q["section"], q.get("topic", ""), q["text"])))
                     if f in form_def]
            if topic:
                route, act = "тема", topic[0]
            elif forms:
                route, act = "форма", latest(form_def[forms[0]])
        act_id = act if act and "_" in act else (latest(acts_of_number[act]) if act else None)
        rows.append({"query_id": live.get(q["text"]), "text": q["text"], "section": q["section"],
                     "topic": q.get("topic", ""), "clauses": q.get("clauses", []), "url": q["url"],
                     "route": route, "act_id": act_id})

    for r, g in zip(rows, groups([r["text"] for r in rows], a.dup)):
        r["group"] = g

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_groups = len({r["group"] for r in rows})
    print(f"questions {len(rows)}, duplicate groups merge {len(rows) - n_groups}, live already {sum(1 for r in rows if r['query_id'])}")
    by_route = collections.Counter((r["route"] or "вне корпуса", bool(r["query_id"])) for r in rows)
    for k, v in sorted(by_route.items(), key=lambda kv: -kv[1]):
        print(f"  {k[0]:<12} {'в 325' if k[1] else 'новые'}: {v}")
    new = [r for r in rows if r["route"] and not r["query_id"]]
    print("new by act:", collections.Counter(r["act_id"] for r in new).most_common(30))


if __name__ == "__main__":
    main()
