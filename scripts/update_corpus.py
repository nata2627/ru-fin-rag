"""Докачка новых актов «Вестника Банка России» в корпус сервиса.

Как это работает. Перечень актов у cbr.ru перестраивается шестью поисковыми
запросами, из него берутся акты, которых в корпусе сервиса еще нет, и только
их выпуски выкачиваются PDF-файлами. Обычная ночь это ноль или один выпуск,
полный сбор части А стоил 727 запросов и 22 минуты, поэтому докачка идет по
разнице, а не заново.

Сеть трогает вежливый клиент части А (`rufin.http.Client`): пауза полторы
секунды между запросами, последовательно, свой user-agent. Пакет части А
ставится из публичного репозитория по закрепленному коммиту, а не копируется
сюда: нарезка обязана совпадать до символа, иначе идентификаторы фрагментов
разойдутся с теми, что уже в корпусе.

Чего скрипт не делает. Он не обновляет акты, которые уже в корпусе: у cbr.ru
нет ни даты изменения, ни версии редакции, а правки выходят отдельными актами
«О внесении изменений», которые в состав корпуса не берутся (решение части А).
Это ограничение написано в README, а не спрятано.

Запуск: python scripts/update_corpus.py [--limit-issues 3] [--dry-run]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.corpus import FIELDS, corpus_path, read_jsonl  # noqa: E402

MIN_CYRILLIC = 0.90
MIN_CHARS = 500
TOKENIZER = "xlm-roberta-base"


def known_acts(path: Path) -> set[str]:
    return {row["act_id"] for row in read_jsonl(path)}


def act_key(number: str, date: str) -> str:
    return f"{number}_{date.replace('.', '')}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=2013)
    parser.add_argument("--pause", type=float, default=1.5)
    parser.add_argument("--limit-issues", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="только разница, без выкачки")
    parser.add_argument("--out", type=Path, default=ROOT / "results/corpus-update.json")
    args = parser.parse_args()

    import re

    from rufin.chunk_configs import BASE
    from rufin.chunking import TokenRuler, chunk_act
    from rufin.http import Client
    from rufin.sources import cbr_vestnik as vestnik
    from rufin.vestnik_split import normalize_number, split_issue

    cyrillic = re.compile(r"[а-яёА-ЯЁ]")
    latin = re.compile(r"[a-zA-Z]")

    corpus = corpus_path()
    known = known_acts(corpus)
    started = time.monotonic()
    client = Client(pause=args.pause)

    index = vestnik.fetch_acts_index(client)
    wanted = vestnik.select_corpus(index, since=args.since)
    fresh = {}
    for act in wanted:
        key = act_key(normalize_number(act["number"]), act["date"])
        if key not in known:
            fresh[key] = act

    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "index_acts": len(index),
        "wanted_acts": len(wanted),
        "known_acts": len(known),
        "new_acts_in_index": len(fresh),
        "new_acts_added": 0,
        "new_chunks": 0,
        "issues_downloaded": 0,
        "dropped": {},
    }

    if not fresh or args.dry_run:
        report["seconds"] = round(time.monotonic() - started, 1)
        write(report, args.out)
        return 0

    issues: dict[str, dict] = {}
    for key, act in fresh.items():
        code = act["issue_url"].rstrip("/").rsplit("/", 1)[-1]
        issues.setdefault(code, {"issue": act["issue"], "url": act["issue_url"], "want": {}})
        issues[code]["want"][normalize_number(act["number"])] = (key, act)
    # Границы актов внутри выпуска считаются по всем его публикациям, иначе
    # неопознанный акт не закрывает предыдущий и прилипает к чужому тексту.
    for act in index:
        code = act["issue_url"].rstrip("/").rsplit("/", 1)[-1]
        if code in issues:
            issues[code].setdefault("bounds", set()).add(normalize_number(act["number"]))

    codes = sorted(issues, key=lambda c: issues[c]["issue"])
    if args.limit_issues:
        codes = codes[: args.limit_issues]

    from transformers import AutoTokenizer

    ruler = TokenRuler(AutoTokenizer.from_pretrained(TOKENIZER))

    added: list[dict] = []
    dropped: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="vestnik-") as tmp:
        for code in codes:
            info = issues[code]
            pdf = Path(tmp) / f"{code}.pdf"
            pdf.write_bytes(client.get(info["url"]).content)
            report["issues_downloaded"] += 1
            parsed = {act.number: act for act in split_issue(str(pdf), expected=info.get("bounds"))}
            for number, (key, meta) in info["want"].items():
                found = parsed.get(number)
                if found is None:
                    dropped["не найден в выпуске"] = dropped.get("не найден в выпуске", 0) + 1
                    continue
                cyr = len(cyrillic.findall(found.text))
                lat = len(latin.findall(found.text))
                share = cyr / (cyr + lat) if cyr + lat else 0.0
                if share < MIN_CYRILLIC:
                    dropped["мало кириллицы"] = dropped.get("мало кириллицы", 0) + 1
                    continue
                if len(found.text) < MIN_CHARS:
                    dropped["слишком короткий"] = dropped.get("слишком короткий", 0) + 1
                    continue
                added.append(
                    {
                        "act_id": key,
                        "number": number,
                        "date": meta["date"],
                        "type": vestnik.act_type(meta["number"]),
                        "title": meta["title"],
                        "issue": info["issue"],
                        "url": info["url"],
                        "text": found.text,
                    }
                )

    rows = []
    for act in added:
        for chunk in chunk_act(
            act,
            ruler,
            size=BASE.size,
            overlap_share=BASE.overlap,
            strategy=BASE.strategy,
            add_heading=BASE.heading,
        ):
            row = chunk.as_dict()
            rows.append({field: row.get(field, act.get(field, "")) for field in FIELDS})

    if rows:
        with gzip.open(corpus, "at", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report["new_acts_added"] = len(added)
    report["new_chunks"] = len(rows)
    report["dropped"] = dropped
    report["acts"] = [
        {"act_id": a["act_id"], "date": a["date"], "title": a["title"]} for a in added
    ]
    report["seconds"] = round(time.monotonic() - started, 1)
    write(report, args.out)
    return 0


def write(report: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
