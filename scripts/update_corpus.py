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

Отказ cbr.ru, пропавший выпуск и поломка это три разные вещи. Выпуск, которого
нет по адресу из перечня (404), это свойство выпуска: его акты уходят в
`skipped.json` с причиной и датой, а прогон идет к следующему выпуску. Отказ
сайта и поломка возвращаются разными кодами.
Сайт не ответил, ответил ошибкой или оборвал соединение: корпус на диске цел,
докачивать нечего, код выхода 75 (`EXIT_SOURCE_UNAVAILABLE`), и ночной прогон
после него продолжает переиндексацию, регрессию и отметку живости. Сайт
ответил, но перечень разобран в ноль актов: это уже поломка разбора, код 1, и
прогон краснеет, потому что молча пустой перечень означал бы «новых актов нет»
навсегда.

Запуск: python scripts/update_corpus.py [--limit-issues 3] [--dry-run]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.corpus import FIELDS, corpus_path, read_jsonl  # noqa: E402
from rufinrag.pipeline import EXIT_SOURCE_UNAVAILABLE  # noqa: E402

MIN_CYRILLIC = 0.90
MIN_CHARS = 500
TOKENIZER = "xlm-roberta-base"
SKIPPED_FILE = "corpus/skipped.json"
# Код ответа достается из текста ошибки: вежливый клиент части А заворачивает
# исключение requests в `RuntimeError` и сам объект наружу не отдает. Разбор
# держится на формате сообщения requests, поэтому неузнанный текст считается
# отказом сайта, то есть более осторожным из двух толкований.
HTTP_CODE = re.compile(r"\b(\d{3}) (?:Client|Server) Error\b")


def http_code(err: Exception) -> int | None:
    found = HTTP_CODE.search(str(err))
    return int(found.group(1)) if found else None


def known_acts(path: Path) -> set[str]:
    return {row["act_id"] for row in read_jsonl(path)}


def load_skipped(path: Path) -> dict[str, dict]:
    """Акты, которые уже пробовали взять и не смогли.

    Без этого файла прогон каждую ночь качает одни и те же выпуски: в перечне
    у cbr.ru 97 актов, которых нет в корпусе, и все 97 не берутся по тем же
    причинам, по которым их не взяла часть А (скан вместо текста, акт не
    опознан в выпуске). Первый прогон 22.10.2026 скачал из-за этого 80 PDF и
    добавил ноль фрагментов. Причина и дата попытки пишутся рядом, повторная
    попытка делается флагом `--retry-skipped`, когда есть повод.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_skipped(path: Path, skipped: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {key: skipped[key] for key in sorted(skipped)}
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pending_acts(wanted: list[dict], known: set[str], blocked: set[str], normalize) -> dict:
    """Акты перечня, которых нет ни в корпусе, ни в памяти о неудачах.

    Отдельной функцией, потому что это и есть идемпотентность докачки: второй
    прогон на том же перечне обязан вернуть пустой разбор, иначе тот же акт
    приедет в корпус дважды.
    """
    fresh: dict[str, dict] = {}
    for act in wanted:
        key = act_key(normalize(act["number"]), act["date"])
        if key not in known and key not in blocked:
            fresh[key] = act
    return fresh


def append_rows(corpus: Path, rows: list[dict]) -> None:
    """Дописать фрагменты так, чтобы обрыв не испортил корпус.

    Дописывание прямо в gzip читается обратно (gzip допускает склейку), но
    процесс, снятый посреди записи, оставляет обрезанный хвост, и тогда файл не
    читается целиком: падает и сборка индекса, и старт сервиса, и сам этот
    скрипт. Поэтому новый файл собирается рядом и подменяет старый одним
    системным вызовом: либо старый корпус, либо новый, промежуточного нет.
    """
    tmp = corpus.with_suffix(corpus.suffix + ".new")
    try:
        with open(corpus, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        with gzip.open(tmp, "at", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, corpus)
    finally:
        tmp.unlink(missing_ok=True)


def act_key(number: str, date: str) -> str:
    return f"{number}_{date.replace('.', '')}"


def chunk_rows(act: dict, ruler) -> list[dict]:
    """Фрагменты одного акта в том виде, в каком они лежат в корпусе сервиса.

    Нарезка идет базовой конфигурацией части А, а поля отбираются по `FIELDS`:
    сервис держит меньше полей, чем часть А, и лишние в корпус не попадают.
    """
    from rufin.chunk_configs import BASE
    from rufin.chunking import chunk_act

    rows = []
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
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=2013)
    parser.add_argument("--pause", type=float, default=1.5)
    parser.add_argument("--limit-issues", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="только разница, без выкачки")
    parser.add_argument(
        "--retry-skipped", action="store_true", help="попробовать акты, которые не дались раньше"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results/corpus-update.json")
    args = parser.parse_args()

    from rufin.chunking import TokenRuler
    from rufin.http import Client
    from rufin.sources import cbr_vestnik as vestnik
    from rufin.vestnik_split import normalize_number, split_issue

    cyrillic = re.compile(r"[а-яёА-ЯЁ]")
    latin = re.compile(r"[a-zA-Z]")

    corpus = corpus_path()
    known = known_acts(corpus)
    skipped_path = corpus.parent.parent / SKIPPED_FILE
    # История попыток читается всегда: `--retry-skipped` снимает фильтр, но не
    # стирает даты и счетчик, иначе акт, который не берется полгода, после
    # каждой повторной попытки выглядел бы впервые увиденным.
    skipped = load_skipped(skipped_path)
    blocked: set[str] = set() if args.retry_skipped else set(skipped)
    started = time.monotonic()
    client = Client(pause=args.pause)

    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "ok",
        "index_acts": 0,
        "wanted_acts": 0,
        "known_acts": len(known),
        "skipped_before": len(blocked),
        "new_acts_in_index": 0,
        "new_acts_added": 0,
        "new_chunks": 0,
        "issues_downloaded": 0,
        "dropped": {},
    }

    def give_up(err: Exception) -> int:
        """Источник отказал: записать причину и выйти кодом «попробуй позже».

        Клиент части А к этому моменту уже сделал свои три попытки с паузой, и
        четвертая ничего не изменит: 403 это отказ, а не сетевая икота.
        """
        report["status"] = "source_unavailable"
        report["source_error"] = str(err)
        report["seconds"] = round(time.monotonic() - started, 1)
        write(report, args.out)
        return EXIT_SOURCE_UNAVAILABLE

    try:
        index = vestnik.fetch_acts_index(client)
    except RuntimeError as err:
        return give_up(err)

    wanted = vestnik.select_corpus(index, since=args.since)
    report["index_acts"] = len(index)
    report["wanted_acts"] = len(wanted)
    if not wanted:
        # Сайт ответил, а под состав корпуса не подошло ничего. В норме это
        # 1656 актов, поэтому ноль означает не спокойную ночь, а то, что
        # разбор перечня больше не понимает верстку cbr.ru.
        report["status"] = "index_not_parsed"
        report["seconds"] = round(time.monotonic() - started, 1)
        write(report, args.out)
        print(
            f"перечень cbr.ru разобран в ноль актов из {len(index)} публикаций: "
            "похоже, поменялась верстка поиска по «Вестнику»",
            file=sys.stderr,
        )
        return 1

    fresh = pending_acts(wanted, known, blocked, normalize_number)
    report["new_acts_in_index"] = len(fresh)

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
    today = time.strftime("%Y-%m-%d")

    def drop(key: str, reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1
        seen = skipped.get(key, {})
        skipped[key] = {
            "reason": reason,
            "first_try": seen.get("first_try", today),
            "last_try": today,
            "tries": seen.get("tries", 0) + 1,
        }

    with tempfile.TemporaryDirectory(prefix="vestnik-") as tmp:
        for code in codes:
            info = issues[code]
            pdf = Path(tmp) / f"{code}.pdf"
            try:
                pdf.write_bytes(client.get(info["url"]).content)
            except RuntimeError as err:
                if http_code(err) == 404:
                    # Выпуска нет по адресу из перечня. Это свойство выпуска, а
                    # не отказ сайта, поэтому его акты уходят в `skipped.json` с
                    # причиной и датой, и прогон идет к следующему выпуску.
                    # Иначе он возвращался бы к этому адресу каждую ночь и
                    # останавливался на нем, не доходя до остальных выпусков.
                    for key, _ in info["want"].values():
                        drop(key, "выпуск не найден на cbr.ru")
                    continue
                # Сайт отказал или не ответил: собранное до этого места
                # сохраняем, остальное возьмет следующий прогон. В
                # `skipped.json` такой акт не пишется, причина не в нем.
                report["status"] = "source_unavailable"
                report["source_error"] = str(err)
                break
            report["issues_downloaded"] += 1
            parsed = {act.number: act for act in split_issue(str(pdf), expected=info.get("bounds"))}
            for number, (key, meta) in info["want"].items():
                found = parsed.get(number)
                if found is None:
                    drop(key, "не найден в выпуске")
                    continue
                cyr = len(cyrillic.findall(found.text))
                lat = len(latin.findall(found.text))
                share = cyr / (cyr + lat) if cyr + lat else 0.0
                if share < MIN_CYRILLIC:
                    drop(key, "мало кириллицы")
                    continue
                if len(found.text) < MIN_CHARS:
                    drop(key, "слишком короткий")
                    continue
                added.append(
                    {
                        "act_id": key,
                        "number": number,
                        "date": meta["date"],
                        # `kind` читает шапка фрагмента в части А, без него нарезка падает
                        "kind": found.kind,
                        "type": vestnik.act_type(meta["number"]),
                        "title": meta["title"],
                        "issue": info["issue"],
                        "issue_url": info["url"],
                        "text": found.text,
                    }
                )

    rows = []
    for act in added:
        rows.extend(chunk_rows(act, ruler))

    if rows:
        append_rows(corpus, rows)

    save_skipped(skipped_path, skipped)
    report["new_acts_added"] = len(added)
    report["new_chunks"] = len(rows)
    report["dropped"] = dropped
    report["skipped_total"] = len(skipped)
    report["acts"] = [
        {"act_id": a["act_id"], "date": a["date"], "title": a["title"]} for a in added
    ]
    report["seconds"] = round(time.monotonic() - started, 1)
    write(report, args.out)
    return EXIT_SOURCE_UNAVAILABLE if report["status"] == "source_unavailable" else 0


def write(report: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
