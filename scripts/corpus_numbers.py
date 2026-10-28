#!/usr/bin/env python3
"""Корпусные числа в README: фрагменты, акты, словарь индекса, размер, дата снимка.

Зачем. Ночной прогон докачивает акты и пересобирает индекс, поэтому
`results/corpus.json` и `results/index.json` меняются, а `verify_numbers.py`
сверяет с ними строки README. Без перезаписи первая же докачка роняла бы
сверку, красный плановый прогон рвал бы цепочку приемки автономности, а
шаг коммита не выполнялся бы. Так же работает `skills_table.py --write`:
числа вписываются перед сверкой, бот коммитит README вместе с результатами.

Почему замена идет по состоянию, а не по шаблону «любое число». В README рядом
стоят числа другой природы: 62 594 это снимок корпуса в наборе на HuggingFace и
в матрице векторов, 51 555 и 85 399 это словарь до стеммера, 991 это запросы,
а «62 606 фрагментов» в разборе графа и цитат это знаменатель замера, снятого на
том составе корпуса, и после докачки он обязан остаться прежним. Поэтому в
`results/corpus-published.json` лежат значения, которые сейчас стоят в README, и
заменяется ровно старый токен, и только в своем контексте (слово после или
перед числом). Контекст повторяет фразы, которые ищет `verify_numbers.py`,
плюс дату снимка.

Форма записи сохраняется: «1555» остается без пробела, «62 606» с пробелом
(или с переносом строки, который стоял), размер с запятой.

Если ожидаемого токена в README нет, скрипт падает: текст переписали руками
и генератор больше не держит это число. Тихо пропустить это нельзя, иначе
число устарело бы, а сверка осталась зеленой на старом месте.

Запуск:
    python scripts/corpus_numbers.py            показать, что изменится
    python scripts/corpus_numbers.py --write    переписать README и состояние
    python scripts/corpus_numbers.py --init     завести состояние по текущим файлам,
                                                если README уже показывает их числа
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "README.md"
STATE = RESULTS / "corpus-published.json"

# Что читает скрипт и что он пишет: ночной прогон обязан положить в `git add`
# все, что пишется, иначе перебазирование падает на чужих правках.
READS = ("results/corpus.json", "results/index.json")
WRITES = ("README.md", "results/corpus-published.json")


def ru_date(iso: str) -> str:
    year, month, day = iso.split("-")
    return f"{day}.{month}.{year}"


# Поле состояния, вид числа, контекст до и после. Контексты те же, что в
# фразах verify_numbers.py: «1555 акта», «62 606 фрагмента», «22,7 МиБ»,
# «словарь индекса 49 643».
FIELDS: dict[str, dict] = {
    "chunks": {"kind": "int", "before": r"", "after": r"\s+фрагмента\b"},
    "acts": {"kind": "int", "before": r"", "after": r"\s+акта\b"},
    "vocab": {"kind": "int", "before": r"словарь\s+индекса\s+", "after": r""},
    "file_mib": {"kind": "mib", "before": r"", "after": r"\s+МиБ"},
    "measured_at": {"kind": "date", "before": r"На\s+", "after": r"\s+в\s+корпусе\b"},
}


def current() -> dict:
    corpus = json.loads((RESULTS / "corpus.json").read_text(encoding="utf-8"))
    index = json.loads((RESULTS / "index.json").read_text(encoding="utf-8"))
    return {
        "chunks": corpus["chunks"],
        "acts": corpus["acts"],
        "vocab": index["vocab"],
        "file_mib": corpus["file_mib"],
        "measured_at": corpus["measured_at"],
    }


def _token_pattern(kind: str, value) -> str:
    """Шаблон одного старого токена, с границами по цифрам."""
    if kind == "int":
        text = str(int(value))
        groups = []
        while text:
            groups.insert(0, text[-3:])
            text = text[:-3]
        # Разделитель тысяч в README бывает пробел, неразрывный пробел или перенос.
        body = r"[  \n]?".join(groups) if len(groups) > 1 else groups[0]
        return rf"(?<!\d)(?P<tok>{body})(?!\d)"
    if kind == "mib":
        return rf"(?<![\d,])(?P<tok>{re.escape(f'{value:.1f}'.replace('.', ','))})(?!\d)"
    return rf"(?<![\d.])(?P<tok>{re.escape(ru_date(value))})(?![\d.])"


def _new_token(kind: str, old_token: str, value) -> str:
    if kind == "int":
        separators = re.findall(r"[  \n]", old_token)
        digits = str(int(value))
        if not separators:
            return digits
        sep = separators[0]
        groups = []
        while digits:
            groups.insert(0, digits[-3:])
            digits = digits[:-3]
        return sep.join(groups)
    if kind == "mib":
        return f"{value:.1f}".replace(".", ",")
    return ru_date(value)


def _pattern(field: str, old_value) -> re.Pattern:
    spec = FIELDS[field]
    token = _token_pattern(spec["kind"], old_value)
    return re.compile(rf"(?P<pre>{spec['before']}){token}(?P<post>{spec['after']})")


def _замена(kind: str, value):
    """Функция замены с привязанными полем и значением, а не замыкание на цикл."""

    def sub(match: re.Match) -> str:
        tok = _new_token(kind, match.group("tok"), value)
        return match.group("pre") + tok + match.group("post")

    return sub


def rewrite(readme: str, old: dict, new: dict) -> tuple[str, dict[str, int]]:
    """README с новыми числами и счет замен по полям.

    Падает, если для поля старый токен не найден ни разу.
    """
    counts: dict[str, int] = {}
    for field, spec in FIELDS.items():
        pattern = _pattern(field, old[field])
        found = len(pattern.findall(readme))
        if not found:
            raise SystemExit(
                f"в README нет ожидаемого '{field}' со значением {old[field]} в своем контексте "
                f"(до: '{spec['before'] or '-'}', после: '{spec['after'] or '-'}'): "
                "текст переписали руками, генератор больше не держит это число. "
                "Верни фразу или поправь FIELDS в scripts/corpus_numbers.py"
            )

        readme = pattern.sub(_замена(spec["kind"], new[field]), readme)
        counts[field] = found
    return readme, counts


def run(readme_path: Path, state_path: Path, write: bool) -> int:
    new = current()
    old = json.loads(state_path.read_text(encoding="utf-8"))
    text = readme_path.read_text(encoding="utf-8")
    updated, counts = rewrite(text, old, new)
    changed = {k: (old[k], new[k]) for k in FIELDS if old[k] != new[k]}
    if not write:
        print("изменится:" if changed else "числа README уже совпадают с замерами")
        for field, (was, now) in changed.items():
            print(f"  {field}: {was} -> {now}, вхождений {counts[field]}")
        return 0
    if updated != text:
        readme_path.write_text(updated, encoding="utf-8")
    # Состояние пишется после README: упавшая замена не должна оставить
    # состояние, которое уже не про текст.
    state_path.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("корпусные числа в README " + ("перезаписаны по замерам" if changed else "уже совпадали"))
    return 0


def init(readme_path: Path, state_path: Path) -> int:
    """Завести состояние по текущим файлам. Токены обязаны уже стоять в README."""
    new = current()
    text = readme_path.read_text(encoding="utf-8")
    rewrite(text, new, new)
    state_path.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"состояние записано: {state_path.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="переписать README и состояние")
    ap.add_argument("--init", action="store_true", help="завести состояние по текущим файлам")
    args = ap.parse_args()
    if args.init:
        return init(README, STATE)
    return run(README, STATE, args.write)


if __name__ == "__main__":
    sys.exit(main())
