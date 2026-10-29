#!/usr/bin/env python3
"""Плакат набора: семь показателей, «как было» и «что стало».

Читает один или два снимка `scripts/nabor_audit.py`. С одним снимком столбец
«что стало» пуст и подписан «еще не измерено». Каждое число плаката берется
из снимка, руками в SVG и в таблицу документа ничего не вписывается: таблица
в `docs/18-nabor-do-i-posle.md` переписывается здесь же, между метками.

    python3 scripts/nabor_plakat.py --do data/results/nabor-audit-do.json
    python3 scripts/nabor_plakat.py --do ...-do.json --posle ...-posle.json --out docs/img/nabor-do-i-posle.svg
"""
from __future__ import annotations

import argparse
import json
import os
import re
from xml.sax.saxutils import escape

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOC = os.path.join(ROOT, "docs", "18-nabor-do-i-posle.md")
TABLE_START = "<!-- tablica: nabor_plakat.py -->"
TABLE_END = "<!-- /tablica -->"
NOT_MEASURED = "еще не измерено"

W = 1000
ROW_H = 118
TOP = 132
COL_DO = 384
COL_POSLE = 704
METER_W = 240
LINE_CHARS = 44


def num(x: float, digits: int | None = None) -> str:
    """Запись документов: пробел между разрядами, десятичная запятая."""
    if digits is None and float(x).is_integer():
        return f"{int(x):,}".replace(",", " ")
    return f"{x:.{digits if digits is not None else 2}f}".replace(".", ",")


def pct(s: dict) -> str:
    # Доли меньше десятой печатаются с одним знаком: 4% вместо 3,8% теряет утечку.
    p = s["доля"] * 100
    return f"{num(p, 1)}%" if 0 < p < 10 else f"{round(p)}%"


def of(s: dict) -> str:
    return f"{num(s['числитель'])} из {num(s['знаменатель'])}"


def blocks(snap: dict) -> list[dict]:
    """Семь блоков: заголовочная доля (метр), подпись к ней и строка подробностей."""
    m1, m2 = snap["M1 плотность меток"], snap["M2 происхождение позитива"]
    m3, m4 = snap["M3 концентрация теста"], snap["M4 утечка"]
    m5, m6, m7 = snap["M5 покрытие корпуса"], snap["M6 копирование"], snap["M7 длина запроса"]
    syn, live = m1["синтетика вместе"], m1["подвыборки"]["живые"]
    leak_where = ", ".join(f"«{k}» {num(v)}" for k, v in m4["по подвыборкам"].items() if v)
    long_key = f"длиннее {m7['порог длины']}"
    threshold = ("порога нет" if m6["порог"] is None
                 else f"выше порога {num(m6['порог'])}: {of(m6['выше порога'])}")
    return [
        {"код": "M1", "имя": "плотность меток",
         "что": "запросы ровно с одной меткой",
         "доля": syn["ровно одна метка"], "подпись": "синтетических запросов",
         "ниже": f"среднее {num(syn['среднее'])} на запрос; живые: "
                 f"{of(live['ровно одна метка'])}, среднее {num(live['среднее'])}"},
        {"код": "M2", "имя": "происхождение позитива",
         "что": "позитивы, подтвержденные судьей или человеком",
         "доля": m2["синтетика вместе"]["подтверждено разметкой"], "подпись": "позитивов синтетики",
         "ниже": f"живые: {of(m2['живые']['подтверждено разметкой'])}; "
                 f"остальное: названные пункты и соседи"},
        {"код": "M3", "имя": "концентрация теста",
         "что": "метки живого теста на самый частый акт",
         "доля": m3["метки самого частого акта"],
         "подпись": f"меток про {m3['самый частый акт'].split('_')[0]}",
         "ниже": f"актов {num(m3['актов'])}, живых запросов {num(m3['запросов'])}"},
        {"код": "M4", "имя": "утечка",
         "что": "эталоны теста среди источников обучающих вопросов",
         "доля": m4["пересечение"], "подпись": "эталонных фрагментов",
         "ниже": f"по подвыборкам: {leak_where or 'нигде'}"},
        {"код": "M5", "имя": "покрытие корпуса",
         "что": "фрагменты, по которым есть обучающий вопрос",
         "доля": m5["фрагменты"], "подпись": "фрагментов корпуса",
         "ниже": f"актов {of(m5['акты'])} ({pct(m5['акты'])}), "
                 f"обучающих вопросов {num(m5['обучающих вопросов'])}"},
        {"код": "M6", "имя": "копирование",
         "что": "доля вопроса, списанная из фрагмента одной цепочкой",
         "доля": {"числитель": None, "знаменатель": None, "доля": m6["медиана"]},
         "заголовок": f"медиана {num(m6['медиана'], 2)}", "подпись": f"по {num(m6['вопросов'])} вопросам",
         "ниже": f"95-я перцентиль {num(m6['95-я перцентиль'], 2)}, "
                 f"максимум {num(m6['максимум'], 2)}; {threshold}"},
        {"код": "M7", "имя": "длина запроса",
         "что": f"живые вопросы {long_key} знаков",
         "доля": m7["живые все"][long_key], "подпись": "живых вопросов",
         "ниже": f"медиана {num(m7['живые все']['медиана'])}, максимум "
                 f"{num(m7['живые все']['максимум'])}; в тесте {of(m7['живые в тесте'][long_key])}, "
                 f"у синтетики {of(m7['синтетика'][long_key])}"},
    ]


def headline(b: dict) -> str:
    return b.get("заголовок") or f"{of(b['доля'])} ({pct(b['доля'])})"


STYLE = """
  .surface { fill: #fcfcfb; }
  .ink { fill: #0b0b0b; }
  .ink2 { fill: #52514e; }
  .muted { fill: #8a8984; }
  .rule { stroke: #e4e3df; }
  .track { fill: #dce8f7; }
  .bar { fill: #2a78d6; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .ink { fill: #f0efec; }
    .ink2 { fill: #c3c2b7; }
    .muted { fill: #8f8e86; }
    .rule { stroke: #33332f; }
    .track { fill: #24364c; }
    .bar { fill: #3987e5; }
  }
  text { font-family: -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }
"""


def cell(x: int, y: int, b: dict | None) -> list[str]:
    if b is None:
        return [f'<text x="{x}" y="{y + 30}" class="muted" font-size="15">{NOT_MEASURED}</text>']
    out = [f'<text x="{x}" y="{y + 26}" class="ink v" font-size="21" font-weight="600">'
           f'{escape(headline(b))}</text>',
           f'<text x="{x}" y="{y + 44}" class="ink2 v" font-size="12">{escape(b["подпись"])}</text>',
           f'<rect x="{x}" y="{y + 54}" width="{METER_W}" height="8" rx="4" class="track"/>']
    fill = max(round(METER_W * b["доля"]["доля"]), 0)
    if fill:
        out.append(f'<rect x="{x}" y="{y + 54}" width="{max(fill, 8)}" height="8" rx="4" class="bar"/>')
    for i, line in enumerate(wrap(b["ниже"])):
        out.append(f'<text x="{x}" y="{y + 80 + 15 * i}" class="ink2 v" font-size="11.5">{escape(line)}</text>')
    return out


def wrap(text: str) -> list[str]:
    """Две колонки стоят рядом, поэтому строка подробностей переносится по словам."""
    lines: list[str] = []
    for part in text.split("; "):
        line = ""
        # Пробел между разрядами числа не место для переноса: «6 112» не рвется.
        for word in re.split(r" (?!\d{3}\b)", part):
            if line and len(line) + 1 + len(word) > LINE_CHARS:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
    return lines


def svg(do: dict, posle: dict | None) -> str:
    rows_do, rows_posle = blocks(do), blocks(posle) if posle else [None] * 7
    h = TOP + ROW_H * len(rows_do) + 44
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" '
           f'viewBox="0 0 {W} {h}" role="img" aria-labelledby="t">',
           '<title id="t">Набор данных: семь показателей, как было и что стало</title>',
           f"<style>{STYLE}</style>",
           f'<rect width="{W}" height="{h}" class="surface"/>',
           '<text x="32" y="44" class="ink" font-size="24" font-weight="700">'
           'Набор данных: как было и что стало</text>',
           '<text x="32" y="70" class="ink2" font-size="13">Семь показателей по определениям, '
           'записанным до первого замера (docs/18-nabor-do-i-posle.md). Метр показывает долю.</text>',
           f'<text x="32" y="{TOP - 16}" class="muted" font-size="12" font-weight="600">ПОКАЗАТЕЛЬ</text>',
           f'<text x="{COL_DO}" y="{TOP - 16}" class="muted" font-size="12" font-weight="600">'
           f'КАК БЫЛО, снимок «{escape(do["тег"])}»</text>',
           f'<text x="{COL_POSLE}" y="{TOP - 16}" class="muted" font-size="12" font-weight="600">'
           f'ЧТО СТАЛО{escape(", снимок «" + posle["тег"] + "»") if posle else ""}</text>']
    for i, (b, a) in enumerate(zip(rows_do, rows_posle)):
        y = TOP + i * ROW_H
        out.append(f'<line x1="32" x2="{W - 32}" y1="{y}" y2="{y}" class="rule" stroke-width="1"/>')
        out.append(f'<text x="32" y="{y + 26}" class="ink" font-size="15" font-weight="600">'
                   f'{b["код"]}. {escape(b["имя"])}</text>')
        out.append(f'<text x="32" y="{y + 46}" class="ink2" font-size="12">{escape(b["что"])}</text>')
        out += cell(COL_DO, y, b) + cell(COL_POSLE, y, a)
    y = TOP + ROW_H * len(rows_do)
    out.append(f'<line x1="32" x2="{W - 32}" y1="{y}" y2="{y}" class="rule" stroke-width="1"/>')
    out.append(f'<text x="32" y="{y + 28}" class="muted" font-size="11.5">Числа из '
               f'data/results/nabor-audit-*.json, рисует scripts/nabor_plakat.py. '
               f'Синтетика и живые вопросы считаются отдельно.</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def table(do: dict, posle: dict | None) -> str:
    rows_do, rows_posle = blocks(do), blocks(posle) if posle else [None] * 7
    lines = ["| Показатель | Что считается | Как было | Подробности | Что стало |",
             "|---|---|---|---|---|"]
    for b, a in zip(rows_do, rows_posle):
        stalo = f"{headline(a)} {a['подпись']}; {a['ниже']}" if a else NOT_MEASURED
        lines.append(f"| {b['код']} {b['имя']} | {b['что']} | {headline(b)} {b['подпись']} "
                     f"| {b['ниже']} | {stalo} |")
    return "\n".join(lines)


def write_table(doc: str, text: str) -> None:
    body = open(doc, encoding="utf-8").read()
    pattern = re.compile(re.escape(TABLE_START) + ".*?" + re.escape(TABLE_END), re.S)
    if not pattern.search(body):
        raise SystemExit(f"в {doc} нет меток таблицы {TABLE_START} ... {TABLE_END}")
    body = pattern.sub(lambda _: f"{TABLE_START}\n{text}\n{TABLE_END}", body)
    with open(doc, "w", encoding="utf-8") as f:
        f.write(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--do", required=True)
    ap.add_argument("--posle")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "img", "nabor-do.svg"))
    ap.add_argument("--doc", default=DOC)
    args = ap.parse_args()
    do = json.load(open(args.do, encoding="utf-8"))
    posle = json.load(open(args.posle, encoding="utf-8")) if args.posle else None
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg(do, posle))
    write_table(args.doc, table(do, posle))
    print(f"записано: {os.path.relpath(args.out, ROOT)}, таблица в {os.path.relpath(args.doc, ROOT)}")


if __name__ == "__main__":
    main()
