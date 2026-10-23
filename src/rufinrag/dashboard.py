"""Статическая страница по журналу запросов: те же модели витрины в разметке.

Своих запросов здесь нет ни одного. На вход приходит вывод моделей витрины в
том виде, в каком его отдает `warehouse.report`, то есть заголовок и строки
мартов, а страница их печатает. Поэтому число на странице это ячейка марта, а
не вычисление рядом с мартом, которое разошлось бы с ним на первой правке
модели.

Почему статика и почему свои графики. Страницу перерисовывает ночной прогон и
кладет на GitHub Pages, то есть ни сервера, ни ClickHouse, ни внешнего сервиса
за ней нет: выживает она ровно столько, сколько живет репозиторий. По той же
причине графики это inline SVG своим кодом: matplotlib стоил бы десятков
мегабайт в прогоне ради четырех картинок, а картинка вместо разметки потеряла
бы темную тему и выделение текста. Внешних запросов у страницы ноль: ни
шрифтов, ни библиотек, ни счетчиков.

Значения журнала попадают в разметку только через `safe`. Текст запроса в
открытый сервис присылает кто угодно, в нем бывают и `<script>`, и нулевые
байты (QUALITY, раздел 3), и показать их страница обязана как данные.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Сутки на странице. Журнал растет годами, а страница отвечает на вопрос «что
# происходит сейчас»: без предела в ней через год стояло бы 365 строк и график
# шириной в пиксель на сутки. Сколько суток осталось за кадром, страница
# говорит рядом.
DAYS_SHOWN = 14

# Управляющие символы в разметке: браузер их не покажет, а в выводе страницы
# они ломают и разбор, и поиск по тексту. Поэтому они заменяются видимым
# знаком, а не выбрасываются молча: пропажа символа это потеря данных.
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

COLUMNS = {
    "day": "Сутки",
    "queries": "Запросов",
    "answered": "Ответов",
    "rejected": "Отказов",
    "unavailable": "Сервис не готов",
    "queries_total": "Всего накоплено",
    "queries_7d": "За 7 суток",
    "queries_change": "Разница с прошлыми",
    "days_in_7d": "Суток в окне",
    "stage": "Ступень",
    "runs": "Прогонов",
    "p50_ms": "p50, мс",
    "p95_ms": "p95, мс",
    "max_ms": "Худший, мс",
    "rated": "С оценкой",
    "up": "Вверх",
    "down": "Вниз",
    "p95_total_ms": "p95 ответа, мс",
    "answered_7d": "Ответов за 7 суток",
    "rated_7d": "С оценкой за 7 суток",
    "up_7d": "Вверх за 7 суток",
    "p95_worst_7d": "Худший p95 за 7 суток",
    "first_time": "Впервые",
    "repeated": "Повторно",
    "distinct_texts": "Разных текстов",
    "median_gap_seconds": "Медиана паузы, с",
    "feedback_events": "Оценок",
    "matched": "Нашли запрос",
    "orphans": "Сирот",
}

NO_VALUE = "нет"


def safe(value: object) -> str:
    return html.escape(CONTROL.sub("�", str(value)), quote=True)


def number(text: str) -> str:
    """Число витрины так, как оно пишется в русском тексте.

    Пустое значение ClickHouse (`\\N`) это не ноль: у первых суток нет
    предыдущих, и разницы с ними не существует. Ноль здесь означал бы
    «трафик не изменился», чего никто не измерял.
    """
    if text in ("", "\\N", "nan"):
        return NO_VALUE
    try:
        value = float(text)
    except ValueError:
        return text
    if value.is_integer() and "." not in text:
        return f"{int(value):,}".replace(",", " ") if abs(value) >= 10_000 else text
    return text.replace(".", ",")


@dataclass(frozen=True)
class Value:
    """Число, которое напечатано на странице, и модель витрины, откуда оно."""

    label: str
    text: str
    source: str


@dataclass(frozen=True)
class Snapshot:
    """Все, что уходит на страницу. Собирает его `scripts/build_dashboard.py`."""

    mart: dict[str, dict]
    journal: dict
    corpus_version: str
    corpus_source: str
    config_version: str
    source: str
    built_at: str
    live: bool
    regulation: dict | None = None

    @property
    def empty(self) -> bool:
        return not self.mart or not self.journal.get("lines")


@dataclass
class Rendered:
    page: str
    values: list[Value] = field(default_factory=list)
    charts: int = 0

    @property
    def numbers(self) -> int:
        return len(self.values)


class _Builder:
    """Разметка и список напечатанных чисел собираются одним проходом.

    Список нужен не для красоты: по нему `scripts/verify_numbers.py`
    пересчитывает страницу, то есть проверяет, что каждое напечатанное число
    это ячейка витрины, а не текст в шаблоне.
    """

    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot
        self.values: list[Value] = []
        self.charts = 0

    def put(self, label: str, text: str, source: str) -> str:
        self.values.append(Value(label, text, source))
        return safe(text)

    def rows(self, model: str) -> list[dict[str, str]]:
        return self.snapshot.mart.get(model, {}).get("rows", [])

    def columns(self, model: str) -> list[str]:
        return self.snapshot.mart.get(model, {}).get("columns", [])

    def tail(self, model: str) -> tuple[list[dict[str, str]], int]:
        """Последние сутки марта и сколько их всего."""
        rows = self.rows(model)
        return rows[-DAYS_SHOWN:], len(rows)

    def table(self, model: str, rows: list[dict[str, str]]) -> str:
        columns = self.columns(model)
        head = "".join(f"<th>{safe(COLUMNS.get(name, name))}</th>" for name in columns)
        body = []
        for row in rows:
            key = row.get("day") or row.get("stage") or ""
            cells = []
            for name in columns:
                raw = row.get(name, "")
                if name in ("day", "stage"):
                    cells.append(f'<th scope="row">{safe(raw)}</th>')
                    continue
                marked = self.put(f"{model}.{name} {key}".strip(), number(raw), model)
                cells.append(f"<td>{marked}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return (
            '<div class="scroll"><table><thead><tr>'
            + head
            + "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table></div>"
        )

    def note(self, shown: int, total: int) -> str:
        if total <= shown or not total:
            return ""
        return f'<p class="note">Показаны последние {shown} суток из {total}.</p>'


def legend(items: list[tuple[str, str]]) -> str:
    """Подпись к цветам. Рисуется там же, где график: цвет без подписи это не
    график, а украшение."""
    keys = "".join(
        f'<span class="key {css}"></span>{safe(name)}' for name, css in dict.fromkeys(items)
    )
    return f'<p class="legend">{keys}</p>'


def _axis_max(values: list[float]) -> float:
    top = max(values, default=0.0)
    return top if top > 0 else 1.0


def vbars(labels: list[str], series: list[tuple[str, str, list[float]]], unit: str) -> str:
    """Столбцы по суткам, сложенные по составу. Ширина считается от числа суток."""
    totals = [sum(part[2][i] for part in series) for i in range(len(labels))]
    top = _axis_max(totals)
    step, gap, height, pad = 44, 10, 150, 26
    # Предел ширины в стиле, а не в атрибуте: иначе страница растянула бы
    # viewBox по ширине колонки, и график на двух сутках стал бы высотой в
    # две тысячи пикселей.
    width = max(len(labels) * step, 320)
    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height + pad * 2}" role="img" '
        f'style="max-width: {width}px" preserveAspectRatio="xMinYMid meet">',
        f'<line class="axis" x1="0" y1="{height + pad}" x2="{width}" y2="{height + pad}" />',
        f'<text class="tick" x="2" y="{pad - 8}">{safe(number(f"{top:.0f}"))} {safe(unit)}</text>',
    ]
    for index, label in enumerate(labels):
        bottom = height + pad
        for name, css, values in series:
            size = values[index] / top * height
            if size <= 0:
                continue
            out.append(
                f'<rect class="bar {css}" x="{index * step + gap / 2:.1f}" '
                f'y="{bottom - size:.1f}" width="{step - gap}" height="{size:.1f}">'
                f"<title>{safe(label)}: {safe(name)} {safe(number(f'{values[index]:g}'))}"
                f"</title></rect>"
            )
            bottom -= size
        out.append(
            f'<text class="tick" x="{index * step + step / 2:.1f}" '
            f'y="{height + pad * 2 - 6}" text-anchor="middle">{safe(label[5:])}</text>'
        )
    out.append("</svg>")
    out.append(legend([(name, css) for name, css, _ in series]))
    return "".join(out)


def hbars(rows: list[tuple[str, list[tuple[str, str, float]]]], unit: str) -> str:
    """Полосы по строкам: на строку приходится столько полос, сколько замеров."""
    top = _axis_max([value for _, bars in rows for _, _, value in bars])
    label_width, bar, gap, width = 96, 13, 4, 640
    track = width - label_width - 80
    height = sum((bar + gap) * len(bars) + gap for _, bars in rows)
    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'style="max-width: {width}px" preserveAspectRatio="xMinYMid meet">'
    ]
    y = gap
    for label, bars in rows:
        out.append(
            f'<text class="tick" x="0" y="{y + bar * len(bars) / 2 + 4:.1f}">{safe(label)}</text>'
        )
        for name, css, value in bars:
            size = value / top * track
            out.append(
                f'<rect class="bar {css}" x="{label_width}" y="{y}" '
                f'width="{max(size, 1):.1f}" height="{bar}">'
                f"<title>{safe(label)}: {safe(name)}</title></rect>"
                f'<text class="tick" x="{label_width + max(size, 1) + 6:.1f}" '
                f'y="{y + bar - 2}">{safe(number(f"{value:g}"))} {safe(unit)}</text>'
            )
            y += bar + gap
        y += gap
    out.append("</svg>")
    out.append(legend([(name, css) for _, bars in rows[:1] for name, css, _ in bars]))
    return "".join(out)


def ratio_rows(rows: list[tuple[str, float, float, str]]) -> str:
    """Доля с знаменателем рядом: заполненная часть полосы и подпись «N из M»."""
    label_width, bar, gap, width = 96, 15, 8, 640
    track = width - label_width - 110
    out = [
        f'<svg class="chart" viewBox="0 0 {width} {(bar + gap) * len(rows) + gap}" role="img" '
        f'style="max-width: {width}px" preserveAspectRatio="xMinYMid meet">'
    ]
    y = gap
    for label, part, whole, caption in rows:
        size = track * (part / whole if whole else 0)
        out.append(
            f'<text class="tick" x="0" y="{y + bar - 3}">{safe(label)}</text>'
            f'<rect class="track" x="{label_width}" y="{y}" width="{track}" height="{bar}" />'
            f'<rect class="bar rated" x="{label_width}" y="{y}" '
            f'width="{size:.1f}" height="{bar}"><title>{safe(caption)}</title></rect>'
            f'<text class="tick" x="{label_width + track + 8}" y="{y + bar - 3}">'
            f"{safe(caption)}</text>"
        )
        y += bar + gap
    out.append("</svg>")
    return "".join(out)


def _floats(rows: list[dict[str, str]], column: str) -> list[float]:
    out = []
    for row in rows:
        try:
            out.append(float(row.get(column, "0") or 0))
        except ValueError:
            out.append(0.0)
    return out


def _state(builder: _Builder) -> str:
    snapshot = builder.snapshot
    journal = snapshot.journal
    tiles = [
        ("Версия корпуса", safe(snapshot.corpus_version), safe(snapshot.corpus_source)),
        ("Связка по умолчанию", safe(snapshot.config_version), "configs/retrieval.yaml"),
        ("Страница собрана", safe(snapshot.built_at), "время прогона, UTC"),
        (
            "Последнее событие",
            safe(journal.get("last_event_at") or "событий нет"),
            "последняя строка журнала",
        ),
        (
            "Строк в журнале",
            builder.put("journal.lines", number(str(journal.get("lines", 0))), "journal"),
            f"суток {safe(journal.get('days', 0))}",
        ),
    ]
    cards = "".join(
        f'<div class="tile"><div class="tile-label">{safe(label)}</div>'
        f'<div class="tile-value">{value}</div>'
        f'<div class="tile-source">{source}</div></div>'
        for label, value, source in tiles
    )
    return f'<section><h2>Состояние</h2><div class="tiles">{cards}</div></section>'


def _traffic(builder: _Builder) -> str:
    rows, total = builder.tail("traffic_by_day")
    if not rows:
        return ""
    builder.charts += 1
    chart = vbars(
        [row["day"] for row in rows],
        [
            ("ответов", "answered", _floats(rows, "answered")),
            ("отказов", "rejected", _floats(rows, "rejected")),
            ("сервис не готов", "unavailable", _floats(rows, "unavailable")),
        ],
        "запросов",
    )
    return (
        "<section><h2>Трафик по суткам</h2>"
        "<p>Столбец это сутки UTC, состав столбца это исход запроса. Рядом с суточным "
        "числом стоит недельное и число суток, которые в недельное окно попали: окно идет "
        "по номеру суток, а сутки без запросов строки не создают.</p>"
        f"{chart}{builder.table('traffic_by_day', rows)}{builder.note(len(rows), total)}"
        "<p class=\"note\">Модель витрины <code>traffic_by_day</code>.</p></section>"
    )


def _latency(builder: _Builder) -> str:
    rows = builder.rows("stage_latency")
    if not rows:
        return ""
    builder.charts += 1
    chart = hbars(
        [
            (
                row["stage"],
                [
                    ("p50", "p50", float(row["p50_ms"])),
                    ("p95", "p95", float(row["p95_ms"])),
                ],
            )
            for row in rows
        ],
        "мс",
    )
    return (
        "<section><h2>Задержка по ступеням</h2>"
        "<p>Ступени приходят из события запроса, а не из колонок витрины: у лексической "
        "связки нет плотной ступени, у связки с переранжированием есть своя. Квантили "
        "точные, а не приближенные, иначе два режима хранения расходились бы на одних "
        "данных.</p>"
        f"{chart}{builder.table('stage_latency', rows)}"
        "<p class=\"note\">Модель витрины <code>stage_latency</code>.</p></section>"
    )


def _rated(builder: _Builder) -> str:
    rows, total = builder.tail("rated_share")
    if not rows:
        return ""
    builder.charts += 1
    captions = []
    for row in rows:
        caption = f"{number(row['rated'])} из {number(row['answered'])}"
        builder.put(f"rated_share.share {row['day']}", caption, "rated_share")
        captions.append((row["day"], float(row["rated"]), float(row["answered"]), caption))
    chart = ratio_rows(captions)
    match = builder.rows("feedback_match")
    orphans = ""
    if match:
        row = match[0]
        text = f"{number(row['orphans'])} из {number(row['feedback_events'])}"
        marked = builder.put("feedback_match.orphans", text, "feedback_match")
        orphans = (
            f"<p>Оценок без своего запроса {marked}. Это не мусор: сервис не проверяет, "
            "существует ли идентификатор, и такая строка приходит и от опечатки клиента, и "
            "от оценки запроса, чьи сутки уже уехали в другой файл.</p>"
            f"{builder.table('feedback_match', match)}"
        )
    return (
        "<section><h2>Запросы с оценкой</h2>"
        "<p>Доля всегда со знаменателем: сколько отвеченных запросов за сутки получили "
        "хотя бы одну оценку. Нажатий на один запрос бывает несколько, и знаменатель от "
        "них не растет.</p>"
        f"{chart}{builder.table('quality_by_day', builder.tail('quality_by_day')[0])}"
        f"{builder.note(len(rows), total)}{orphans}"
        "<p class=\"note\">Модели витрины <code>rated_share</code>, "
        "<code>quality_by_day</code> и <code>feedback_match</code>.</p></section>"
    )


def _repeats(builder: _Builder) -> str:
    rows, total = builder.tail("repeat_queries")
    if not rows:
        return ""
    builder.charts += 1
    chart = vbars(
        [row["day"] for row in rows],
        [
            ("впервые", "first", _floats(rows, "first_time")),
            ("повторно", "repeat", _floats(rows, "repeated")),
        ],
        "запросов",
    )
    return (
        "<section><h2>Повторы запросов</h2>"
        "<p>Место запроса в ряду повторов своего текста считается по строке журнала, до "
        "любой группировки. Текстов запросов на странице нет: витрина отвечает числами, а "
        "сами тексты лежат в журнале в репозитории.</p>"
        f"{chart}{builder.table('repeat_queries', rows)}{builder.note(len(rows), total)}"
        "<p class=\"note\">Модель витрины <code>repeat_queries</code>.</p></section>"
    )


def _no_data(builder: _Builder) -> str:
    where = "журнал в репозитории пуст" if builder.snapshot.live else "событий нет"
    return (
        "<section><h2>Данных пока нет</h2>"
        f"<p>Числа на этой странице считаются по журналу запросов сервиса, а {safe(where)}. "
        "События попадают в репозиторий ночным прогоном, который забирает их у живого "
        "процесса на хостинге, поэтому первая строка появляется здесь после первой "
        "выгрузки. Выдуманных чисел страница вместо них не показывает.</p>"
        "<p>Как страница выглядит с числами, видно на собранном наборе событий: "
        "<code>python scripts/build_dashboard.py --sample</code>.</p></section>"
    )


def _banner(snapshot: Snapshot) -> str:
    if snapshot.live:
        return ""
    return (
        '<p class="banner">Числа сняты на собранном наборе событий, а не на трафике '
        "сервиса: живого трафика сервис еще не видел. Схема события и модели витрины при "
        "этом настоящие.</p>"
    )


def _regulation(snapshot: Snapshot) -> str:
    stamp = (snapshot.regulation or {}).get("last_success")
    if not stamp:
        return ""
    runs = (snapshot.regulation or {}).get("runs")
    return (
        f"<p>Последний успешный прогон регламента {safe(stamp)}, всего прогонов "
        f"{safe(runs)} (<code>results/regulation.json</code>).</p>"
    )


STYLE = """
:root {
  --bg: #ffffff; --fg: #1b1d20; --muted: #5d646d; --line: #d9dde2;
  --card: #f6f7f9; --answered: #2f6f9f; --rejected: #c08a2e; --unavailable: #a4443c;
  --p50: #2f6f9f; --p95: #6f4f9f; --first: #2f6f9f; --repeat: #4f8f6f;
  --rated: #2f6f9f; --track: #dfe3e8;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #15181c; --fg: #e6e8ea; --muted: #9aa3ad; --line: #2d333a;
    --card: #1d2227; --answered: #6ba7d6; --rejected: #d8ae63; --unavailable: #d97a71;
    --p50: #6ba7d6; --p95: #a78bd0; --first: #6ba7d6; --repeat: #6fb48f;
    --rated: #6ba7d6; --track: #2d333a;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0 auto; padding: 24px 16px 56px; max-width: 900px; background: var(--bg);
  color: var(--fg); line-height: 1.5;
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 32px 0 8px; }
p { margin: 8px 0; }
a { color: inherit; }
code {
  font-size: 13px; background: var(--card); padding: 1px 4px; border-radius: 3px;
  /* Без лигатур: моноширинный шрифт склеивает два дефиса ключа командной строки
     в длинное тире, которого в текстах проекта нет. */
  font-variant-ligatures: none;
}
.lead, .note, .legend, .tile-source, .tile-label { color: var(--muted); font-size: 13px; }
.legend .key {
  display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin: 0 5px 0 14px; vertical-align: baseline;
}
.legend .key:first-child { margin-left: 0; }
.key.answered, .key.p50, .key.first, .key.rated { background: var(--answered); }
.key.rejected { background: var(--rejected); }
.key.unavailable { background: var(--unavailable); }
.key.p95 { background: var(--p95); }
.key.repeat { background: var(--repeat); }
.banner { border-left: 3px solid var(--rejected); padding: 8px 12px; background: var(--card); }
.tiles { display: flex; flex-wrap: wrap; gap: 8px; }
.tile {
  flex: 1 1 180px; background: var(--card); border: 1px solid var(--line);
  border-radius: 6px; padding: 10px 12px;
}
.tile-value { font-size: 16px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0; }
th, td { border-bottom: 1px solid var(--line); padding: 5px 8px; text-align: right; }
thead th { color: var(--muted); font-weight: 600; }
tbody th { text-align: left; font-weight: 600; white-space: nowrap; }
td { font-variant-numeric: tabular-nums; }
.chart { width: 100%; height: auto; max-width: 100%; margin: 8px 0; }
.chart .axis { stroke: var(--line); stroke-width: 1; }
.chart .tick { fill: var(--muted); font-size: 11px; }
.chart .track { fill: var(--track); }
.bar.answered, .bar.p50, .bar.first, .bar.rated { fill: var(--answered); }
.bar.rejected { fill: var(--rejected); }
.bar.unavailable { fill: var(--unavailable); }
.bar.p95 { fill: var(--p95); }
.bar.repeat { fill: var(--repeat); }
footer { margin-top: 40px; border-top: 1px solid var(--line); padding-top: 12px; }
""".strip()


def render(snapshot: Snapshot) -> Rendered:
    builder = _Builder(snapshot)
    if snapshot.empty:
        body = _no_data(builder)
    else:
        body = _traffic(builder) + _latency(builder) + _rated(builder) + _repeats(builder)
    page = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ru-fin-rag: журнал запросов</title>
<style>{STYLE}</style>
</head>
<body>
<h1>ru-fin-rag: журнал запросов</h1>
<p class="lead">Страницу перерисовывает ночной прогон
(<code>.github/workflows/nightly.yml</code>) по моделям витрины из
<code>dbt/models/marts</code>. Числа взяты из вывода этих моделей и лежат рядом машинным
видом: <a href="mart.json">mart.json</a>. Источник событий: {safe(snapshot.source)}.</p>
{_banner(snapshot)}
{_state(builder)}
{body}
<footer>
<p>Сервис: <a href="https://ru-fin-rag.onrender.com">ru-fin-rag.onrender.com</a>. Код и
отчет с числами: <a href="https://github.com/nata2627/ru-fin-rag">github.com/nata2627/ru-fin-rag</a>.</p>
{_regulation(snapshot)}
<p class="note">Страница статическая: ни сервера, ни базы, ни внешних запросов за ней нет.
Собирает ее <code>scripts/build_dashboard.py</code>, графики это inline SVG того же кода.</p>
</footer>
</body>
</html>
"""
    return Rendered(page, builder.values, builder.charts)


def journal_summary(directory: Path, pattern: str = "queries-*.jsonl") -> dict:
    """Что лежит в журнале: файлы, сутки, строки и момент последнего события.

    Момент берется из последней строки каждого файла, а не из имени файла:
    имя это сутки, а на странице стоит время, и оно обязано быть временем
    события, а не временем сборки страницы (AGENTS.md, раздел 2).
    """
    files = sorted(directory.glob(pattern)) if directory.is_dir() else []
    lines = 0
    last = ""
    for path in files:
        text = path.read_text(encoding="utf-8")
        rows = [line for line in text.splitlines() if line.strip()]
        lines += len(rows)
        if rows:
            last = max(last, json.loads(rows[-1]).get("received_at", ""))
    return {
        "files": [path.name for path in files],
        "days": len(files),
        "lines": lines,
        "last_event_at": last,
    }
