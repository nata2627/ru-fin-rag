#!/usr/bin/env python3
"""Картинки README, собранные из файлов результатов, и манифест их чисел.

Число на картинке ничем не лучше числа в тексте: оно так же устаревает и так же
может оказаться набранным руками. Поэтому каждое число картинки берется из
файла результата и записывается в манифест `docs/img/readme/figures.json`
вместе с путем к файлу и ключом. Функция `stale()` пересчитывает манифест и
называет расхождения: поменялся замер, а картинку не перерисовали.

Источники только замороженные. Файлы, которые переписывает прогон по
расписанию, сюда не идут: бот картинку не перерисует, и сверка покраснела бы
ночью без участия человека. Это держит тест `tests/test_readme_figures.py`.

Счет чисел (`figures_data`) от matplotlib не зависит, библиотека импортируется
только при отрисовке, поэтому проверке в CI ставить ее не нужно.

Запуск: python scripts/readme_figures.py      (нужна группа зависимостей figures)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img" / "readme"
MANIFEST = OUT / "figures.json"
FOOTER = "ru-fin-rag · github.com/nata2627/ru-fin-rag"

INK = "#1b1b1b"
NAVY = "#1f3355"
MUTED = "#6b6b6b"
ACCENT = "#f2542d"
GOOD = "#2e8b57"
BAR = "#d9d6cf"
TILE = "#eeece7"
GRID = "#e6e6e6"


def ru(value: float, digits: int = 3, sign: bool = False) -> str:
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text.replace(".", ",").replace("-", "−")


def spaced(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def number(source: str, *path: str | int) -> dict:
    """Значение из файла результата вместе с тем, откуда оно взято.

    Путь пишется через косую черту: с точками «rufin.Recall@50.mean» поиск
    секретов принимает за адрес почты.
    """
    value = json.loads((ROOT / source).read_text(encoding="utf-8"))
    for step in path:
        value = value[step]
    return {"value": value, "source": source, "key": "/".join(str(p) for p in path)}


def interval(source: str, *path: str | int, names: tuple[str, str, str]) -> dict:
    """Среднее и границы интервала, которые лежат рядом под одним ключом."""
    mean, lo, hi = names
    return {
        "value": number(source, *path, mean)["value"],
        "lo": number(source, *path, lo)["value"],
        "hi": number(source, *path, hi)["value"],
        "source": source,
        "key": "/".join(str(p) for p in path),
    }


def _stemmer_row() -> int:
    """Номер строки стеммера на всех запросах: в файле это список без ключей."""
    rows = json.loads(
        (ROOT / "retrieval/data/results/morphology.json").read_text(encoding="utf-8")
    )["прибавка"]
    for i, row in enumerate(rows):
        if (row["вариант"], row["против"], row["подвыборка"]) == ("стеммер", "как есть", "все"):
            return i
    raise SystemExit("в morphology.json нет строки стеммера на всех запросах")


def _key_numbers() -> dict:
    rehearsal = "results/render-rehearsal.json"
    version = number(rehearsal, "health", "corpus_version")
    acts = int(re.fullmatch(r"chunks\d+-acts(\d+)", version["value"]).group(1))
    chunks = number(rehearsal, "health", "chunks")
    best = number(
        "results/rerank-quality.json", "runs", "hybrid-rerank, связка сервиса", "ndcg10", "mean"
    )
    bm25 = number(
        "results/rerank-quality-bm25.json", "runs", "bm25, связка сервиса", "ndcg10", "mean"
    )
    queries = number("results/rerank-quality.json", "queries_scored")
    passed = "results/acceptance-passed.json"
    runs = number(passed, "scheduled_runs")
    green = number(passed, "scheduled_runs_green")
    days = number(passed, "days_required")
    peak = number(rehearsal, "peak_memory_mb")
    limit = number(rehearsal, "memory_limit")
    limit_mb = int(limit["value"].rstrip("m"))
    return {
        "tiles": [
            {
                "big": spaced(chunks["value"]),
                "text": f"фрагмента из {acts} актов\nБанка России в наборе",
                "numbers": [chunks, {**version, "value": acts}],
            },
            {
                "big": ru(best["value"]),
                "text": (
                    f"NDCG@10 лучшей связки\nна {queries['value']} запросе, "
                    f"у BM25 {ru(bm25['value'])}"
                ),
                "numbers": [best, bm25, queries],
            },
            {
                "big": f"{green['value']} из {runs['value']}",
                "text": f"ночных прогонов зеленые,\n{days['value']} суток без рук",
                "numbers": [green, runs, days],
            },
            {
                "big": f"{ru(peak['value'], 0)} МБ",
                "text": f"пик памяти сервиса при\n{limit_mb} МБ бесплатного тарифа",
                "numbers": [peak, {**limit, "value": limit_mb}],
            },
        ]
    }


def _latency(config: str) -> dict:
    return number("results/latency-local.json", "configs", config, "p50_ms")


def _latency_text(ms: float) -> str:
    if ms >= 1000:
        return f"{ru(ms / 1000, 1)} с"
    return f"{ru(ms, 2 if ms < 10 else 1)} мс"


def _quality() -> dict:
    base = "retrieval/data/results/base.json"
    names = ("mean", "lo", "hi")
    hybrid_rerank = ("runs", "hybrid-rerank, связка сервиса", "ndcg10")
    bm25_rerank = ("runs", "bm25-rerank, связка сервиса", "ndcg10")
    spec = [
        (
            "hybrid-rerank",
            "гибрид + кросс-энкодер",
            interval("results/rerank-quality.json", *hybrid_rerank, names=names),
        ),
        (
            "bm25-rerank",
            "BM25 + кросс-энкодер",
            interval("results/rerank-quality-bm25.json", *bm25_rerank, names=names),
        ),
        (
            "hybrid",
            "гибрид, слияние по рангам",
            interval(base, "metrics", "hybrid-rufin", "NDCG@10", names=names),
        ),
        ("bm25", "BM25 со стеммером", interval(base, "metrics", "bm25", "NDCG@10", names=names)),
        (
            "dense",
            "плотный поиск",
            interval(base, "metrics", "dense-rufin", "NDCG@10", names=names),
        ),
    ]
    rows = []
    for config, label, ndcg in spec:
        latency = _latency(config)
        rows.append(
            {
                "config": config,
                "label": label,
                "ndcg10": ndcg,
                "p50_ms": latency,
                "text": ru(ndcg["value"]),
                "latency_text": _latency_text(latency["value"]),
            }
        )
    queries = number("results/rerank-quality.json", "queries_scored")
    ceiling = number(
        "retrieval/data/results/depth-ceiling.json",
        "groups",
        "все размеченные",
        "configs",
        "hybrid-rufin",
        "Recall@50",
        "mean",
    )
    slow = rows[0]["p50_ms"]["value"] / 1000
    return {
        "title": f"Кросс-энкодер дает лучшее качество, но стоит {ru(slow, 0)} секунды на запрос",
        "subtitle": (
            f"NDCG@10 на {queries['value']} размеченном запросе, линия: 95% интервал бутстрэпа; "
            "рядом p50 одного запроса на процессоре мака"
        ),
        "rows": rows,
        "queries": queries,
        "ceiling": {
            **ceiling,
            "text": (
                f"потолок кросс-энкодера:\nв первых 50 кандидатах\nгибрида лежит "
                f"{ru(ceiling['value'])}\nэталонных фрагментов\n(Recall@50, до стеммера)"
            ),
        },
    }


def _negative() -> dict:
    names = ("mean", "lo", "hi")
    gate = "retrieval/data/results/gate12.json"
    graph = "results/graph-expansion.json"
    stem = _stemmer_row()
    live = number(gate, "весь тест", "вопросов")
    queries = number("results/rerank-quality.json", "queries_scored")
    rows = [
        {
            "label": "кросс-энкодер поверх гибрида",
            "kept": True,
            "delta": interval("results/rerank-quality.json", "difference_ndcg10", names=names),
        },
        {
            "label": "стеммер в лексической ступени",
            "kept": True,
            "delta": interval(
                "retrieval/data/results/morphology.json",
                "прибавка",
                stem,
                "NDCG@10",
                names=("diff", "lo", "hi"),
            ),
        },
        {
            "label": f"дообученный переранжировщик\n({live['value']} живых вопроса)",
            "kept": False,
            "delta": interval(gate, "весь тест", "разница", names=("среднее", "lo", "hi")),
        },
        {
            "label": "граф ссылок: соседи в запас выдачи",
            "kept": False,
            "delta": interval(
                graph, "variants", "reserve", "difference_ndcg10", "hybrid", "all", names=names
            ),
        },
        {
            "label": "граф ссылок: слияние с выдачей",
            "kept": False,
            "delta": interval(
                graph, "variants", "fusion", "difference_ndcg10", "hybrid", "all", names=names
            ),
        },
    ]
    for row in rows:
        d = row["delta"]
        # Малая разница в трех знаках читается как ноль, поэтому ей четыре.
        digits = 4 if abs(d["value"]) < 0.02 else 3
        row["text"] = (
            f"{ru(d['value'], digits, sign=True)} "
            f"[{ru(d['lo'], digits, sign=True)}; {ru(d['hi'], digits, sign=True)}]"
        )
    rule = number(gate, "правило")
    threshold = float(re.search(r"не меньше \+([\d.]+)", rule["value"]).group(1))
    return {
        "title": "Что прибавило качество, а что проверено и в сервис не вошло",
        "subtitle": (
            f"Разница NDCG@10 с 95% интервалом парного бутстрэпа, {queries['value']} размеченный "
            "запрос, если не сказано иное"
        ),
        "rows": rows,
        "queries": queries,
        "live_queries": live,
        "threshold": {
            **rule,
            "value": threshold,
            "text": f"порог {ru(threshold, 3, sign=True)}, записан до обучения",
        },
    }


def _autonomy() -> dict:
    passed = "results/acceptance-passed.json"
    days = json.loads((ROOT / passed).read_text(encoding="utf-8"))["days"]
    runs = number(passed, "scheduled_runs")
    green = number(passed, "scheduled_runs_green")
    manual = number(passed, "manual_runs")
    bot = number(passed, "commits", "bot")
    required = number(passed, "days_required")
    return {
        "title": (
            f"{required['value']} суток без рук: {green['value']} плановых прогонов "
            f"из {runs['value']} зеленые"
        ),
        "subtitle": (
            f"Окно приемки автономности {_day(days[0]['date'])}-{_day(days[-1]['date'])}.2026, "
            f"квадрат это плановый прогон; коммитов бота {bot['value']}, "
            f"запусков руками {manual['value']}"
        ),
        "days": [
            {
                "date": d["date"],
                "runs": len(d["runs"]),
                "source": passed,
                "key": f"days.{i}.runs",
            }
            for i, d in enumerate(days)
        ],
        "numbers": [runs, green, manual, bot, required],
    }


def _day(iso: str) -> str:
    return f"{iso[8:10]}.{iso[5:7]}"


def _memory() -> dict:
    rehearsal = "results/render-rehearsal.json"
    limit = number(rehearsal, "memory_limit")
    limit_mb = int(limit["value"].rstrip("m"))
    qdrant = number("results/vector-store-container.json", "stores", "qdrant", "container_memory")
    qdrant_mib = float(qdrant["value"].removesuffix("MiB"))
    rows = [
        {
            "label": "сервис, лексический режим",
            "mb": number("results/memory.json", "configs", "bm25", "peak_mb"),
            "unit": "МБ",
        },
        {
            "label": f"он же в контейнере с потолком {limit_mb} МБ",
            "mb": number(rehearsal, "peak_memory_mb"),
            "unit": "МБ",
        },
        {
            "label": "контейнер Qdrant рядом с сервисом",
            # Docker печатает МиБ, столбец рисуется в тех же МБ, что и остальные.
            "mb": {**qdrant, "value": qdrant_mib},
            "unit": "МиБ",
            "plot_mb": round(qdrant_mib * 1.048576, 1),
        },
        {
            "label": "сервис с плотной ступенью",
            "mb": number("results/memory.json", "configs", "hybrid", "peak_mb"),
            "unit": "МБ",
        },
    ]
    for row in rows:
        row["fits"] = row.get("plot_mb", row["mb"]["value"]) <= limit_mb
        row["text"] = f"{ru(row['mb']['value'], 1)} {row['unit']}"
    return {
        "title": f"В бесплатные {limit_mb} МБ влезает только лексический режим",
        "subtitle": "Пик памяти процесса или контейнера, замеры 21-22.10.2026",
        "rows": rows,
        "limit": {**limit, "value": limit_mb, "text": f"тариф {limit_mb} МБ"},
    }


def figures_data() -> dict:
    """Все числа всех картинок. Ключ словаря это имя файла без расширения."""
    return {
        "key_numbers": _key_numbers(),
        "quality": _quality(),
        "negative": _negative(),
        "autonomy": _autonomy(),
        "memory": _memory(),
    }


def sources(data: dict | list) -> set[str]:
    """Все файлы, из которых взято хотя бы одно число."""
    found: set[str] = set()
    if isinstance(data, dict):
        if "source" in data:
            found.add(data["source"])
        for value in data.values():
            found |= sources(value)
    elif isinstance(data, list):
        for value in data:
            found |= sources(value)
    return found


def _flat(data, prefix: str = "") -> dict:
    if isinstance(data, dict):
        out = {}
        for key, value in data.items():
            out.update(_flat(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(data, list):
        out = {}
        for i, value in enumerate(data):
            out.update(_flat(value, f"{prefix}.{i}"))
        return out
    return {prefix: data}


def stale() -> list[str]:
    """Чем манифест на диске отличается от пересчета по файлам результатов."""
    if not MANIFEST.exists():
        return [f"нет манифеста {MANIFEST.relative_to(ROOT)}: запустить `make figures`"]
    on_disk = _flat(json.loads(MANIFEST.read_text(encoding="utf-8")))
    fresh = _flat(json.loads(json.dumps(figures_data(), ensure_ascii=False)))
    problems = []
    for key in sorted(on_disk.keys() | fresh.keys()):
        if on_disk.get(key) != fresh.get(key):
            problems.append(
                f"{key}: на картинке {on_disk.get(key)!r}, по замеру {fresh.get(key)!r}"
            )
    for name in fresh_names():
        if not (OUT / f"{name}.png").exists():
            problems.append(f"нет картинки docs/img/readme/{name}.png")
    return problems


def fresh_names() -> list[str]:
    return list(figures_data())


def _figure(plt, width: float, height: float):
    fig = plt.figure(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor("white")
    return fig


def _header(fig, title: str, subtitle: str, left: float = 0.04) -> None:
    height = fig.get_figheight()
    fig.text(left, 1 - 0.34 / height, title, fontsize=15.5, fontweight="bold", color=INK, va="top")
    fig.text(left, 1 - 0.78 / height, subtitle, fontsize=9.5, color=MUTED, va="top")
    fig.text(left, 0.16 / height, FOOTER, fontsize=7.5, color=MUTED, va="bottom")


def _clean(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, labelsize=9, colors=MUTED)
    ax.tick_params(axis="y", labelsize=10, colors=INK)


def _axes(fig, left: float, bottom_in: float, top_in: float, right: float = 0.96):
    height = fig.get_figheight()
    bottom = bottom_in / height
    return fig.add_axes([left, bottom, right - left, 1 - top_in / height - bottom])


def draw_key_numbers(plt, data: dict) -> None:
    from matplotlib.patches import FancyBboxPatch

    fig = _figure(plt, 10, 2.1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.1)
    ax.axis("off")
    gap, width = 0.22, (10 - 0.3 * 2 - 0.22 * 3) / 4
    for i, tile in enumerate(data["tiles"]):
        x = 0.3 + i * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.22),
                width,
                1.66,
                boxstyle="round,pad=0,rounding_size=0.1",
                linewidth=0,
                facecolor=TILE,
            )
        )
        ax.text(x + 0.2, 1.36, tile["big"], fontsize=21, fontweight="bold", color=NAVY, va="center")
        ax.text(
            x + 0.2, 0.68, tile["text"], fontsize=8.4, color=MUTED, va="center", linespacing=1.5
        )
    fig.savefig(OUT / "key_numbers.png", facecolor="white")
    plt.close(fig)


def draw_quality(plt, data: dict) -> None:
    fig = _figure(plt, 10, 4.3)
    _header(fig, data["title"], data["subtitle"])
    ax = _axes(fig, 0.26, 0.75, 1.3)
    rows = data["rows"][::-1]
    for y, row in enumerate(rows):
        n = row["ndcg10"]
        main = row["config"] == "hybrid-rerank"
        ax.barh(y, n["value"], height=0.6, color=ACCENT if main else BAR, zorder=2)
        ax.plot([n["lo"], n["hi"]], [y, y], color=INK if main else MUTED, linewidth=1.3, zorder=3)
        ax.text(
            n["lo"] - 0.012,
            y,
            row["text"],
            va="center",
            ha="right",
            fontsize=10.5,
            fontweight="bold",
            color="white" if main else INK,
            zorder=4,
        )
        ax.text(
            n["hi"] + 0.012,
            y,
            row["latency_text"],
            va="center",
            fontsize=9.5,
            color=INK if main else MUTED,
            zorder=4,
        )
    ceiling = data["ceiling"]
    ax.axvline(ceiling["value"], color=NAVY, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(
        ceiling["value"] + 0.014,
        2.0,
        ceiling["text"],
        fontsize=8,
        color=NAVY,
        va="center",
        linespacing=1.5,
    )
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0,25", "0,5", "0,75", "1"])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _clean(ax)
    fig.savefig(OUT / "quality.png", facecolor="white")
    plt.close(fig)


def draw_negative(plt, data: dict) -> None:
    fig = _figure(plt, 10, 4.3)
    _header(fig, data["title"], data["subtitle"])
    ax = _axes(fig, 0.3, 0.75, 1.3)
    rows = data["rows"][::-1]
    # Между двумя группами оставлен зазор под подпись группы.
    heights = [y + (0.75 if row["kept"] else 0) for y, row in enumerate(rows)]
    for y, row in zip(heights, rows, strict=True):
        d = row["delta"]
        color = GOOD if row["kept"] else ACCENT
        ax.plot(
            [d["lo"], d["hi"]], [y, y], color=color, linewidth=2.4, solid_capstyle="round", zorder=3
        )
        ax.plot(d["value"], y, "o", color=color, markersize=7, zorder=4)
        ax.text(0.19, y, row["text"], va="center", fontsize=9.5, color=INK)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=2)
    threshold = data["threshold"]
    ax.plot(
        [threshold["value"]] * 2,
        [1.62, 2.38],
        color=NAVY,
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.text(threshold["value"] + 0.006, 2, threshold["text"], fontsize=7.8, color=NAVY, va="center")
    group = {"fontsize": 8.5, "fontweight": "bold"}
    ax.text(0.19, heights[-1] + 0.5, "прибавило качество", color=GOOD, **group)
    ax.text(0.19, 2.5, "проверено и не вошло", color=ACCENT, **group)
    ax.set_yticks(heights, [r["label"] for r in rows])
    ax.set_xlim(-0.16, 0.335)
    ax.set_ylim(-0.6, heights[-1] + 0.9)
    ticks = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]
    ax.set_xticks(ticks, [ru(t, 2, sign=True) if t else "0" for t in ticks])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _clean(ax)
    fig.savefig(OUT / "negative.png", facecolor="white")
    plt.close(fig)


def draw_autonomy(plt, data: dict) -> None:
    from matplotlib.patches import FancyBboxPatch

    fig = _figure(plt, 10, 2.9)
    _header(fig, data["title"], data["subtitle"])
    ax = _axes(fig, 0.04, 0.5, 1.25)
    days = data["days"]
    ax.set_xlim(0, len(days))
    ax.set_ylim(-0.75, 2.2)
    ax.axis("off")
    for x, day in enumerate(days):
        for level in range(day["runs"]):
            ax.add_patch(
                FancyBboxPatch(
                    (x + 0.14, level * 1.08 + 0.05),
                    0.72,
                    0.9,
                    boxstyle="round,pad=0,rounding_size=0.08",
                    linewidth=0,
                    facecolor=GOOD,
                )
            )
        ax.text(
            x + 0.5, -0.4, _day(day["date"]), ha="center", va="center", fontsize=8.5, color=MUTED
        )
    fig.savefig(OUT / "autonomy.png", facecolor="white")
    plt.close(fig)


def draw_memory(plt, data: dict) -> None:
    fig = _figure(plt, 10, 3.7)
    _header(fig, data["title"], data["subtitle"])
    ax = _axes(fig, 0.33, 0.75, 1.3)
    rows = data["rows"][::-1]
    for y, row in enumerate(rows):
        value = row.get("plot_mb", row["mb"]["value"])
        ax.barh(y, value, height=0.6, color=GOOD if row["fits"] else BAR, zorder=2)
        ax.text(value + 18, y, row["text"], va="center", fontsize=10, color=INK)
    limit = data["limit"]
    ax.axvline(limit["value"], color=ACCENT, linewidth=1.4, zorder=3)
    ax.text(
        limit["value"] + 12,
        len(rows) - 0.45,
        limit["text"],
        fontsize=9,
        color=ACCENT,
        fontweight="bold",
        va="center",
    )
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.set_xlim(0, 1320)
    ax.set_ylim(-0.5, len(rows) - 0.2)
    ax.set_xticks([0, 250, 500, 750, 1000, 1250])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _clean(ax)
    fig.savefig(OUT / "memory.png", facecolor="white")
    plt.close(fig)


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    OUT.mkdir(parents=True, exist_ok=True)
    data = figures_data()
    draw_key_numbers(plt, data["key_numbers"])
    draw_quality(plt, data["quality"])
    draw_negative(plt, data["negative"])
    draw_autonomy(plt, data["autonomy"])
    draw_memory(plt, data["memory"])
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"картинок {len(data)}, манифест {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
