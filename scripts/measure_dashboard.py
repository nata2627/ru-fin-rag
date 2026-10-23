"""Замер сборки дашборда: что страница печатает и чего это стоит.

Два состояния, оба штатные. На собранном наборе событий страница показывает
числа витрины, на пустом журнале говорит, что данных нет, и не падает. Пустой
журнал это не крайний случай: до первой выгрузки с хостинга он пуст всегда.

Числа этого замера идут в README, поэтому живой журнал репозитория здесь не
меряется: он меняется каждой ночью, и напечатанное число ломало бы проверку на
следующем же прогоне. Состояние живой сборки видно на самой странице (время
сборки и момент последнего события) и в `site/mart.json` рядом с ней.

Заодно замер сверяет вывод витрины с `results/warehouse.json`: набор событий у
них один, и если числа разошлись, значит разошлись модели или их исполнитель, а
не страница.

Запуск: python scripts/measure_dashboard.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag import dashboard  # noqa: E402

WAREHOUSE = ROOT / "results/warehouse.json"

# Ссылка на чужую машину в разметке: шрифт, библиотека, картинка, счетчик.
# Таких в странице быть не должно ни одной, иначе страница переживет репозиторий
# ровно до того дня, когда чужая машина ответит иначе.
EXTERNAL = re.compile(
    r"<(?:script|link|img|iframe|object|embed)\b[^>]*?" r"(?:src|href)\s*=\s*[\"'](?:https?:)?//",
    re.IGNORECASE,
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(builder, directory: Path, out: Path, *, live: bool, source: str) -> dict:
    report = builder.build(directory, out, live=live, source=source)
    page = (out / "index.html").read_text(encoding="utf-8")
    report["page_kb"] = round(report["page_bytes"] / 1024, 1)
    report["external_resources"] = len(EXTERNAL.findall(page))
    # Страницу замер собирает в каталоге процесса, и путь к нему в отчет не
    # идет: он у каждого прогона свой, а отчет лежит в репозитории.
    report.pop("page")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results/dashboard.json")
    args = parser.parse_args()

    builder = load(ROOT / "scripts/build_dashboard.py", "build_dashboard")
    events = load(ROOT / "scripts/compare_warehouses.py", "compare_warehouses")

    with tempfile.TemporaryDirectory(prefix="rufinrag-dashboard-") as temporary:
        root = Path(temporary)
        journal = root / "journal"
        journal.mkdir()
        events.write_sample(journal, events.sample_events())
        sample = measure(
            builder, journal, root / "sample", live=False, source=builder.SAMPLE_SOURCE
        )
        empty = measure(
            builder,
            root / "empty-journal",
            root / "empty",
            live=True,
            source=builder.JOURNAL_SOURCE,
        )
        page = (root / "sample/index.html").read_text(encoding="utf-8")
        mart = json.loads((root / "sample/mart.json").read_text(encoding="utf-8"))

    known = json.loads(WAREHOUSE.read_text(encoding="utf-8"))["mart"]
    report = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "builds": {"sample": sample, "empty": empty},
        "mart_matches_warehouse_json": mart == known,
        "days_shown": dashboard.DAYS_SHOWN,
        # Снимок входа сборки на собранном наборе: по нему `verify_numbers`
        # пересобирает страницу и сверяет каждое напечатанное число, не
        # поднимая движок витрины.
        "snapshot": {
            "journal": sample["journal"],
            "corpus_version": sample["corpus_version"],
            "corpus_source": builder.corpus_version()[1],
            "config_version": sample["config_version"],
            "source": sample["source"],
            "built_at": sample["built_at"],
        },
        "values": sample.pop("values"),
    }
    empty.pop("values")
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "values"}, ensure_ascii=False))
    if not report["mart_matches_warehouse_json"]:
        print(
            "вывод витрины на собранном наборе разошелся с results/warehouse.json",
            file=sys.stderr,
        )
        return 1
    if sample["external_resources"] or empty["external_resources"]:
        print("в странице есть ссылка на внешний ресурс", file=sys.stderr)
        return 1
    if len(page.encode("utf-8")) != sample["page_bytes"]:
        print("страница на диске не совпала с измеренной", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
