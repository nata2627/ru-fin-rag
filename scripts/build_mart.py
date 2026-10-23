"""Сборка витрины dbt в ClickHouse и сверка ее с бесплатным режимом.

Что делает прогон: заливает журнал в ClickHouse, собирает проект dbt
(`dbt build`, то есть модели и тесты одной командой), а потом сравнивает вывод
каждой собранной модели с тем, что по тем же моделям считает `chdb` поверх
файлов. Сравнение нужно затем, что модель лежит в одном месте, а исполнителей у
нее два: dbt на сервере и разворот в один запрос для бесплатного режима
(`src/rufinrag/warehouse.py`). Разойтись они могут молча.

Строки сравниваются множеством, а не текстом: внутри представления ClickHouse
`order by` соблюдать не обязан, и порядок строк, прочитанных из представления,
это не то свойство, которое модель обещает.

События здесь собранные, а не живые, как и в сверке режимов хранения: открытый
сервис трафика пока не видел. Собирает их тот же код, которым пишет события
сервис, поэтому схема настоящая, а числа трафика это числа собранного набора.

Запуск: python scripts/build_mart.py [--server URL] [--events DIR]
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag import warehouse  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "compare_warehouses", ROOT / "scripts/compare_warehouses.py"
)
compare_warehouses = importlib.util.module_from_spec(spec)
sys.modules["compare_warehouses"] = compare_warehouses
spec.loader.exec_module(compare_warehouses)

PROJECT = ROOT / "dbt"
# Корень замыкания зависимостей: от этих двух пакетов считается вес, который
# витрина добавляет окружению.
DBT_PACKAGES = ("dbt-core", "dbt-clickhouse")
REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def dbt_executable() -> Path:
    """`dbt` из того же окружения, что и запускающий python."""
    candidate = Path(sys.executable).parent / "dbt"
    if candidate.exists():
        return candidate
    raise SystemExit(f"нет {candidate}: поставить витрину в окружение, pip install -e '.[dbt]'")


def run_dbt(command: list[str], url: str) -> subprocess.CompletedProcess:
    host, _, port = url.split("//", 1)[1].partition(":")
    env = {
        **os.environ,
        "CLICKHOUSE_HOST": host,
        "CLICKHOUSE_PORT": port or "8123",
        # Телеметрия dbt в прогоне CI это только лишние секунды ожидания.
        "DO_NOT_TRACK": "1",
    }
    return subprocess.run(
        [
            str(dbt_executable()),
            *command,
            "--project-dir",
            str(PROJECT),
            "--profiles-dir",
            str(PROJECT),
            "--target",
            "build",
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def build_report(url: str) -> dict:
    """Итог `dbt build` по его собственному отчету, а не по разбору вывода."""
    started = time.monotonic()
    done = run_dbt(["build"], url)
    seconds = time.monotonic() - started
    print(done.stdout[-4000:])
    if done.stderr.strip():
        print(done.stderr[-2000:], file=sys.stderr)

    results = json.loads((PROJECT / "target/run_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((PROJECT / "target/manifest.json").read_text(encoding="utf-8"))
    nodes = manifest["nodes"]

    models: list[dict] = []
    tests: list[dict] = []
    for item in results["results"]:
        uid = item["unique_id"]
        node = nodes[uid]
        if uid.startswith("model."):
            models.append(
                {
                    "name": node["name"],
                    "layer": node["fqn"][1],
                    "materialized": node["config"]["materialized"],
                    "status": item["status"],
                }
            )
        elif uid.startswith("test."):
            generic = node.get("test_metadata", {}).get("name")
            tests.append(
                {
                    "name": node["name"],
                    # Типовой тест описан в schema.yml одной строкой, свой это
                    # запрос в dbt/tests. Их число считается отдельно: своих
                    # тестов ровно столько, сколько мест, которые ломались бы
                    # молча.
                    "kind": "generic" if generic else "singular",
                    "generic": generic,
                    "status": item["status"],
                }
            )

    statuses = [item["status"] for item in results["results"]]
    return {
        "returncode": done.returncode,
        "seconds": round(seconds, 1),
        "models": models,
        "tests": tests,
        "counts": {
            "models": len(models),
            "staging_models": sum(1 for m in models if m["layer"] == warehouse.STAGING),
            "mart_models": sum(1 for m in models if m["layer"] == warehouse.MARTS),
            "tests": len(tests),
            "generic_tests": sum(1 for t in tests if t["kind"] == "generic"),
            "singular_tests": sum(1 for t in tests if t["kind"] == "singular"),
            "passed": sum(1 for s in statuses if s in ("success", "pass")),
            "warned": sum(1 for s in statuses if s == "warn"),
            "failed": sum(1 for s in statuses if s in ("error", "fail", "runtime error")),
        },
    }


def footprint() -> dict:
    """Вес замыкания dbt на диске: что реально лежит в окружении.

    Считается по записям установленных пакетов, а не по размеру колес: колеса
    сжаты, а памяти и диску достается распакованное. Необязательные
    дополнения (`extra ==` в условии) в замыкание не берутся, их никто не
    ставил.
    """
    seen: dict[str, int] = {}
    queue = list(DBT_PACKAGES)
    while queue:
        name = queue.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        size = 0
        for file in dist.files or ():
            path = Path(dist.locate_file(file))
            if path.is_file():
                size += path.stat().st_size
        seen[key] = size
        for requirement in dist.requires or ():
            if "extra ==" in requirement:
                continue
            match = REQUIREMENT_NAME.match(requirement)
            if match:
                queue.append(match.group(0))
    return {
        "packages": len(seen),
        "disk_mb": round(sum(seen.values()) / 1024**2, 1),
        "versions": {name: metadata.version(name) for name in DBT_PACKAGES},
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
    }


def chdb_version() -> str:
    import chdb

    return str(chdb.query("SELECT version()", "TSV")).strip()


def rows_of(text: str) -> tuple[list[str], list[list[str]]]:
    header, body = warehouse.parse_table(text)
    return header, sorted(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server", default=os.environ.get("CLICKHOUSE_URL", "http://127.0.0.1:8123")
    )
    parser.add_argument("--events", type=Path, help="готовый каталог журнала вместо собранного")
    parser.add_argument("--out", type=Path, default=ROOT / "results/dbt.json")
    args = parser.parse_args()

    if args.events and not sorted(args.events.glob("queries-*.jsonl")):
        # Пустой журнал это не поломка: открытый сервис мог не получить ни
        # одного запроса за сутки. Собирать витрину не над чем, и прогон не
        # должен от этого краснеть.
        print(f"в {args.events} нет файлов журнала, витрину собирать не над чем")
        return 0

    server = warehouse.ServerWarehouse(args.server)
    if not server.available():
        print(
            f"нет ClickHouse на {args.server}: dbt работает только с сервером, "
            "поднять его командой make logs-up",
            file=sys.stderr,
        )
        return 1

    temporary = None
    if args.events:
        directory = args.events
        sample = {"source": str(directory)}
    else:
        temporary = tempfile.TemporaryDirectory(prefix="rufinrag-mart-")
        directory = Path(temporary.name)
        sample = compare_warehouses.write_sample(directory, compare_warehouses.sample_events())
    sample["lines_loaded"] = server.load(directory)

    build = build_report(args.server)

    free = warehouse.FilesWarehouse(directory)
    agreed: dict[str, bool] = {}
    columns: dict[str, list[str]] = {}
    for name in sorted(warehouse.MART):
        built_header, built_rows = rows_of(server.read_model(name))
        free_header, free_rows = rows_of(free.run(name))
        agreed[name] = built_header == free_header and built_rows == free_rows
        columns[name] = built_header

    report = {
        "project": str(PROJECT.relative_to(ROOT)),
        "dbt": footprint(),
        "build_mode": {
            "engine": "clickhouse-server",
            "url": args.server,
            "version": server.scalar("SELECT version()"),
        },
        "free_mode": {"engine": "chdb", "version": chdb_version()},
        "events": sample,
        "build": build,
        "columns": columns,
        "agreed": agreed,
        "all_agreed": all(agreed.values()),
        "mart": warehouse.report(
            {name: server.read_model(name) for name in sorted(warehouse.MART)}
        ),
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if temporary is not None:
        temporary.cleanup()

    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("mart", "columns")},
            ensure_ascii=False,
            indent=2,
        )
    )
    if build["returncode"] != 0 or build["counts"]["failed"]:
        print("dbt build красный", file=sys.stderr)
        return 1
    if not report["all_agreed"]:
        differing = [name for name, same in agreed.items() if not same]
        print(
            f"dbt и бесплатный режим разошлись на моделях: {', '.join(differing)}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
