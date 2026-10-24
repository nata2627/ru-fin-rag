"""Запись реестра версий: results/registry.json.

Задача 6.3. Файл собирается кодом из репозитория и коммитится: сервис его
только читает, а проверка сравнивает с ним то, что подняла машина
(`src/rufinrag/registry.py`). MLflow здесь не нужен нарочно, иначе реестр
нельзя было бы пересобрать там, где его нет.

Запуск: python scripts/register_versions.py [--out results/registry.json]
Стоит около 6 с на маке: 62 606 фрагментов читаются ради версии корпуса, а
448,8 МБ весов и 45,9 МБ векторов считаются целиком ради отпечатка.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rufinrag.corpus import load_corpus  # noqa: E402
from rufinrag.registry import build, load  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/registry.json")
    args = parser.parse_args()

    started = time.monotonic()
    corpus = load_corpus()
    # Прежняя запись нужна не для сравнения, а чтобы не потерять отпечатки
    # артефактов, которых на этой машине нет (ночной прогон без весов).
    entry = build(ROOT, corpus_version=corpus.version, previous=load(ROOT))
    entry = {
        "built_at": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "seconds": round(time.monotonic() - started, 1),
        **entry,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    missing = [name for name, value in entry["artifacts"].items() if value is None]
    inherited = [
        name for name, value in entry["artifacts"].items()
        if value and value.get("from_previous")
    ]
    print(f"реестр {entry['version']}: корпус {entry['corpus_version']}, "
          f"связка по умолчанию {entry['default_config']}, ответ {entry['answer_version']}")
    if inherited:
        print(f"отпечатки перенесены из прежней записи: {len(inherited)} из "
              f"{len(entry['artifacts'])} ({', '.join(inherited)})")
    if missing:
        print(f"артефактов нет на диске: {len(missing)} из {len(entry['artifacts'])} "
              f"({', '.join(missing)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
