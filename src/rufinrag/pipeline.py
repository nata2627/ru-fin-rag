"""Шаги регламента: отпечатки входов, датчики свежести, отметки о прогоне.

Граница такая: скрипты делают работу (докачка, сборка индекса, регрессия), а
DAG отвечает за порядок, ретраи и за то, чтобы шаг не запускался, когда его
вход не менялся. Знание о том, какой файл чему вход, лежит здесь, в одном
месте: его читают и DAG, и тесты, и проверка повторного прогона.

Только стандартная библиотека, намеренно. DAG разбирает интерпретатор Airflow,
в котором зависимостей сервиса нет вовсе: они стоят в отдельном окружении, и
шаги запускаются в нем подпроцессом. `import yaml` здесь уронил бы разбор DAG.

Свежесть считается по содержимому, а не по времени файла. Прогон CI клонирует
репозиторий, и у всех файлов клона одно время изменения, то есть сравнение
`mtime` входа с `mtime` выхода там не значит ничего. Поэтому шаг при успехе
пишет отметку с отпечатком sha256 каждого входа, а датчик сравнивает отпечатки.
Отметки лежат рядом с артефактами, которые описывают, и в репозиторий не едут:
индекс на каждой машине собирается свой.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CORPUS_FILE = "data/corpus/chunks.jsonl.gz"
STAMP_DIR = "data/stamps"
UPDATE_REPORT = "results/corpus-update.json"
ALERT_THRESHOLD = "results/alert-threshold.json"
# Отдельный код выхода под отказ источника. 75 это EX_TEMPFAIL из `sysexits.h`:
# «попробуй позже», ровно тот смысл, который здесь и нужен. Контракт один на
# три места: сам скрипт докачки, шаг ночного прогона и шаг DAG.
EXIT_SOURCE_UNAVAILABLE = 75
# Как часто вообще можно спрашивать у cbr.ru перечень «Вестника». Расписание
# ходит дважды в сутки, то есть раз в 12 часов, поэтому шесть часов не мешают
# плановому прогону, но превращают повторный запуск руками в пропуск шага, а не
# в еще шесть запросов к госсайту.
SOURCE_COOLDOWN_HOURS = 6.0
DIGEST_CHARS = 16
CHECKED_AT = "%Y-%m-%dT%H:%M:%S"


@dataclass(frozen=True)
class Step:
    """Шаг регламента: чем он питается и что оставляет после себя."""

    name: str
    sources: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class Freshness:
    """Решение датчика и причина, которую он пишет в лог задачи."""

    stale: bool
    reason: str


INDEX = Step(
    name="index",
    sources=(CORPUS_FILE,),
    outputs=("data/index/bm25.npz", "data/index/texts.blocks"),
)
REGRESSION = Step(
    name="regression",
    # Эталонный набор и файл связок тут не для красоты: регрессию надо
    # пересчитать и когда корпус не менялся, а поменялись вопросы разметки или
    # состав связки.
    sources=(
        "data/index/bm25.npz",
        "data/benchmark/qrels.tsv",
        "data/benchmark/queries.jsonl",
        "configs/retrieval.yaml",
    ),
    outputs=("results/regression.json",),
)
STEPS = {step.name: step for step in (INDEX, REGRESSION)}


def digest(path: Path) -> str:
    """Отпечаток файла. Корпус это 23 МБ, на маке счет стоит около 0,05 с."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()[:DIGEST_CHARS]


def stamp_path(root: Path, step: Step) -> Path:
    return root / STAMP_DIR / f"{step.name}.json"


def read_stamp(root: Path, step: Step) -> dict | None:
    path = stamp_path(root, step)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_stamp(root: Path, step: Step, extra: dict | None = None) -> dict:
    """Отметка об успешном шаге: отпечатки входов на момент, когда шаг прошел."""
    stamp = {
        "step": step.name,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {name: digest(root / name) for name in step.sources},
        **(extra or {}),
    }
    path = stamp_path(root, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stamp


def freshness(root: Path, step: Step) -> Freshness:
    """Нужно ли запускать шаг: вход изменился или вывода нет на диске."""
    missing = [name for name in step.outputs if not (root / name).exists()]
    if missing:
        return Freshness(True, f"нет вывода прошлого прогона: {', '.join(missing)}")
    stamp = read_stamp(root, step)
    if stamp is None:
        return Freshness(True, "нет отметки о прошлом прогоне шага")
    known = stamp.get("sources", {})
    changed = [
        name
        for name in step.sources
        if known.get(name) != (digest(root / name) if (root / name).exists() else None)
    ]
    if changed:
        return Freshness(True, f"вход изменился со прошлого прогона: {', '.join(changed)}")
    return Freshness(False, f"вход не менялся с прогона {stamp['finished_at']}, шаг пропущен")


def source_freshness(
    root: Path,
    hours: float = SOURCE_COOLDOWN_HOURS,
    now: datetime | None = None,
) -> Freshness:
    """Пора ли снова спрашивать у cbr.ru перечень «Вестника».

    У внешнего источника отпечатка входа нет: узнать, появился ли новый выпуск,
    можно только спросив. Поэтому честный датчик здесь один, и он про частоту
    обращений: время прошлого опроса лежит в отчете докачки, и пока не прошел
    срок, шаг пропускается, а не повторяет шесть запросов к госсайту.
    """
    path = root / UPDATE_REPORT
    if not path.exists():
        return Freshness(True, "отчета о прошлой докачке нет, спрашиваем впервые")
    checked = json.loads(path.read_text(encoding="utf-8")).get("checked_at")
    if not checked:
        return Freshness(True, f"в {UPDATE_REPORT} нет времени опроса, спрашиваем")
    stamped = datetime.strptime(checked, CHECKED_AT)
    elapsed = ((now or datetime.now()) - stamped).total_seconds() / 3600
    if elapsed < 0:
        # Отчет пишется тем прогоном, который докачивал, а часы прогона CI и
        # мака расходятся на часовой пояс. Отрицательный срок читается как
        # «только что спрашивали»: это осторожное из двух толкований.
        return Freshness(False, f"опрос помечен будущим временем {checked}, шаг пропущен")
    if elapsed < hours:
        return Freshness(
            False, f"cbr.ru спрашивали {elapsed:.1f} ч назад при сроке {hours:.0f} ч, шаг пропущен"
        )
    return Freshness(True, f"прошлый опрос cbr.ru {elapsed:.1f} ч назад, пора спрашивать")


def live_run() -> bool:
    """Боевой режим: без него шаги наружу не ходят (CLAUDE.md, «Наружу»)."""
    return os.environ.get("RUFINRAG_LIVE", "").strip() == "1"


def alert_threshold(root: Path) -> float:
    """Порог алерта по деградации метрики: число из замера, а не из кода.

    Замер лежит в `results/alert-threshold.json`, его пересчитывает
    `scripts/measure_alert_threshold.py`. Числом в двух местах порог быть не
    может: прогон сравнивал бы одно, README обещал бы другое, и разошлись бы
    они молча. Замер, который не прошел свою же проверку (порог уперся в
    потолок), алертом пользоваться не дает: тогда мониторинг на этом наборе
    не имеет смысла, и это ошибка, а не повод взять число по умолчанию.
    """
    report = json.loads((root / ALERT_THRESHOLD).read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise ValueError(f"{ALERT_THRESHOLD}: замер порога не прошел свою проверку")
    return float(report["threshold"])
