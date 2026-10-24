"""Чтение конфигурации поиска и переменных окружения.

Падаем громко на пустой обязательной переменной и на неизвестном имени связки:
тихая подстановка значения по умолчанию прячет опечатку в имени переменной на
хостинге, и прогон молча отвечает не тем, чем думает автор (SPEC, раздел 3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_FILE = Path("configs") / "retrieval.yaml"
DATA_DIR = Path("data")


class ConfigError(Exception):
    pass


def project_root(marker: Path = CONFIG_FILE) -> Path:
    """Где лежат `configs` и `data` в этом запуске.

    Путь не выводится из места пакета: при `pip install .` пакет уезжает в
    site-packages, и `parents[2]` указывает на `/usr/local/lib/python3.11`.
    Репетиция бесплатного тарифа 22.10.2026 так и упала, причем молча до
    первого запроса. Поэтому корень ищется по маркеру: сначала папка над
    пакетом (рабочее дерево и редактируемая установка), потом рабочая папка
    процесса (так запускается хостинг, который клонирует репозиторий и ставит
    пакет в систему).
    """
    for candidate in (Path(__file__).resolve().parents[2], Path.cwd()):
        if (candidate / marker).exists():
            return candidate
    raise ConfigError(
        f"не найден корень проекта: нет {marker} ни над пакетом, ни в {Path.cwd()}. "
        "Задайте RUFINRAG_CONFIG_PATH или RUFINRAG_DATA_DIR"
    )


# Направления обхода графа ссылок. Список повторяет `graph.DIRECTIONS`, потому
# что `config` не импортирует `graph` (тот читает корпус, а корпус читает
# конфигурацию). Совпадение двух списков держит тест.
GRAPH_DIRECTIONS = ("forward", "backward", "both")
GRAPH_MODES = ("fusion", "reserve")


@dataclass(frozen=True)
class GraphSpec:
    """Расширение выдачи по цитированию: режим, направление, размеры.

    Замер режимов лежит в `results/graph-expansion.json` (EXPERIMENTS, S20):
    `fusion` проиграл базе 0,1301 NDCG@10 на 991 запросе, `reserve` дал 0,0001
    [-0,0014; 0,0018], то есть ничего. Параметры здесь, а не в коде, потому что
    смена связки это правка файла конфигурации (SPEC, раздел 3).
    """

    mode: str
    direction: str
    seeds: int
    depth: int
    weight: float
    reserve: int


@dataclass(frozen=True)
class RerankSpec:
    model: str
    depth: int
    batch_size: int
    max_length: int


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    first_stage: str
    candidates: int
    dense_model: str | None
    rrf_k: int
    rerank: RerankSpec | None
    graph: GraphSpec | None
    note: str

    @property
    def version(self) -> str:
        """Версия связки для ответа API и для строки лога.

        Имя плюс состав: по нему видно, тем же ли путем получены две выдачи.
        """
        parts = [self.name, self.first_stage, f"cand{self.candidates}"]
        if self.dense_model:
            parts.append(self.dense_model)
        if self.rerank:
            parts.append(f"rerank{self.rerank.depth}")
        if self.graph:
            parts.append(f"graph{self.graph.mode}{self.graph.direction}{self.graph.seeds}")
        return "+".join(parts)


@dataclass(frozen=True)
class Catalog:
    version: int
    default: str
    configs: dict[str, RetrievalConfig]

    def get(self, name: str | None = None) -> RetrievalConfig:
        key = name or self.default
        if key not in self.configs:
            known = ", ".join(sorted(self.configs))
            raise ConfigError(f"нет связки {key!r}, известны: {known}")
        return self.configs[key]


def config_path() -> Path:
    override = env("RUFINRAG_CONFIG_PATH")
    return Path(override).expanduser() if override else project_root() / CONFIG_FILE


def load_catalog(path: Path | None = None) -> Catalog:
    path = path or config_path()
    if not path.exists():
        raise ConfigError(f"нет файла конфигурации {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "configs" not in raw:
        raise ConfigError(f"{path}: нет раздела configs")

    configs: dict[str, RetrievalConfig] = {}
    for name, item in raw["configs"].items():
        rerank = item.get("rerank")
        graph = item.get("graph")
        configs[name] = RetrievalConfig(
            name=name,
            first_stage=_required(item, "first_stage", name),
            candidates=int(_required(item, "candidates", name)),
            dense_model=item.get("dense_model"),
            rrf_k=int(item.get("rrf_k", 60)),
            rerank=RerankSpec(
                model=_required(rerank, "model", f"{name}.rerank"),
                depth=int(rerank.get("depth", 50)),
                batch_size=int(rerank.get("batch_size", 8)),
                max_length=int(rerank.get("max_length", 512)),
            )
            if rerank
            else None,
            graph=_graph_spec(graph, name) if graph else None,
            note=str(item.get("note", "")).strip(),
        )

    default = raw.get("default")
    if default not in configs:
        raise ConfigError(f"{path}: связка по умолчанию {default!r} не описана")

    # Связку по умолчанию можно переопределить переменной окружения: нагрузочный
    # замер и сравнение конфигураций гоняют один и тот же образ с разными
    # связками, и правка файла на время прогона потерялась бы в истории.
    override = env("RUFINRAG_DEFAULT_CONFIG")
    if override:
        if override not in configs:
            known = ", ".join(sorted(configs))
            raise ConfigError(
                f"RUFINRAG_DEFAULT_CONFIG={override!r}: такой связки нет, известны: {known}"
            )
        default = override

    return Catalog(version=int(raw.get("version", 1)), default=default, configs=configs)


def _graph_spec(raw: dict, name: str) -> GraphSpec:
    where = f"{name}.graph"
    mode = str(_required(raw, "mode", where))
    direction = str(_required(raw, "direction", where))
    if mode not in GRAPH_MODES:
        raise ConfigError(f"{where}: режим {mode!r}, известны: {', '.join(GRAPH_MODES)}")
    if direction not in GRAPH_DIRECTIONS:
        raise ConfigError(
            f"{where}: направление {direction!r}, известны: {', '.join(GRAPH_DIRECTIONS)}"
        )
    spec = GraphSpec(
        mode=mode,
        direction=direction,
        seeds=int(_required(raw, "seeds", where)),
        depth=int(raw.get("depth", 0)),
        weight=float(raw.get("weight", 1.0)),
        reserve=int(raw.get("reserve", 0)),
    )
    if spec.seeds < 1:
        raise ConfigError(f"{where}: seeds {spec.seeds}, нужно не меньше одного")
    if mode == "fusion" and spec.depth < 1:
        raise ConfigError(f"{where}: режиму fusion нужна глубина depth больше нуля")
    if mode == "reserve" and spec.reserve < 1:
        raise ConfigError(f"{where}: режиму reserve нужно reserve больше нуля")
    return spec


def _required(item: dict, key: str, where: str):
    if not isinstance(item, dict) or item.get(key) in (None, ""):
        raise ConfigError(f"{where}: нет обязательного поля {key}")
    return item[key]


def env(name: str, required: bool = False) -> str:
    """Переменная окружения. Пустая обязательная переменная это ошибка, а не пустая строка."""
    value = os.environ.get(name, "").strip()
    if required and not value:
        raise ConfigError(f"не задана переменная окружения {name}")
    return value


def live_mode() -> bool:
    return env("RUFINRAG_LIVE") == "1"


def data_dir() -> Path:
    override = env("RUFINRAG_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return project_root(DATA_DIR) / DATA_DIR
