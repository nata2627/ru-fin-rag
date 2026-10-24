import pytest
import yaml

from rufinrag import config
from rufinrag.config import ConfigError, env, load_catalog

ROOT = config.project_root()


def test_project_catalog_loads():
    catalog = load_catalog()
    assert catalog.default == "hybrid"
    assert set(catalog.configs) == {
        "bm25",
        "dense",
        "bm25-rerank",
        "hybrid",
        "hybrid-rerank",
        "hybrid-graph",
    }


def test_default_config_has_no_online_rerank():
    """Решение G0: в онлайне по умолчанию кросс-энкодера нет, он стоит 12,5 с на CPU."""
    assert load_catalog().get().rerank is None


def test_default_config_has_no_graph_expansion():
    """S20: расширение по цитированию замерено и прибавки не дало, связка остается отдельной."""
    assert load_catalog().get().graph is None


def test_version_describes_pipeline():
    catalog = load_catalog()
    assert catalog.get("hybrid").version == "hybrid+hybrid+cand50+rufin"
    assert catalog.get("hybrid-rerank").version == "hybrid-rerank+hybrid+cand50+rufin+rerank50"


def test_unknown_config_name_fails_loudly():
    catalog = load_catalog()
    with pytest.raises(ConfigError, match="нет связки"):
        catalog.get("hybrid-rerank-v2")


def test_missing_field_fails_loudly(tmp_path):
    path = tmp_path / "retrieval.yaml"
    path.write_text(
        yaml.safe_dump({"default": "a", "configs": {"a": {"first_stage": "bm25"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="candidates"):
        load_catalog(path)


def test_default_pointing_nowhere_fails_loudly(tmp_path):
    path = tmp_path / "retrieval.yaml"
    path.write_text(
        yaml.safe_dump(
            {"default": "b", "configs": {"a": {"first_stage": "bm25", "candidates": 10}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="по умолчанию"):
        load_catalog(path)


def test_empty_required_env_fails_loudly(monkeypatch):
    monkeypatch.setenv("RUFINRAG_TEST_KEY", "   ")
    with pytest.raises(ConfigError, match="RUFINRAG_TEST_KEY"):
        env("RUFINRAG_TEST_KEY", required=True)
    assert env("RUFINRAG_TEST_KEY") == ""


def test_default_config_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("RUFINRAG_DEFAULT_CONFIG", "bm25")
    assert load_catalog().default == "bm25"


def test_unknown_override_fails_loudly(monkeypatch):
    monkeypatch.setenv("RUFINRAG_DEFAULT_CONFIG", "bm25-fast")
    with pytest.raises(ConfigError, match="такой связки нет"):
        load_catalog()


def test_root_falls_back_to_working_directory(tmp_path, monkeypatch):
    """Репетиция 22.10.2026: установленный пакет искал configs в site-packages.

    Хостинг клонирует репозиторий и ставит пакет в систему, поэтому папка над
    пакетом маркера не содержит, и корнем обязана становиться рабочая папка.
    """
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/retrieval.yaml").write_text(
        (ROOT / "configs/retrieval.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    # Пакет лежит так, что папка над ним маркера не содержит: иначе первый же
    # кандидат срабатывает, и запасной вариант остается непроверенным.
    installed = tmp_path / "venv/lib/python3.11/site-packages/rufinrag/config.py"
    installed.parent.mkdir(parents=True)
    monkeypatch.setattr(config, "__file__", str(installed))
    assert not (installed.resolve().parents[2] / config.CONFIG_FILE).exists()

    assert config.project_root() == tmp_path
    assert config.data_dir() == tmp_path / "data"
    assert config.load_catalog().configs


def test_missing_root_names_the_variable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    installed = tmp_path / "venv/lib/python3.11/site-packages/rufinrag/config.py"
    installed.parent.mkdir(parents=True)
    monkeypatch.setattr(config, "__file__", str(installed))
    with pytest.raises(config.ConfigError, match="RUFINRAG_CONFIG_PATH"):
        config.project_root()
