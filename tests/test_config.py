import pytest
import yaml

from rufinrag.config import ConfigError, env, load_catalog


def test_project_catalog_loads():
    catalog = load_catalog()
    assert catalog.default == "hybrid"
    assert set(catalog.configs) == {"bm25", "dense", "bm25-rerank", "hybrid", "hybrid-rerank"}


def test_default_config_has_no_online_rerank():
    """Решение G0: в онлайне по умолчанию кросс-энкодера нет, он стоит 12,5 с на CPU."""
    assert load_catalog().get().rerank is None


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
