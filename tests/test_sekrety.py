"""Проверка на секреты проверяется делом, а не чтением.

Скрипт ищет в репозитории то, что нельзя публиковать. Пока он не поймал ни
одной подложенной строки, его зеленый вывод ничего не значит: ровно так
выглядел бы скрипт с опечаткой в шаблоне. Поэтому каждая проверка здесь
подкладывает находку в отдельный временный репозиторий и ждет отказа.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

КОРЕНЬ = Path(__file__).resolve().parents[1]
# Метка приватного ключа собирается из частей нарочно. Целиком она в файле
# стоять не может: git-хук проекта такой коммит не пропускает, и это верно,
# строки, похожей на ключ, в репозитории быть не должно даже в тесте.
МЕТКА_КЛЮЧА = "-----BEGIN " + "OPENSSH PRIVATE KEY" + "-----"
СКРИПТ = КОРЕНЬ / "scripts" / "check_secrets.sh"


def репозиторий(путь: Path, файлы: dict[str, str]) -> Path:
    """Временный репозиторий с копией скрипта и подложенными файлами."""
    (путь / "scripts").mkdir(parents=True)
    shutil.copy(СКРИПТ, путь / "scripts" / "check_secrets.sh")
    for имя, текст in файлы.items():
        файл = путь / имя
        файл.parent.mkdir(parents=True, exist_ok=True)
        файл.write_text(текст, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=путь, check=True)
    subprocess.run(["git", "add", "-A"], cwd=путь, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "test"],
        cwd=путь,
        check=True,
    )
    return путь


def прогон(путь: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(путь / "scripts" / "check_secrets.sh")],
        capture_output=True,
        text=True,
    )


def test_chistyi_repozitorii_prohodit(tmp_path: Path) -> None:
    р = репозиторий(tmp_path, {"README.md": "ничего лишнего\n"})
    итог = прогон(р)
    assert итог.returncode == 0, итог.stderr
    assert "секретов не найдено" in итог.stdout


@pytest.mark.parametrize(
    "имя, содержимое, ожидание",
    [
        ("src/upload.py", 'TOKEN = "hf_' + "a" * 34 + '"\n', "похожее на секрет"),
        ("ci.yml", "token: ghp_" + "b" * 36 + "\n", "похожее на секрет"),
        ("key.pem", МЕТКА_КЛЮЧА + "\n", "похожее на секрет"),
        ("docs/people.md", "писать на petrov@example.com\n", "адрес почты"),
        (".env", "VAST_API_KEY=" + "c" * 40 + "\n", "файл с ключами"),
        (".env.example", "HF_TOKEN=hf_realvalue\n", "содержит значения"),
    ],
)
def test_nahodka_ronyaet_proverku(tmp_path: Path, имя: str, содержимое: str, ожидание: str) -> None:
    р = репозиторий(tmp_path, {имя: содержимое})
    итог = прогон(р)
    assert итог.returncode == 1, итог.stdout
    assert ожидание in итог.stderr


def test_sekret_iz_istorii_nahoditsya(tmp_path: Path) -> None:
    """Удаленный ключ остается в истории, и проверка обязана его видеть."""
    р = репозиторий(tmp_path, {"deploy.sh": "echo sk-" + "d" * 24 + "\n"})
    (р / "deploy.sh").write_text("echo ключ убран\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=р, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "clean"],
        cwd=р,
        check=True,
    )
    итог = прогон(р)
    assert итог.returncode == 1, итог.stdout
    assert "история git" in итог.stderr
