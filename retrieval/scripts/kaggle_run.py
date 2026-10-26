#!/usr/bin/env python3
"""Управление вычислениями на Kaggle из командной строки.

Тяжёлые шаги считаются на чужой видеокарте, и до сих пор это означало
ручные действия в браузере: загрузить датасет, создать ноутбук, подключить
входы, включить ускоритель, дождаться, скачать. Каждый такой шаг — место,
где легко ошибиться и трудно повторить.

Официальный Kaggle CLI умеет всё то же самое, поэтому прогон описывается
файлом и запускается одной командой. Ноутбук собирается из скрипта этапа:
на Kaggle уезжает не рукописная тетрадь, а тот же код, что лежит в проекте.

Команды:
  dataset   обновить датасет с корпусом и кодом
  queries   загрузить набор запросов отдельным датасетом
  weights   загрузить обученные веса отдельным датасетом
  run       собрать ноутбук, запустить и дождаться
  fetch     забрать результат последнего прогона
  time      остаток недельной квоты, сколько ядро считает и сколько осталось
            до предела сессии
  watch     следить за идущим ядром, печатать его живые числа и погасить его,
            когда вышел объявленный срок
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_kaggle_package import QUERIES  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIST = os.path.join(ROOT, "dist")
PKG = os.path.join(DIST, "kaggle")

DATASET = "ru-fin-retrieval"
QUERIES_DATASET = "ru-fin-queries"
WEIGHTS_DATASET = "ru-fin-weights"

# Модели, подключаемые входом вместо скачивания с HuggingFace. Секреты Kaggle
# через API не прицепить вовсе — ядро, созданное командой, токена не увидит, —
# а без токена скачивание режется по скорости. Входом модель приезжает
# мгновенно и без сети.
#
# Это чужие зеркала, поэтому `gpu_prepare.py` их сверяет: устройство модели
# против ожидаемого и, главное, воспроизведение уже посчитанной выдачи
# `base__hybrid-rerank.jsonl`. Совпали первые места — тот самый учитель,
# которым измерено 0,690. Не совпали — прогон отказывается считать.
MODELS = {
    # зеркало intfloat/multilingual-e5-small, закреплено за коммитом
    # 614241f622f53c4eeff9890bdc4f31cfecc418b3
    "e5-small": "dangkhoa2016/intfloat-multilingual-e5-small/transformers/default/1",
    # зеркало BAAI/bge-reranker-v2-m3, 2,29 ГБ — размер весов в fp32
    "reranker": "andreasbis/baai-bge-reranker-v2-m3/transformers/default/1",
    # судья пула, Qwen2.5-14B-Instruct-AWQ, 9,98 ГБ. Выложен самим Qwen,
    # а не чужим зеркалом, и это та же модель, которой размечен прежний пул:
    # каппа с человеком 0,408. Проверяется она в том же прогоне, потому что
    # выборка человека идёт в задании первой (prioritet 0).
    #
    # Квантование важно: четырёхбитные веса это 10 ГБ и одна карта T4,
    # полные были бы 29 ГБ и две. На P100 ядра AWQ не работают, нужна
    # Turing или новее, то есть `--machine ""` (T4 x2), а не `p100`.
    "judge": "qwen-lm/qwen2.5/transformers/14b-instruct-awq/1",
    # Второй судья, Qwen2.5-32B-Instruct-AWQ, 18,00 ГиБ по пяти файлам весов.
    # Берётся затем, что один судья неотличим от собственной погрешности:
    # голосуют двое, и пара, где они разошлись, в метрику не идёт.
    #
    # Почему именно 32B и именно в четырёх битах. В одну карту T4 (15,84 ГиБ)
    # эти 18 ГиБ не входят, а на двух с тензорным параллелизмом выходит
    # по 9,0 ГиБ весов на карту и при `--gpu-util 0.90` остаётся по 5,2 ГиБ
    # под кэш ключей. Следующая выкладка того же Qwen это 72B-instruct-awq,
    # у неё веса около 41 ГиБ, то есть в 31,7 ГиБ узла она не входит вовсе.
    # Полные веса 32B были бы 65 ГиБ. Турингу нужны ядра AWQ, а не Marlin,
    # то есть `--machine ""` (T4 x2), не `p100`.
    "judge2": "qwen-lm/qwen2.5/transformers/32b-instruct-awq/1",
    # Генератор вопросов, Qwen2.5-7B-Instruct-AWQ. Тот же путь, что у судьи,
    # и по той же причине: ядру, созданному командой, токен HuggingFace
    # прицепить нельзя, а без токена скачивание весов режется по скорости.
    # Прежде генерация шла из браузера, где токен есть.
    "gen": "qwen-lm/qwen2.5/transformers/7b-instruct-awq/1",
}


# Предел одной сессии Kaggle. По нему планируются шаги: прогон, которому
# осталось меньше, чем он просит, лучше резать на два заранее, а не узнавать
# об обрыве на одиннадцатом часу.
SESSION_HOURS = 12.0


def quota() -> list[dict]:
    """Недельная квота ускорителей: сколько потрачено, сколько осталось, когда
    обновится. Отдает ее сам клиент (`kaggle quota`), и это надежнее любой
    своей бухгалтерии по журналам прогонов: в квоту идет время ядра целиком,
    включая установку пакетов и монтирование весов, а не время счета.
    """
    import csv

    r = run(["kaggle", "quota", "--csv"])
    все = (r.stdout or "").splitlines()
    начало = next((i for i, l in enumerate(все) if l.startswith("resource,")), None)
    if начало is None:
        return []
    return list(csv.DictReader(все[начало:]))


def kernel_times(user: str, фильтр: str = "ru-fin") -> list[dict]:
    """Когда ядра запускались и сколько им осталось до предела сессии.

    Время старта берется из `lastRunTime` списка ядер (он в UTC), состояние
    спрашивается отдельно: в списке его нет вовсе.
    """
    import csv
    import datetime as dt

    # Поиск `-s` ищет по заголовку, а заголовки у Kaggle с пробелами («ru fin
    # e5 train»), поэтому подстрока вроде `ru-fin` не находит ничего. Отбор
    # идет по ссылке на ядро, здесь.
    r = run(["kaggle", "kernels", "list", "--mine", "-v", "--page-size", "50"])
    # Клиент печатает предупреждение о своей версии перед таблицей, и в нем
    # тоже есть запятые: разбор начинается со строки заголовка, иначе
    # предупреждение становится заголовком таблицы.
    все = (r.stdout or "").splitlines()
    начало = next((i for i, l in enumerate(все) if l.startswith("ref,")), None)
    if начало is None:
        return []
    строки = все[начало:]
    out = []
    сейчас = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    for row in csv.DictReader(строки):
        ref = (row.get("ref") or "").strip()
        сырое = (row.get("lastRunTime") or "").strip()
        if not ref or not сырое or фильтр not in ref:
            continue
        try:
            старт = dt.datetime.fromisoformat(сырое)
        except ValueError:
            continue
        прошло = (сейчас - старт).total_seconds() / 3600
        состояние = run(["kaggle", "kernels", "status", ref]).stdout or ""
        out.append({"ref": ref, "старт": старт, "часов прошло": прошло,
                    "часов осталось": SESSION_HOURS - прошло,
                    "состояние": состояние.strip().split('"')[-2]
                                 if '"' in состояние else состояние.strip()})
    return out


def _разобрать(сырое) -> str:
    """Текст лога из того, что отдал клиент.

    Русские строки приезжают дважды закодированными: клиент отдает UTF-8
    байты от строки, которая уже была испорчена однобайтовой кодировкой,
    поэтому прямое чтение дает «Ð¿ÑÐ¸Ð½ÑÑÐ¾» вместо «принято». Обратная
    перекодировка возвращает текст; если она не удалась, берется прямое
    чтение, чтобы латинские строки лога не потерять.
    """
    текст = сырое.decode("utf-8", "replace") if isinstance(сырое, bytes) else (сырое or "")
    # Символы вне однобайтовой кодировки (рамки, проценты загрузки) при
    # обратной перекодировке теряются, поэтому и encode, и decode идут
    # терпимыми, а результат берется только если русский в нем нашелся.
    исправленное = текст.encode("latin-1", "ignore").decode("utf-8", "replace")
    кириллица = re.compile(r"[а-яА-Я]")
    return исправленное if кириллица.search(исправленное) else текст


def live_tail(ref: str, seconds: int = 20, игла: str = "") -> list[str]:
    """Последние строки лога идущей сессии.

    Клиент умеет отдавать лог вживую (`kernels logs -f`), и утверждать
    обратное нельзя: на этом проекте молчащий прогон дважды оказывался
    вставшим и стоил по двенадцать часов квоты. Поток обрывается по
    таймауту, потому что сам он не кончается до конца сессии.
    """
    try:
        п = subprocess.run(["kaggle", "kernels", "logs", "-f", ref],
                           text=True, capture_output=True, timeout=seconds)
        вывод = _разобрать(п.stdout)
    except subprocess.TimeoutExpired as истек:
        вывод = _разобрать(истек.stdout)
    строки = [l.strip() for l in вывод.splitlines() if l.strip()]
    if игла:
        строки = [l for l in строки if игла in l]
    # Клиент повторяет каждую строку дважды (поток и его копия в выводе
    # ячейки), поэтому подряд идущие повторы убираются.
    без_повторов: list[str] = []
    for l in строки:
        if not без_повторов or без_повторов[-1] != l:
            без_повторов.append(l)
    return без_повторов


def watch(ref: str, deadline_hours: float, every: int = 300,
          игла: str = "принято") -> str:
    """Следить за ядром, печатать его числа и погасить по сроку.

    Срок объявляется до запуска и берется из замера. Гасит тот, кто
    запускал: зависшее ядро стоит двенадцати часов квоты, а потерянный
    прогон стоит минут. Отмены у клиента нет, ядро останавливает удаление.
    """
    начало = time.monotonic()
    while True:
        состояние = (run(["kaggle", "kernels", "status", ref]).stdout or "").lower()
        прошло = (time.monotonic() - начало) / 3600
        if "complete" in состояние:
            print(f"{ref}: завершилось, слежение {прошло:.1f} ч", flush=True)
            return "complete"
        if "error" in состояние or "cancel" in состояние:
            print(f"{ref}: упало или отменено, слежение {прошло:.1f} ч", flush=True)
            return "error"
        for строка in live_tail(ref, игла=игла)[-1:]:
            print(f"{ref} [{прошло:.1f} ч]: {строка}", flush=True)
        if прошло >= deadline_hours:
            print(f"{ref}: вышел объявленный срок {deadline_hours:.1f} ч, гашу "
                  f"ядро удалением (команды отмены у клиента нет)", flush=True)
            р = run(["kaggle", "kernels", "delete", ref, "-y"])
            print((р.stdout or р.stderr).strip(), flush=True)
            return "deadline"
        time.sleep(every)


def flags_of(args_line: str) -> set[str]:
    """Ключи прогона словами, а не подстрокой.

    `--train-file` содержит `--train`, и проверка подстрокой считала фильтр
    учителя прогоном генерации: ядро отказывалось запускаться с правилом
    «vLLM и sentence-transformers вместе не ставятся», хотя языковая модель
    в нем не поднимается вовсе.
    """
    return {слово for слово in args_line.split() if слово.startswith("--")}


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def папка_ключа() -> str:
    """Где лежит ключ Kaggle.

    **Аккаунт переключается подменой `HOME`, а не `KAGGLE_CONFIG_DIR`,
    и это проверено.** Клиент читает каталог, указанный в
    `KAGGLE_CONFIG_DIR`, но авторизуется при этом прежним `access_token`
    из `~/.kaggle`, и тот перебивает и файл ключа, и переменные
    `KAGGLE_USERNAME` с `KAGGLE_KEY`. Со стороны это выглядит как удачное
    переключение: каталог в выводе новый, а работа идёт под старым
    аккаунтом и тратит его квоту.

        HOME=~/.kaggle-home-2 python3 scripts/kaggle_run.py ...

    Внутри такого каталога лежит `.kaggle/kaggle.json` нужного аккаунта.
    Имя пользователя печатается первой строкой: читать его перед каждым
    запуском, иначе легко потратить не ту квоту.
    """
    return os.path.expanduser(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle"))


def require_credentials() -> str:
    """Имя пользователя Kaggle; заодно проверка, что ключ на месте.

    Ключ бывает в двух видах: старый kaggle.json с парой «имя и ключ»
    и новый access_token, где имени нет вовсе. Поэтому имя спрашивается
    у самого клиента, а не вычитывается из файла.
    """
    home = папка_ключа()
    if not any(os.path.exists(os.path.join(home, n))
               for n in ("kaggle.json", "access_token")):
        raise SystemExit(
            f"ключ Kaggle не найден в {home}. Взять: kaggle.com -> Settings -> "
            f"API -> Create New Token. Положить туда и выставить права 600. "
            f"Для второго аккаунта: KAGGLE_CONFIG_DIR=~/.kaggle-2")
    probe = subprocess.run(["kaggle", "config", "view"], text=True, capture_output=True)
    for line in (probe.stdout or "").splitlines():
        if line.strip().startswith("- username:"):
            name = line.split(":", 1)[1].strip()
            if name and name != "None":
                return name
    raise SystemExit("клиент Kaggle не сообщает имя пользователя: проверьте ключ")


def push_dataset(user: str, folder: str, slug: str, title: str, message: str) -> None:
    meta = {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "unknown"}]}
    with open(os.path.join(folder, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    probe = run(["kaggle", "datasets", "status", f"{user}/{slug}"])
    exists = probe.returncode == 0 and "error" not in probe.stdout.lower()
    if exists:
        r = run(["kaggle", "datasets", "version", "-p", folder, "-m", message,
                 "-r", "zip", "--dir-mode", "zip"])
    else:
        r = run(["kaggle", "datasets", "create", "-p", folder, "-r", "zip",
                 "--dir-mode", "zip"])
    print((r.stdout or r.stderr).strip())
    if r.returncode != 0:
        raise SystemExit("не удалось загрузить датасет")


def notebook_from_script(script: str, args_line: str,
                         packages: tuple[str, ...] = ("sentence-transformers",)) -> dict:
    """Ноутбук из двух ячеек: подготовка окружения и запуск скрипта этапа."""
    setup = [
        "import glob, os, shutil, subprocess, sys, zipfile\n",
        "\n",
        "# Без токена HuggingFace режет скорость скачивания, и на больших моделях\n",
        "# это превращается в многочасовое ожидание. Токен кладётся в секреты\n",
        "# ноутбука под именем HF_TOKEN; переменная наследуется дочерним процессом.\n",
        "# Имён два: у библиотеки старое HUGGING_FACE_HUB_TOKEN и новое HF_TOKEN.\n",
        "try:\n",
        "    from kaggle_secrets import UserSecretsClient\n",
        "    _tok = UserSecretsClient().get_secret('HF_TOKEN')\n",
        "    os.environ['HF_TOKEN'] = _tok\n",
        "    os.environ['HUGGING_FACE_HUB_TOKEN'] = _tok\n",
        "    print('токен HuggingFace подключён')\n",
        "except Exception as e:\n",
        "    print('токен HuggingFace недоступен, скачивание будет медленным:', e)\n",
        "\n",
        f"ПАКЕТЫ = {list(packages)!r}\n",
        "# Одной командой на все пакеты, и это не сокращение записи. По одному\n",
        "# вторая установка сдвигает то, что поставила первая: vLLM тянет свой\n",
        "# torch, а следующая команда подбирает к нему версию по отдельности.\n",
        "print('ставлю:', ', '.join(ПАКЕТЫ), flush=True)\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U',\n",
        "                *ПАКЕТЫ], check=False)\n",
        "\n",
        "\n",
        "# Импорт проверяется свежим процессом, а не в этом: установка vLLM\n",
        "# обновляет torch вместе с библиотеками CUDA, а уже загруженный сюда\n",
        "# старый torch держит прежнюю версию.\n",
        "#\n",
        "# `AutoProcessor` проверяется нарочно: через него transformers тянет\n",
        "# torchaudio, и расхождение версий CUDA вылезает именно там. Сам\n",
        "# по себе `import vllm` при этом проходит, то есть проверка одного\n",
        "# импорта пропустила бы поломку до загрузки модели, где она стоит\n",
        "# уже пяти минут монтирования весов.\n",
        "def импортируется(код):\n",
        "    п = subprocess.run([sys.executable, '-c', код], capture_output=True,\n",
        "                       text=True)\n",
        "    return п.returncode == 0, (п.stderr or '')[-1600:]\n",
        "\n",
        "\n",
        "ЦЕЛЬ = ('import ' + ', '.join(п.replace('-', '_') for п in ПАКЕТЫ)\n",
        "        + \"; from transformers import AutoProcessor; print('ок')\")\n",
        "ок, беда = импортируется(ЦЕЛЬ)\n",
        "\n",
        "# Лечение одного известного конфликта, и только его. vLLM приносит свой\n",
        "# torch, собранный под CUDA 13, а torchaudio в образе Kaggle собран под\n",
        "# 12.8 и сверяет это при импорте. На PyPI сборки под 13 нет вовсе,\n",
        "# поэтому сначала она берётся из индекса PyTorch, а если и там её нет,\n",
        "# torchaudio снимается: звук в этом проекте не нужен никому, его тянет\n",
        "# один transformers и только когда тот установлен.\n",
        "if not ок and 'TorchAudio' in беда:\n",
        "    куда = subprocess.run(\n",
        "        [sys.executable, '-c', 'import torch; print(torch.version.cuda)'],\n",
        "        capture_output=True, text=True).stdout.strip()\n",
        "    индекс = 'https://download.pytorch.org/whl/cu' + куда.replace('.', '')\n",
        "    print('torchaudio разошёлся с torch по CUDA; беру сборку из', индекс,\n",
        "          flush=True)\n",
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U',\n",
        "                    '--no-deps', '--index-url', индекс, 'torchaudio'],\n",
        "                   check=False)\n",
        "    ок, беда = импортируется(ЦЕЛЬ)\n",
        "    if not ок:\n",
        "        print('сборки под эту CUDA нет, снимаю torchaudio', flush=True)\n",
        "        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q',\n",
        "                        'torchaudio'], check=False)\n",
        "        ок, беда = импортируется(ЦЕЛЬ)\n",
        "\n",
        "assert ок, 'не импортируется то, что поставлено: ' + беда\n",
        "print('библиотеки на месте:', ', '.join(ПАКЕТЫ))\n",
        "subprocess.run([sys.executable, '-c', 'import torch, transformers; '\n",
        "                'print(\"torch\", torch.__version__, \"CUDA\", '\n",
        "                'torch.version.cuda, \"| transformers\", '\n",
        "                'transformers.__version__)'], check=False)\n",
        "\n",
        "# Вывод предыдущего этапа тоже лежит среди входов и содержит копию кода:\n",
        "# ноутбук этапа A копировал датасет в рабочую папку, и всё это стало его\n",
        "# выводом. Если взять код оттуда, приедет старая версия. Поэтому источники\n",
        "# ранжируются: каталог со следами прогона (__notebook__.ipynb, embeddings)\n",
        "# идёт последним.\n",
        "def find_source(root='/kaggle/input', max_depth=4):\n",
        "    found, archives = [], []\n",
        "    for cur, dirs, files in os.walk(root):\n",
        "        if cur.count('/') - root.count('/') >= max_depth:\n",
        "            dirs.clear()\n",
        "        if 'rufin' in dirs:\n",
        "            stale = ('__notebook__.ipynb' in files or 'embeddings' in dirs\n",
        "                     or '__output__.json' in files)\n",
        "            found.append((1 if stale else 0, cur))\n",
        "        archives += [os.path.join(cur, f) for f in files if f.endswith('.zip')]\n",
        "    if found:\n",
        "        found.sort()\n",
        "        return found[0][1], None\n",
        "    return (os.path.dirname(archives[0]), archives[0]) if archives else (None, None)\n",
        "\n",
        "src, archive = find_source()\n",
        "assert src, 'датасет с кодом не подключён'\n",
        "print('источник:', src)\n",
        "if archive:\n",
        "    with zipfile.ZipFile(archive) as z:\n",
        "        z.extractall('/kaggle/working')\n",
        "else:\n",
        "    for item in glob.glob(os.path.join(src, '*')):\n",
        "        dst = os.path.join('/kaggle/working', os.path.basename(item))\n",
        "        if os.path.isdir(item):\n",
        "            shutil.copytree(item, dst, dirs_exist_ok=True)\n",
        "        else:\n",
        "            shutil.copy2(item, dst)\n",
        "os.chdir('/kaggle/working')\n",
        "print('в рабочей папке:', sorted(os.listdir('.'))[:12])\n",
        "\n",
        "# Каноническая нарезка кладётся туда, откуда скрипты берут готовую.\n",
        "# Пересчитывать её здесь нельзя: нарезка зависит от версии токенизатора,\n",
        "# а установка vLLM тянет свой transformers и сдвигает границы — 61 922\n",
        "# фрагмента вместо 62 594 на том же корпусе. Идентификатор позиционный\n",
        "# («акт#номер»), поэтому при сдвиге он не исчезает, а начинает указывать\n",
        "# на другой текст, и эталон съезжает молча. Кэш сносится: упавший прогон\n",
        "# успевает записать туда свою нарезку, а готовая берётся раньше новой.\n",
        "готовая = (glob.glob('/kaggle/input/**/chunks_base.jsonl', recursive=True)\n",
        "           + glob.glob('/kaggle/input/**/chunks_base.jsonl.gz', recursive=True))\n",
        "if готовая:\n",
        "    import gzip\n",
        "    os.makedirs('/kaggle/working/chunks', exist_ok=True)\n",
        "    цель = '/kaggle/working/chunks/base.jsonl'\n",
        "    if os.path.exists(цель):\n",
        "        os.remove(цель)\n",
        "    opener = gzip.open if готовая[0].endswith('.gz') else open\n",
        "    with opener(готовая[0], 'rb') as fi, open(цель, 'wb') as fo:\n",
        "        shutil.copyfileobj(fi, fo, length=1 << 20)\n",
        "    print('нарезка из входов:', sum(1 for _ in open(цель, encoding='utf-8')),\n",
        "          'фрагментов')\n",
        "else:\n",
        "    print('ВНИМАНИЕ: канонической нарезки среди входов нет, будет посчитана '\n",
        "          'заново — для этапа C это недопустимо')\n",
        "\n",
        "# vLLM поднимает движок отдельным процессом и по умолчанию форкается,\n",
        "# а форк не годится, когда CUDA уже тронута в родителе.\n",
        "os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')\n",
    ]
    launch = [
        "import os, subprocess, sys\n",
        f"cmd = [sys.executable, {script!r}] + {args_line!r}.split()\n",
        "print('запуск:', ' '.join(cmd), flush=True)\n",
        "# RUFIN_CARDS снимает сокрытие второй карты (gpu_common). Нужно только\n",
        "# судье в тридцать два миллиарда параметров: 18 ГиБ весов в одну T4\n",
        "# не входят, и тензорный параллелизм тут не ускорение, а условие.\n",
        "env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn')\n",
        "if '--tensor-parallel 2' in ' '.join(cmd):\n",
        "    env['RUFIN_CARDS'] = '0,1'\n",
        "p = subprocess.run(cmd, cwd='/kaggle/working', env=env)\n",
        "\n",
        "# Корпус и нарезка приехали из входов, а рабочая папка целиком\n",
        "# становится выводом версии: `fetch` тащил бы четыреста мегабайт\n",
        "# того же самого обратно. Убирается после счёта и независимо от\n",
        "# кода возврата: упавшему прогону результат нужен тем более.\n",
        "import shutil\n",
        "shutil.rmtree('/kaggle/working/chunks', ignore_errors=True)\n",
        "for имя in ('acts.jsonl', 'acts.jsonl.gz'):\n",
        "    путь = os.path.join('/kaggle/working', имя)\n",
        "    if os.path.exists(путь):\n",
        "        os.remove(путь)\n",
        "осталось = sum(os.path.getsize(os.path.join(б, и))\n",
        "               for б, _, ф in os.walk('/kaggle/working') for и in ф)\n",
        "print(f'в выводе версии {осталось / 1048576:.1f} МБ')\n",
        "raise SystemExit(p.returncode)\n",
    ]
    cells = [{"cell_type": "code", "metadata": {}, "source": s,
              "outputs": [], "execution_count": None} for s in (setup, launch)]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


def push_kernel(user: str, slug: str, title: str, script: str, script_args: str,
                dataset_sources: list[str], kernel_sources: list[str],
                gpu: bool = True, machine: str = "", notebook: str = "",
                model_sources: list[str] | None = None,
                packages: tuple[str, ...] = ("sentence-transformers",)) -> str:
    folder = os.path.join(DIST, "kernel_" + slug)
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    if notebook:
        # Готовая тетрадь из проекта вместо собранной на лету. Нужна там, где
        # прогон — не одна команда, а порядок шагов с проверками между ними:
        # этап C ставит библиотеки, тянет веса под предохранителем, гоняет
        # пробу и только потом считает. Повторять это генератором из двух
        # ячеек значило бы держать одну логику в двух местах, и они разойдутся.
        shutil.copyfile(notebook, os.path.join(folder, "run.ipynb"))
        print(f"  тетрадь: {os.path.relpath(notebook, ROOT)}")
    else:
        with open(os.path.join(folder, "run.ipynb"), "w", encoding="utf-8") as f:
            json.dump(notebook_from_script(script, script_args, packages), f,
                      ensure_ascii=False, indent=1)
    # machine_shape задаёт конфигурацию узла. Пустое значение означает выбор
    # по умолчанию — узел с двумя T4, который дефицитнее и потому дольше ждёт
    # очереди. Наш код работает с одной картой, поэтому просим одиночную.
    meta = {
        "id": f"{user}/{slug}", "title": title, "code_file": "run.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
        "machine_shape": machine,
        "dataset_sources": dataset_sources, "kernel_sources": kernel_sources,
        "competition_sources": [], "model_sources": model_sources or [],
    }
    if model_sources:
        for m in model_sources:
            print(f"  модель входом: {m}")
    with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    # Новая версия не отменяет запуск предыдущей: Kaggle оставляет обе считаться,
    # и обе расходуют квоту. Отмены в клиенте нет, остановить можно только
    # в браузере, поэтому хотя бы предупреждаем.
    probe = run(["kaggle", "kernels", "status", f"{user}/{slug}"])
    state = (probe.stdout or "").lower()
    # очередь опаснее счёта: версия, стоящая в очереди, всё равно запустится
    if "running" in state or "queued" in state:
        print(f"  ВНИМАНИЕ: предыдущий запуск {user}/{slug} ещё не завершён "
              f"(в очереди или считается) и после нового пуша не отменится. "
              f"Остановить можно только в браузере: меню «...» у БОЛЕЕ РАННЕЙ "
              f"версии -> Stop.")

    r = run(["kaggle", "kernels", "push", "-p", folder])
    print((r.stdout or r.stderr).strip())
    if r.returncode != 0:
        raise SystemExit("не удалось запустить ноутбук")
    return f"{user}/{slug}"


def wait(ref: str, poll: int = 60, limit_hours: float = 12.0) -> str:
    print(f"\nжду завершения {ref}; опрос раз в {poll} с", flush=True)
    started = time.monotonic()
    last = ""
    while time.monotonic() - started < limit_hours * 3600:
        r = run(["kaggle", "kernels", "status", ref])
        out = (r.stdout or r.stderr).strip()
        if out != last:
            print("  " + out.replace("\n", " "), flush=True)
            last = out
        low = out.lower()
        if "complete" in low:
            return "complete"
        if "error" in low or "cancel" in low:
            return "error"
        time.sleep(poll)
    return "timeout"


def fetch(ref: str, dest: str, только_лог: bool = False) -> None:
    """Забрать вывод ядра. `только_лог` тащит один журнал прогона.

    Журнал нужен чаще всего остального: по нему разбирают, почему ядро
    упало. А вывод версии весит сотни мегабайт, и клиент тащит его целиком,
    по одному файлу, без возможности остановиться. Разбор падения стоил
    так восьми минут скачивания корпуса, который и уехал-то отсюда.
    """
    os.makedirs(dest, exist_ok=True)
    if только_лог:
        # Отбор по имени есть только в питонном клиенте, у командного его нет.
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        файлы, _ = api.kernels_output(ref, path=dest, file_pattern=r"\.log$",
                                      quiet=False, force=True)
        print("\n".join(файлы) or "журнала нет")
        return
    r = run(["kaggle", "kernels", "output", ref, "-p", dest])
    print((r.stdout or r.stderr).strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dataset", help="обновить датасет с корпусом и кодом")
    p.add_argument("-m", "--message", default="обновление кода")

    p = sub.add_parser("queries", help="загрузить набор запросов")

    p = sub.add_parser("weights", help="загрузить обученные веса")
    p.add_argument("метки", nargs="*", default=None,
                   help="какие этапы заливать; по умолчанию все из data/weights")

    p = sub.add_parser("run", help="собрать ноутбук, запустить и дождаться")
    p.add_argument("stage", choices=["a", "b", "c", "d0", "d", "export", "rerank",
                                     "variants"])
    p.add_argument("--slug", default=None)
    p.add_argument("--source", default="ru-fin",
                   help="ядро, чей вывод подключается: там лежат матрицы этапа A")
    p.add_argument("--args", default="", help="аргументы скрипта этапа, строкой")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--machine", default="p100",
                   help="конфигурация узла: p100 — одна карта, быстрее получить; "
                        "пусто — по умолчанию T4 x2, дефицитнее")
    p.add_argument("--notebook", default=None,
                   help="запустить готовую тетрадь из notebooks/ вместо собранной "
                        "из скрипта этапа")
    p.add_argument("--weights-input", action="store_true",
                   help="подключить датасет с обученными весами. Нужно одному "
                        "финалу: весят они под гигабайт, и Kaggle монтирует "
                        "их перед стартом каждого прогона")
    p.add_argument("--kernels", nargs="*", default=None,
                   help="вывод каких ядер подключить входом, помимо обычного. "
                        "Нужно продолжению рецепта: журнал прежнего прогона "
                        "лежит в выводе предыдущего ядра")
    p.add_argument("--no-gpu", action="store_true",
                   help="считать без видеокарты: квота GPU не тратится, а ядер "
                        "процессора сессии достаётся больше")

    p = sub.add_parser("time", help="сколько ядро считает и сколько осталось")
    p.add_argument("--filter", default="ru-fin",
                   help="по какой подстроке искать ядра в своем списке")
    p.add_argument("--running", action="store_true",
                   help="только те, что сейчас считаются или стоят в очереди")

    p = sub.add_parser("watch", help="следить за ядром и погасить его по сроку")
    p.add_argument("slug")
    p.add_argument("--deadline-hours", type=float, required=True,
                   help="объявленный срок. Берется из замера, а не из надежды: "
                        "зависшее ядро стоит двенадцати часов квоты")
    p.add_argument("--every", type=int, default=300, help="как часто смотреть")
    p.add_argument("--match", default="принято",
                   help="какие строки лога печатать")

    p = sub.add_parser("fetch", help="забрать результат")
    p.add_argument("slug")
    p.add_argument("--dest", default=os.path.join(ROOT, "data", "kaggle_out"))
    p.add_argument("--log", action="store_true",
                   help="только журнал прогона: по нему разбирают падение, "
                        "а вывод версии весит сотни мегабайт")

    args = ap.parse_args()
    user = require_credentials()
    print(f"пользователь Kaggle: {user}  (ключ из {папка_ключа()})\n")

    if args.cmd == "dataset":
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts",
                                                     "make_kaggle_package.py")], check=True)
        push_dataset(user, PKG, DATASET, "ru-fin-retrieval", args.message)

    elif args.cmd == "queries":
        # Вместе с запросами уезжает то, что нужно маленьким прогонам:
        # готовые выдачи и фрагменты базовой нарезки. Так переранжирование
        # обходится без повторной нарезки корпуса.
        folder = os.path.join(DIST, "queries")
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        # Сплит и лист пула — не «на всякий случай», а условие запуска: без
        # сплита этап C откажется генерировать (вопросы ушли бы на акты теста,
        # и утечку потом не отследить), без пула судье нечего размечать.
        # Отсутствие любого из них — не молчаливый пропуск, а предупреждение:
        # иначе оно всплывёт на видеокарте, когда квота уже пошла.
        # Этапу D нужны ещё обучающая выборка, dev, разметка и состав
        # подвыборок: dev на видеокарте обязан считаться тем же составом
        # и тем же эталоном, что в локальном отчёте.
        #
        # Список один и лежит в `make_kaggle_package`. Второй копии здесь
        # быть не должно: она была, и они разошлись молча — лист пула живых
        # вопросов остался только в той, которой собирается архив для
        # браузера, а команда `queries` увозила датасет без него. Судье
        # было бы нечего размечать, и выяснилось бы это на карте.
        for rel in QUERIES:
            name = os.path.basename(rel)
            src = os.path.join(ROOT, rel)
            if os.path.exists(src):
                shutil.copy2(src, folder)
            elif name == "queries.jsonl":
                raise SystemExit("нет data/queries/queries.jsonl: `make queries`")
            else:
                print(f"  ВНИМАНИЕ: {name} нет, уезжает датасет без него")
        # Переранжированная выдача едет не для счёта, а для сверки: ею
        # проверяется, что подключённое зеркало учителя — та самая модель,
        # которой измерено 0,690.
        for name in ("base__bm25.jsonl", "base__dense-bge-m3.jsonl",
                     "base__hybrid.jsonl", "base__hybrid-rerank.jsonl"):
            src = os.path.join(ROOT, "data", "runs", name)
            if os.path.exists(src):
                shutil.copy2(src, folder)
        # Журнал прежнего прогона и подготовка едут датасетом, а не входом
        # от ядра. Kaggle не отдаёт вывод упавшего ядра как источник, и
        # продолжение рецепта на этом спотыкается ровно тогда, когда оно
        # нужнее всего — после падения.
        for rel in ("docs/raw/journal.json", "data/queries/train_prepared.jsonl"):
            src = os.path.join(ROOT, rel)
            if os.path.exists(src):
                shutil.copy2(src, folder)
                print(f"  в датасет: {os.path.basename(rel)}")

        # Список фрагментов той матрицы, по которой посчитаны прежние выдачи.
        # Им сверяется нарезка: при сдвиге границ идентификатор «акт#номер»
        # не исчезает, а указывает на другой текст, и выдачи молча
        # оказываются несравнимыми.
        ids = os.path.join(ROOT, "data", "embeddings", "base", "rufin", "ids.txt")
        if os.path.exists(ids):
            shutil.copy2(ids, os.path.join(folder, "ids_base.txt"))
            print("  в датасет: ids_base.txt для сверки нарезки")

        chunks = os.path.join(ROOT, "data", "chunks", "base.jsonl")
        if os.path.exists(chunks):
            import gzip
            with open(chunks, "rb") as fi, gzip.open(
                    os.path.join(folder, "chunks_base.jsonl.gz"), "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo, length=1 << 20)
        total = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder))
        print(f"в датасет уходит {len(os.listdir(folder))} файлов, "
              f"{total / 1048576:.0f} МБ")
        push_dataset(user, folder, QUERIES_DATASET, "ru-fin-queries",
                     "запросы, готовые выдачи и фрагменты")

    elif args.cmd == "weights":
        # Веса едут датасетом, а не выводом ядра. Вывод ядра не переживает
        # ни падения этого ядра, ни переход на другой аккаунт, а финальный
        # прогон без весов лучшего этапа сделать нельзя.
        источник = os.path.join(ROOT, "data", "weights")
        if not os.path.isdir(источник):
            raise SystemExit(f"нет {os.path.relpath(источник, ROOT)}: "
                             f"привезите веса с Kaggle")
        метки = args.метки or sorted(os.listdir(источник))
        folder = os.path.join(DIST, "weights")
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        всего = 0
        for метка in метки:
            откуда = os.path.join(источник, метка)
            if not os.path.isdir(откуда):
                raise SystemExit(f"нет весов этапа {метка}")
            shutil.copytree(откуда, os.path.join(folder, метка))
            размер = sum(os.path.getsize(os.path.join(б, и))
                         for б, _, ф in os.walk(откуда) for и in ф)
            всего += размер
            print(f"   {метка:<16} {размер / 1048576:>7.0f} МБ")
        print(f"   {'итого':<16} {всего / 1048576:>7.0f} МБ")
        push_dataset(user, folder, WEIGHTS_DATASET, "ru-fin-weights",
                     "веса этапов рецепта: " + ", ".join(метки))

    elif args.cmd == "run":
        ключи = flags_of(args.args)
        # вывод этапа A подключается как источник: там лежат матрицы эмбеддингов
        source = [f"{user}/{args.source}"] if args.source else []
        stages = {
            "a": ("run_phase_a.py", "ru-fin-phase-a", "ru-fin phase A", [], True),
            "export": ("export_chunks.py", "ru-fin-export-chunks", "ru-fin export chunks",
                       source, True),
            "b": ("run_phase_b.py", "ru-fin-phase-b", "ru-fin phase B", source, True),
            # Переранжирование отдельным маленьким прогоном: всё остальное
            # уже посчитано, и повторять его незачем.
            "rerank": ("gpu_rerank.py", "ru-fin-rerank-only", "ru fin rerank only", [], True),
            # Этап C поднимает языковую модель сам и матрицы этапа A не трогает:
            # вывод прежнего ядра ему не нужен, а подключённый — только лишние
            # гигабайты на монтирование и лишний источник старого кода.
            "c": ("run_phase_c.py", "ru-fin-phase-c", "ru-fin phase C", [], True),
            # D0 считается отдельным прогоном: он самый дорогой в этапе,
            # его файл нужен сразу двум этапам рецепта, и повторять его
            # ради перезапуска обучения незачем.
            "d0": ("gpu_prepare.py", "ru-fin-d0", "ru fin d0", [], True),
            # Рецепту нужен вывод подготовки: трудные негативы и оценки
            # учителя считаются один раз и подключаются входом.
            "d": ("run_phase_d.py", "ru-fin-phase-d", "ru-fin phase D",
                  [f"{user}/ru-fin-d0"], True),
            # Опыт с номерами актов: выдачи по вариантам живых вопросов.
            # Вывод прежних ядер не нужен, матрицы считаются здесь же
            # и кладутся в вывод — на этапе A их не сохранили, и это
            # стоило пересчёта корпуса при каждом опыте над запросами.
            "variants": ("run_variants.py", "ru-fin-variants", "ru fin variants",
                         [], True),
        }
        script, slug, title, kernels, gpu = stages[args.stage]
        if args.slug:
            # Kaggle требует, чтобы заголовок приводился к слагу, иначе
            # отказывается принимать ядро
            slug = args.slug
            title = args.slug.replace("-", " ")
        if args.no_gpu:
            gpu = False
            # Отдельное имя: иначе прогон без карты перезапишет версию с картой.
            # Заголовок правим вместе со слагом — Kaggle требует, чтобы одно
            # выводилось из другого, и иначе отказывается принимать ядро.
            if not args.slug:
                slug = f"{slug}-cpu"
                title = f"{title} cpu"
        datasets = [f"{user}/{DATASET}"]
        if args.stage in ("b", "c", "d0", "d", "rerank", "variants"):
            datasets.append(f"{user}/{QUERIES_DATASET}")
        # Веса подключаются только по просьбе. Нужны они одному финалу,
        # а весят под гигабайт, и Kaggle монтирует их перед стартом каждого
        # прогона: полторы минуты впустую на запуск, которых к тому же
        # не видно в журнале. Каждый этап обучается с исходных весов,
        # так что всем прочим шагам они безразличны.
        if args.weights_input:
            проба = run(["kaggle", "datasets", "status",
                         f"{user}/{WEIGHTS_DATASET}"])
            if проба.returncode != 0 or "error" in (проба.stdout or "").lower():
                raise SystemExit(
                    f"запрошены веса входом, но датасета {user}/{WEIGHTS_DATASET} "
                    f"нет. Залить: python3 scripts/kaggle_run.py weights")
            datasets.append(f"{user}/{WEIGHTS_DATASET}")
            print(f"  веса входом: {user}/{WEIGHTS_DATASET}")
        # Подготовке нужны оба: ученик считает плотную часть выдачи,
        # учитель — оценки. Самому обучению учитель уже не нужен, его оценки
        # посчитаны один раз и лежат в train_prepared.jsonl, а судья не нужен
        # ни тому, ни другому: десять гигабайтов монтируются перед стартом
        # и стоят минут впустую. Остальным этапам модели входом не нужны,
        # у них свои источники весов.
        models = []
        if args.stage == "d0":
            models = [MODELS["e5-small"], MODELS["reranker"]]
        elif args.stage == "d":
            models = [MODELS["e5-small"]]
        # Этапу C модель нужна только когда он поднимает судью, и только эта:
        # генерация запросов идёт своими моделями, а монтирование десяти
        # гигабайтов перед каждым прогоном стоит минут впустую.
        packages = ("sentence-transformers",)
        if args.stage == "c" and "--judge" in ключи:
            # Судей двое, и монтировать обоих разом незачем: 18 ГиБ второго
            # стоят минут впустую в прогоне первого. Выбор по подстроке имени
            # в аргументах, потому что `--judge-model` это hub-идентификатор,
            # а вход задаётся слагом Kaggle, и связать их больше негде.
            # Судей в пробе больше двух, и вход выбирается по имени модели
            # в аргументах: 7B это те же веса, которыми идёт генерация.
            по_имени = (("32B", "judge2"), ("7B", "gen"), ("14B", "judge"))
            ключ = next((k for игла, k in по_имени if игла in args.args), "judge")
            models = [MODELS[ключ]]
            # Судью поднимает vLLM, а его в образе Kaggle нет: пакеты
            # выбираются ниже одним правилом на все шаги, которые поднимают
            # языковую модель. Прежде это значило, что команда из
            # docs/11-polnota-razmetki.md падала на импорте уже на карте.
            #
            # Без sentence-transformers, и это не экономия. Установка vLLM
            # подтягивает свой transformers, а `pip install -U
            # sentence-transformers` следом сдвигает его ещё раз, и transformers
            # перестаёт импортироваться вовсе (`Could not import module
            # 'TrainerCallback'`). Судье sentence-transformers не нужен:
            # кросс-энкодер поднимает только фильтр учителя, а он в этом
            # прогоне не участвует.
            #
            # torchaudio сюда не ставится: на PyPI он собран под CUDA 12.8,
            # а torch от vLLM приезжает под 13.0, и `-U torchaudio` ничего
            # не меняет, потому что новее на PyPI нет. Этот конфликт лечит
            # сама ячейка подготовки, после проверки импорта.
        if args.stage == "c" and "--teacher-filter" in ключи:
            # Фильтр учителя поднимает кросс-энкодер, и без входа он поехал бы
            # за ним в хаб. Токена у ядра, созданного командой, нет вовсе, и
            # скачивание там режется по скорости: прогон встаёт, а квота идёт.
            models = models + [MODELS["reranker"]]
        if args.stage == "c":
            # Генерация вопросов тоже поднимает vLLM, а не только судья, и в
            # образе Kaggle его нет. Прогон E4 на этом упал: ставился
            # sentence-transformers, а падало на `import vllm` через минуту
            # после старта.
            нужен_vllm = bool(ключи & {"--judge", "--train", "--dev", "--test"})
            if нужен_vllm and "--teacher-filter" in ключи:
                raise SystemExit(
                    "в одном прогоне просят и языковую модель, и фильтр учителя. "
                    "Ставить vLLM и sentence-transformers вместе нельзя: второй "
                    "сдвигает transformers, который привёз первый, и transformers "
                    "перестаёт импортироваться вовсе. Разнесите на два прогона.")
            if нужен_vllm:
                packages = ("vllm",)
            # Генератор вопросов подключается входом, как и судья. Без этого
            # он поехал бы за весами в хаб без токена и встал бы там.
            if ключи & {"--train", "--dev", "--test"}:
                models = models + [MODELS["gen"]]
        # Опыту нужна только исходная модель ученика: учитель весит 2,29 ГБ,
        # и Kaggle монтирует его перед каждым запуском впустую.
        if args.stage == "variants":
            models = [MODELS["e5-small"]]
        if args.kernels:
            kernels = list(kernels) + [k if "/" in k else f"{user}/{k}"
                                       for k in args.kernels]
        ref = push_kernel(user, slug, title, script, args.args, datasets, kernels,
                          gpu, args.machine, args.notebook or "", models, packages)
        if not args.no_wait:
            state = wait(ref)
            print(f"\nсостояние: {state}")
            if state == "complete":
                fetch(ref, os.path.join(ROOT, "data", "kaggle_out", slug.split("/")[-1]))

    elif args.cmd == "time":
        for ряд in quota():
            if ряд.get("resource") == "GPU":
                print(f"квота GPU: потрачено {ряд.get('used')}, осталось "
                      f"{ряд.get('remaining')} из {ряд.get('total')}, "
                      f"обновится {ряд.get('refreshAt', '')[:10]}")
        ядра = kernel_times(user, args.filter)
        if args.running:
            ядра = [k for k in ядра
                    if "running" in k["состояние"].lower()
                    or "queued" in k["состояние"].lower()]
        if not ядра:
            print("таких ядер нет")
        for k in sorted(ядра, key=lambda k: k["старт"], reverse=True):
            состояние = k["состояние"].rsplit(".", 1)[-1].lower()
            строка = (f"{k['ref']:<40} {состояние:<10} "
                      f"старт {k['старт']:%d.%m %H:%M} UTC")
            # Остаток сессии печатается только тому, кто считается: у
            # закончившего ядра он ушел бы в минус и читался бы как ошибка.
            if состояние in ("running", "queued"):
                строка += (f", прошло {k['часов прошло']:.1f} ч, "
                           f"до предела сессии {k['часов осталось']:.1f} ч")
            print(строка)

    elif args.cmd == "watch":
        ref = args.slug if "/" in args.slug else f"{user}/{args.slug}"
        исход = watch(ref, args.deadline_hours, args.every, args.match)
        print(f"исход слежения: {исход}")
        if исход == "complete":
            fetch(ref, os.path.join(ROOT, "data", "kaggle_out",
                                    ref.split("/")[-1]))

    elif args.cmd == "fetch":
        ref = args.slug if "/" in args.slug else f"{user}/{args.slug}"
        fetch(ref, args.dest, args.log)


if __name__ == "__main__":
    main()
