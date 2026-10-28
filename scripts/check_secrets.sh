#!/usr/bin/env bash
# Шаблоны секретов и личных данных в том, что уходит наружу.
#
# Проверяются четыре места, а не одно. Рабочее дерево это то, что видно в
# репозитории сегодня, но публикуется вместе с историей: ключ, удаленный
# следующим коммитом, остается в истории навсегда и достается одной командой.
# Третье место это сам `.env`: он должен быть закрыт `.gitignore` и не
# встречаться в истории ни одним коммитом. Четвертое это выгрузки и логи: они
# не под контролем git, но уезжают на HuggingFace и в отчеты.
#
# Чего этот скрипт сознательно не делает: он не читает значения из `.env`,
# чтобы искать их в файлах. Такая проверка выглядит сильнее, но кладет секрет
# в переменную, и любой запуск с трассировкой (`bash -x`) или случайный вывод
# ошибки печатает ключ целиком. Проверено на себе 27.10.2026: ключ vast.ai
# уехал в вывод и его пришлось пересоздавать. Опознать ключ вида
# шестнадцатеричной строки по шаблону все равно нельзя: он неотличим от
# sha256, которых в документах проекта десятки. Поэтому защита здесь другая:
# `.env` закрыт и в историю не попадал, а печатать его запрещено guard.
#
# Имена переменных только латиницей: bash принимает кириллическое имя за
# команду, печатает "command not found" и идет дальше, не упав.
set -euo pipefail

cd "$(dirname "$0")/.."

patterns='hf_[A-Za-z0-9]{34}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{70,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----'
# Чужая почта в публичном репозитории это личные данные, а не секрет, и попасть
# туда ей проще всего: имя машины, подпись в примере, адрес из переписки.
#
# Домены из RFC 2606 (`example.com`, `.invalid`, `.test`, `.localhost`) выброшены
# из шаблона нарочно: они зарезервированы именно под примеры, за ними нет ничьего
# почтового ящика, и образец в документации это законное их применение. Без этого
# исключения проверка падала на собственном тесте, который подкладывает
# `petrov@example.com` и ждет отказа, то есть ловила сама себя.
mail_pattern='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
mail_allow='@example\.(com|org|net)|\.(example|invalid|test|localhost)($|[^A-Za-z0-9])'
# Домашний путь с именем пользователя внутри публичного файла это не утечка, но
# это мусор, который читающий видит первым: он описывает чужой диск, а не проект.
home_pattern='(/Users/|/home/)[a-z][a-z0-9._-]+/'

bad=0

tree_hits=$(git ls-files -z | xargs -0 grep -InE "$patterns" 2>/dev/null || true)
if [ -n "$tree_hits" ]; then
  echo "рабочее дерево: похожее на секрет" >&2
  echo "$tree_hits" >&2
  bad=1
fi

mail_hits=$(git ls-files -z | xargs -0 grep -InE "$mail_pattern" 2>/dev/null \
  | grep -vE "$mail_allow" || true)
if [ -n "$mail_hits" ]; then
  echo "рабочее дерево: адрес почты, проверьте чей он" >&2
  echo "$mail_hits" >&2
  bad=1
fi

# История: все блобы всех ветвей. Большие файлы корпуса пропускаются по
# размеру, ключей в них быть не может, а чтение стоило бы минуты.
# `/home/runner/` это путь машины GitHub Actions, он попадает в замеры
# регламента как измеренный факт и утечкой не является.
home_hits=$(git ls-files -z | xargs -0 grep -InE "$home_pattern" 2>/dev/null \
  | grep -v '/home/runner/' || true)
if [ -n "$home_hits" ]; then
  echo "рабочее дерево: домашний путь с именем пользователя" >&2
  echo "$home_hits" >&2
  bad=1
fi

history_hits=$(git rev-list --objects --all \
  | awk '{print $1}' \
  | git cat-file --batch-check='%(objectname) %(objecttype) %(objectsize)' 2>/dev/null \
  | awk '$2 == "blob" && $3 < 1000000 {print $1}' \
  | git cat-file --batch 2>/dev/null \
  | grep -acE "$patterns" || true)
if [ "${history_hits:-0}" -gt 0 ]; then
  echo "история git: похожее на секрет, совпадений $history_hits" >&2
  bad=1
fi

# Сам файл с ключами: закрыт ли он и не попадал ли в историю.
if [ -f .env ] && ! git check-ignore -q .env; then
  echo ".env не закрыт .gitignore" >&2
  bad=1
fi
if [ -n "$(git log --all --oneline -- .env 2>/dev/null)" ]; then
  echo ".env встречается в истории git" >&2
  bad=1
fi
# `.env.example` и подобные образцы лежат в репозитории нарочно: в них имена
# переменных без значений. Отсекаем их по суффиксу, а не по месту.
for tracked in $(git ls-files | grep -E '(^|/)\.env($|\.local$|\.production$)' || true); do
  echo "под контролем git лежит файл с ключами: $tracked" >&2
  bad=1
done
samples=$(git ls-files | grep -E '(^|/)\.env\.(example|sample|template)$' || true)
for sample in $samples; do
  filled=$(grep -cE '^[A-Z_]+=.+' "$sample" || true)
  if [ "${filled:-0}" -gt 0 ]; then
    echo "образец $sample содержит значения, а должен только имена" >&2
    bad=1
  fi
done

# Выгрузки и логи. Под контролем git их нет, но набор уехал на HuggingFace,
# а логи попадают в отчеты и в приложения к письмам.
for dir in logs results retrieval/logs retrieval/data/results retrieval/data/benchmark; do
  [ -d "$dir" ] || continue
  outside_hits=$(grep -rInE "$patterns" "$dir" 2>/dev/null | head -5 || true)
  if [ -n "$outside_hits" ]; then
    echo "выгрузки и логи ($dir): похожее на секрет" >&2
    echo "$outside_hits" >&2
    bad=1
  fi
done

if [ "$bad" -ne 0 ]; then
  exit 1
fi
echo "секретов не найдено"
