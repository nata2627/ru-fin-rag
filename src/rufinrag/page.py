"""Главная страница: форма поиска поверх `POST /answer`.

Одна страница без сторонних библиотек: стили и скрипт лежат внутри, ответ
рисуется из полей `support` и `citations`. Поле `answer` страница не читает
вовсе: его первая строка объясняет, почему выключена генерация, и человеку с
улицы она ничего не говорит.

Текст ответа вставляется только через `textContent`: цитата это текст чужого
документа, и разметкой она стать не должна.
"""

from __future__ import annotations

from html import escape

REPO_URL = "https://github.com/nata2627/ru-fin-rag"

# Примеры проверены прогоном: на каждый из них сервис находит опору и отдает
# выдержки. Отказ здесь не показывается нарочно, его легко получить самому.
EXAMPLES = (
    "порядок формирования резервов по ссудам",
    "норматив достаточности капитала Н1.0",
    "обязательные резервы кредитных организаций",
    "идентификация клиентов при переводе денежных средств",
)

_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ru-fin-rag</title>
<style>
:root { --ink: #111827; --muted: #6b7280; --line: #e5e7eb; --accent: #2563eb;
        --soft: #f3f4f6; --ok-bg: #dcfce7; --ok: #166534; --no-bg: #fee2e2; --no: #991b1b; }
* { box-sizing: border-box; }
body { margin: 0; background: #fff; color: var(--ink);
       font: 16px/1.5 -apple-system, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 52rem; margin: 0 auto; padding: 28px 16px 40px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: 14px; margin-bottom: 20px; }
form { display: flex; gap: 8px; border: 1.5px solid var(--accent); border-radius: 8px;
       padding: 6px 6px 6px 14px; }
input { flex: 1; min-width: 0; border: 0; outline: 0; font: inherit; font-size: 17px; }
button { background: var(--accent); color: #fff; border: 0; border-radius: 6px;
         padding: 8px 16px; font: inherit; font-size: 15px; cursor: pointer; }
button:disabled { opacity: .6; cursor: default; }
.examples { margin: 10px 0 0; font-size: 14px; color: var(--muted); }
.examples a { color: var(--accent); text-decoration: none; border-bottom: 1px dashed;
              margin-right: 12px; cursor: pointer; display: inline-block; }
.meta { display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0; }
.pill { background: var(--soft); border-radius: 999px; padding: 4px 12px; font-size: 13px; }
.pill.ok { background: var(--ok-bg); color: var(--ok); }
.pill.no { background: var(--no-bg); color: var(--no); }
.status { color: var(--muted); font-size: 15px; margin: 18px 0; }
.note { border: 1px solid var(--line); border-radius: 10px; padding: 14px 18px; }
.card { display: flex; gap: 16px; border: 1px solid var(--line); border-radius: 10px;
        padding: 16px 18px; margin-bottom: 12px; }
.rank { flex: none; width: 28px; height: 28px; border-radius: 50%; background: #eff6ff;
        color: #1d4ed8; font-weight: 600; font-size: 14px;
        display: flex; align-items: center; justify-content: center; }
.card > div:last-child { min-width: 0; }
.act { font-weight: 600; font-size: 15px; }
.sec { color: #4b5563; font-size: 13.5px; margin: 2px 0 8px; }
.q { font-size: 14.5px; color: var(--ink); border-left: 3px solid #93c5fd; padding-left: 12px;
     overflow-wrap: anywhere; }
.src { font-size: 12px; margin-top: 8px; }
.src a { color: #6b7280; overflow-wrap: anywhere; }
details { margin-top: 32px; font-size: 14px; color: #374151; }
pre { background: var(--soft); border-radius: 4px; padding: 10px; overflow-x: auto;
      font-size: 13px; }
footer { margin-top: 28px; padding-top: 14px; border-top: 1px solid var(--line);
         font-size: 14px; color: var(--muted); }
footer a { color: var(--accent); margin-right: 16px; }
</style></head>
<body><main>
<h1>Поиск по нормативным актам Банка России</h1>
<div class="sub">__STATS__, ответ из выдержек со ссылкой на акт и пункт</div>
<form id="ask">
<input id="query" name="query" maxlength="__MAX__" autocomplete="off" required
       placeholder="Например, порядок формирования резервов по ссудам">
<button id="go" type="submit">Найти</button>
</form>
<div class="examples">Примеры: __EXAMPLES__</div>
<div id="out"></div>
<details>
<summary>Тот же запрос через API</summary>
<pre>curl -s -X POST /answer \\
  -H 'Content-Type: application/json' \\
  -d '{"query": "порядок формирования резервов по ссудам"}'</pre>
<p>Это бесплатный тариф: 512 МБ памяти, сон после 15 минут без запросов. Плотная ступень
поиска в онлайне выключена, выдача лексическая (BM25).</p>
</details>
<footer>
<a href="/docs">/docs</a><a href="/health">/health</a><a href="__REPO__">Код и отчет с числами</a>
</footer>
</main>
<script>
"use strict";
const out = document.getElementById("out");
const input = document.getElementById("query");
const button = document.getElementById("go");
// Render будит уснувший сервис около 30 секунд, и все это время запрос
// висит без ответа или получает 502/503 от прокси. Такие ответы повторяются,
// пока не выйдет окно: иначе первый запрос после сна выглядел бы поломкой.
const WAKE_HINT_MS = 2500;
const WAKE_WINDOW_MS = 90000;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function num(x) {
  return x.toFixed(2).replace(".", ",");
}

function status(text) {
  out.replaceChildren(el("div", "status", text));
}

function render(body) {
  const s = body.support;
  const meta = el("div", "meta");
  meta.append(
    el("span", body.answered ? "pill ok" : "pill no",
       body.answered ? "опора в корпусе есть" : "опоры в корпусе нет"),
    el("span", "pill", "выдержек " + body.citations.length),
    el("span", "pill", "покрытие запроса " + num(s.value) + " при пороге " + num(s.threshold)),
  );
  const nodes = [meta];
  if (!body.answered) {
    nodes.push(el("div", "note", s.hits
      ? "В актах нет фрагмента, который покрывал бы слова этого запроса. Сервис не "
        + "подбирает ответ наугад. Попробуйте сформулировать вопрос словами нормативного "
        + "текста или назвать акт."
      : "Поиск не нашел ни одного фрагмента со словами этого запроса. Попробуйте "
        + "сформулировать вопрос иначе."));
  }
  for (const c of body.citations) {
    const card = el("div", "card");
    const text = el("div");
    let sec = c.section;
    if (c.clauses.length) {
      sec += (sec ? ", " : "") + (c.clauses.length === 1 ? "пункт " : "пункты ")
        + c.clauses.join(", ");
    }
    const src = el("div", "src");
    const link = el("a", "", "cbr.ru · " + c.chunk_id);
    link.href = c.url;
    link.target = "_blank";
    link.rel = "noopener";
    src.append(link);
    text.append(el("div", "act", c.act), el("div", "sec", sec),
                el("div", "q", "«" + c.quote + "»"), src);
    card.append(el("div", "rank", String(c.rank)), text);
    nodes.push(card);
  }
  out.replaceChildren(...nodes);
}

async function ask(query) {
  button.disabled = true;
  status("Ищу…");
  const hint = setTimeout(() => status("Сервис просыпается после сна, это около 30 секунд…"),
                          WAKE_HINT_MS);
  const deadline = Date.now() + WAKE_WINDOW_MS;
  try {
    for (;;) {
      let response = null;
      try {
        response = await fetch("/answer", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query: query}),
        });
      } catch (err) {
        // Сеть оборвалась: на пробуждении так бывает, повторяем ниже.
      }
      if (response && response.ok) {
        render(await response.json());
        return;
      }
      if (response && response.status === 422) {
        status("Запрос не принят: он пустой или длиннее __MAX__ символов.");
        return;
      }
      const waking = !response || [502, 503, 504].includes(response.status);
      if (!waking || Date.now() > deadline) {
        status("Сервис сейчас не отвечает. Попробуйте еще раз через минуту.");
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  } finally {
    clearTimeout(hint);
    button.disabled = false;
  }
}

document.getElementById("ask").addEventListener("submit", (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (query) ask(query);
});
for (const link of document.querySelectorAll(".examples a")) {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    input.value = link.textContent;
    ask(link.textContent);
  });
}
</script>
</body></html>"""


def _thousands(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def render(chunks: int | None, acts: int | None, max_query_chars: int) -> str:
    stats = (
        f"{_thousands(acts)} актов, {_thousands(chunks)} фрагментов"
        if chunks is not None and acts is not None
        else "Индекс не поднялся, причина в /health"
    )
    examples = "".join(f'<a href="#">{escape(q)}</a>' for q in EXAMPLES)
    return (
        _PAGE.replace("__STATS__", stats)
        .replace("__EXAMPLES__", examples)
        .replace("__MAX__", str(max_query_chars))
        .replace("__REPO__", REPO_URL)
    )
