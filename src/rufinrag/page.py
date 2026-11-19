"""Главная страница: форма поиска поверх `POST /answer`.

Одна страница без сторонних библиотек: стили и скрипт лежат внутри, ответ
рисуется из полей `support` и `citations`. Поле `answer` страница не читает
вовсе: его первая строка объясняет, почему выключена генерация, и человеку с
улицы она ничего не говорит.

Текст ответа вставляется только через `textContent` и текстовые узлы: цитата
это текст чужого документа, и разметкой она стать не должна.
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
:root { --bg1: #eef3fb; --bg2: #e3ebf7; --panel: #fff; --ink: #111827; --muted: #6b7280;
        --line: #e5e7eb; --accent: #2563eb; --accent-ink: #1d4ed8; --accent-soft: #eff6ff;
        --quote: #93c5fd; --soft: #f3f4f6; --ok-bg: #dcfce7; --ok: #166534;
        --no-bg: #fee2e2; --no: #991b1b; --mark: #fef08a; --shadow: rgba(30, 50, 90, .14); }
@media (prefers-color-scheme: dark) {
  :root { --bg1: #0b1220; --bg2: #111a2e; --panel: #151e31; --ink: #e5e7eb; --muted: #9ca3af;
          --line: #263046; --accent: #3b82f6; --accent-ink: #93c5fd; --accent-soft: #1e2a44;
          --quote: #3b82f6; --soft: #1f2940; --ok-bg: #12351f; --ok: #86efac;
          --no-bg: #3b1518; --no: #fca5a5; --mark: #854d0e; --shadow: rgba(0, 0, 0, .4); }
}
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; color: var(--ink);
       background: linear-gradient(135deg, var(--bg1), var(--bg2)) fixed;
       font: 16px/1.5 -apple-system, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 56rem; margin: 0 auto; padding: 40px 16px; }
.panel { background: var(--panel); border-radius: 14px; box-shadow: 0 20px 50px var(--shadow);
         padding: 32px 36px 28px; }
@media (max-width: 600px) { .panel { padding: 22px 18px; } main { padding: 16px; } }
.brand { font-size: 13px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
         color: var(--accent-ink); margin-bottom: 6px; }
h1 { font-size: 26px; line-height: 1.25; margin: 0 0 10px; }
.stats { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }
.stat { background: var(--soft); border-radius: 999px; padding: 3px 12px; font-size: 13px;
        color: var(--muted); }
.stat b { color: var(--ink); font-weight: 600; }
form { display: flex; gap: 8px; border: 1.5px solid var(--accent); border-radius: 10px;
       padding: 6px 6px 6px 16px; background: var(--panel);
       box-shadow: 0 0 0 4px var(--accent-soft); }
input { flex: 1; min-width: 0; border: 0; outline: 0; font: inherit; font-size: 17px;
        color: var(--ink); background: transparent; }
button { background: var(--accent); color: #fff; border: 0; border-radius: 7px;
         padding: 9px 20px; font: inherit; font-size: 15px; font-weight: 600; cursor: pointer; }
button:hover { filter: brightness(1.08); }
button:disabled { opacity: .6; cursor: default; }
.examples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; align-items: center; }
.examples span { font-size: 13px; color: var(--muted); margin-right: 2px; }
.chip { font-size: 13.5px; color: var(--accent-ink); background: var(--accent-soft);
        border-radius: 999px; padding: 4px 12px; text-decoration: none; }
.chip:hover { background: var(--accent); color: #fff; }
.how { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 28px; }
@media (max-width: 600px) { .how { grid-template-columns: 1fr; } }
.how div { border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
           font-size: 13.5px; color: var(--muted); }
.how b { display: block; color: var(--ink); font-size: 14.5px; margin-bottom: 2px; }
.meta { display: flex; flex-wrap: wrap; gap: 8px; margin: 22px 0 16px; }
.pill { background: var(--soft); border-radius: 999px; padding: 4px 12px; font-size: 13px; }
.pill.ok { background: var(--ok-bg); color: var(--ok); font-weight: 600; }
.pill.no { background: var(--no-bg); color: var(--no); font-weight: 600; }
.status { color: var(--muted); font-size: 15px; margin: 22px 0 12px; }
.note { border: 1px solid var(--line); border-radius: 10px; padding: 14px 18px; }
.card { display: flex; gap: 16px; border: 1px solid var(--line); border-radius: 12px;
        padding: 18px 20px; margin-bottom: 12px; transition: border-color .15s; }
.card:hover { border-color: var(--quote); }
.rank { flex: none; width: 30px; height: 30px; border-radius: 50%; background: var(--accent-soft);
        color: var(--accent-ink); font-weight: 600; font-size: 14px;
        display: flex; align-items: center; justify-content: center; }
.body { min-width: 0; flex: 1; }
.act { font-weight: 600; font-size: 15.5px; }
.sec { color: var(--muted); font-size: 13.5px; margin: 2px 0 10px; }
.q { font-size: 14.5px; line-height: 1.6; border-left: 3px solid var(--quote);
     padding-left: 12px; overflow-wrap: anywhere; }
mark { background: var(--mark); color: inherit; border-radius: 3px; padding: 0 1px; }
.src { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; margin-top: 12px;
       font-size: 12.5px; color: var(--muted); }
.src a { color: var(--accent-ink); text-decoration: none; font-weight: 600; }
.src a:hover { text-decoration: underline; }
.skeleton { height: 112px; border-radius: 12px; margin-bottom: 12px;
            background: linear-gradient(90deg, var(--soft) 25%, var(--line) 50%, var(--soft) 75%);
            background-size: 200% 100%; animation: shine 1.2s infinite linear; }
@keyframes shine { to { background-position: -200% 0; } }
details { margin-top: 28px; font-size: 14px; color: var(--muted); }
summary { cursor: pointer; }
pre { background: var(--soft); color: var(--ink); border-radius: 6px; padding: 10px;
      overflow-x: auto; font-size: 13px; }
footer { display: flex; flex-wrap: wrap; gap: 8px 18px; margin-top: 20px; padding: 0 4px;
         font-size: 14px; color: var(--muted); }
footer a { color: var(--accent-ink); text-decoration: none; }
footer a:hover { text-decoration: underline; }
</style></head>
<body><main>
<div class="panel">
<div class="brand">ru-fin-rag</div>
<h1>Поиск по нормативным актам Банка России</h1>
<div class="stats">__STATS__</div>
<form id="ask">
<input id="query" name="query" maxlength="__MAX__" autocomplete="off" required
       placeholder="Например, порядок формирования резервов по ссудам">
<button id="go" type="submit">Найти</button>
</form>
<div class="examples"><span>Примеры:</span>__EXAMPLES__</div>
<div id="out">
<div class="how">
<div><b>Поиск</b>По словам запроса среди фрагментов актов «Вестника Банка России».</div>
<div><b>Проверка опоры</b>Если слова запроса почти не встречаются в актах, сервис
отказывает вместо ответа наугад.</div>
<div><b>Цитата и источник</b>Каждая выдержка с актом, главой, пунктами и ссылкой на
cbr.ru.</div>
</div>
</div>
<details>
<summary>Тот же запрос через API</summary>
<pre>curl -s -X POST /answer \\
  -H 'Content-Type: application/json' \\
  -d '{"query": "порядок формирования резервов по ссудам"}'</pre>
<p>Это бесплатный тариф: 512 МБ памяти, сон после 15 минут без запросов. Плотная ступень
поиска в онлайне выключена, выдача лексическая (BM25).</p>
</details>
</div>
<footer>
<a href="/docs">Документация API</a><a href="/health">Состояние сервиса</a>
<a href="__REPO__">Код и отчет с числами</a>
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

function status(text, skeleton) {
  const nodes = [el("div", "status", text)];
  if (skeleton) for (let i = 0; i < 3; i++) nodes.push(el("div", "skeleton"));
  out.replaceChildren(...nodes);
}

// Подсветка грубая: по началу слова, без морфологии. Ее дело показать глазу,
// за что зацепился поиск, а точное совпадение считает сервис.
function stems(query) {
  return query.toLowerCase().split(/[^0-9a-zа-яё]+/i)
    .filter((w) => w.length >= 4)
    .map((w) => w.slice(0, Math.max(4, Math.min(w.length - 2, 7))));
}

function highlighted(text, roots) {
  const node = el("div", "q");
  const words = text.split(/([0-9A-Za-zА-Яа-яЁё]+)/);
  for (const part of words) {
    const lower = part.toLowerCase();
    if (part && roots.some((r) => lower.startsWith(r))) {
      node.append(el("mark", "", part));
    } else if (part) {
      node.append(document.createTextNode(part));
    }
  }
  return node;
}

function render(body, query) {
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
  const roots = stems(query);
  for (const c of body.citations) {
    const card = el("div", "card");
    const text = el("div", "body");
    let sec = c.section;
    if (c.clauses.length) {
      sec += (sec ? ", " : "") + (c.clauses.length === 1 ? "пункт " : "пункты ")
        + c.clauses.join(", ");
    }
    const src = el("div", "src");
    const link = el("a", "", "Открыть на cbr.ru ↗");
    link.href = c.url;
    link.target = "_blank";
    link.rel = "noopener";
    src.append(link, el("span", "", "фрагмент " + c.chunk_id));
    text.append(el("div", "act", c.act), el("div", "sec", sec),
                highlighted("«" + c.quote + "»", roots), src);
    card.append(el("div", "rank", String(c.rank)), text);
    nodes.push(card);
  }
  out.replaceChildren(...nodes);
}

async function ask(query) {
  button.disabled = true;
  status("Ищу…", true);
  const hint = setTimeout(
    () => status("Сервис просыпается после сна, это около 30 секунд…", true), WAKE_HINT_MS);
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
        render(await response.json(), query);
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
    if chunks is not None and acts is not None:
        stats = (
            f'<span class="stat"><b>{_thousands(acts)}</b> актов</span>'
            f'<span class="stat"><b>{_thousands(chunks)}</b> фрагментов</span>'
            '<span class="stat">ответ из выдержек со ссылкой на акт и пункт</span>'
        )
    else:
        stats = '<span class="stat">Индекс не поднялся, причина в /health</span>'
    examples = "".join(f'<a class="chip" href="#">{escape(q)}</a>' for q in EXAMPLES)
    return (
        _PAGE.replace("__STATS__", stats)
        .replace("__EXAMPLES__", examples)
        .replace("__MAX__", str(max_query_chars))
        .replace("__REPO__", REPO_URL)
    )
