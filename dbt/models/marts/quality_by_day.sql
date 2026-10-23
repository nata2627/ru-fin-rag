-- Качество суток: доля оценок рядом со знаменателем, разбивка оценок на
-- «вверх» и «вниз» и задержка ответа.
--
-- Окно, а не группировка. На трафике открытого демо одни сутки это десятки
-- запросов и единицы оценок, и доля за сутки скачет так, что по одной точке
-- вывод делать нельзя. Поэтому рядом с суточными числами стоит то же самое за
-- семь календарных суток: такую сумму группировка по суткам не дает, а
-- соединение с собой по диапазону дат на каждом шаге читало бы таблицу
-- заново.
--
-- Окно идет по номеру суток (`range`), а не по строкам: сутки без запросов
-- строки не создают, и `rows` считал бы «последние семь суток с трафиком».
-- `days_in_7d` показывает, сколько суток в окно попало, то есть знаменатель
-- недельных чисел.
--
-- Знаменатель оценок берется из `rated_share`, а не пересчитывается: модель
-- отвечает на тот же вопрос, и две реализации одного знаменателя разошлись бы
-- на первой же правке.

with marks as (
    select
        q.query_day as day,
        countIf(f.rating = 'up') as up,
        countIf(f.rating = 'down') as down
    from {{ ref('stg_feedback') }} as f
    inner join (
        select request_id, day as query_day
        from {{ ref('stg_queries') }}
    ) as q using (request_id)
    group by day
),

latency as (
    select
        day,
        round(quantileExact(0.95)(total_ms), 3) as p95_total_ms
    from {{ ref('stg_queries') }}
    where outcome = 'ok'
    group by day
),

joined as (
    select
        s.day as day,
        s.answered as answered,
        s.rated as rated,
        ifNull(m.up, 0) as up,
        ifNull(m.down, 0) as down,
        ifNull(l.p95_total_ms, 0) as p95_total_ms
    from {{ ref('rated_share') }} as s
    left join marks as m using (day)
    left join latency as l using (day)
)

select
    day,
    answered,
    rated,
    up,
    down,
    p95_total_ms,
    sum(answered) over week as answered_7d,
    sum(rated) over week as rated_7d,
    sum(up) over week as up_7d,
    count() over week as days_in_7d,
    max(p95_total_ms) over week as p95_worst_7d
from joined
window week as (
    order by toUInt32(toDate(day)) asc range between 6 preceding and current row
)
order by day
