-- Трафик по суткам: сколько запросов пришло и чем они кончились.
--
-- Окно, а не группировка. Группировка отвечает только про свои сутки, а
-- трафик читают рядом: сколько накопилось всего и каким был последний
-- недельный уровень. Накопление (`queries_total`) и недельная сумма
-- (`queries_7d`) группировкой не считаются вовсе, разница с прошлыми сутками
-- потребовала бы соединения таблицы с собой по сдвигу даты.
--
-- Окно недели идет по номеру суток (`range`), а не по строкам (`rows`): сутки
-- без запросов строки не создают, и `rows between 6 preceding` считал бы
-- «последние семь суток с трафиком» вместо последних семи суток. Рядом стоит
-- `days_in_7d`, то есть сколько суток в окно на самом деле попало: это
-- знаменатель недельного числа, и без него дыру в ряду не видно.

with by_day as (
    select
        day,
        count() as queries,
        countIf(outcome = 'ok') as answered,
        countIf(outcome = 'rejected') as rejected,
        countIf(outcome = 'unavailable') as unavailable
    from {{ ref('stg_queries') }}
    group by day
)
select
    day,
    queries,
    answered,
    rejected,
    unavailable,
    sum(queries) over (
        order by day asc rows between unbounded preceding and current row
    ) as queries_total,
    sum(queries) over week as queries_7d,
    count() over week as days_in_7d,
    -- Для первых суток предыдущих нет, и ноль здесь означал бы «трафик не
    -- изменился», чего никто не измерял.
    if(
        row_number() over (order by day asc) = 1,
        null,
        queries - lagInFrame(queries) over (
            order by day asc rows between unbounded preceding and current row
        )
    ) as queries_change
from by_day
window week as (
    order by toUInt32(toDate(day)) asc range between 6 preceding and current row
)
order by day
