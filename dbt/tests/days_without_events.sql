-- Пропуск суток в ряду трафика.
--
-- Это предупреждение, а не поломка: бесплатный хостинг засыпает, и сутки без
-- единого запроса там нормальны. Вред приносит чтение такого ряда как
-- непрерывного: суточные доли в дашборде лежат подряд, и пропуск выглядит
-- так, как будто его нет. Недельные окна витрины от дыры не портятся, они идут по
-- номеру суток (`range`), а не по строкам, и показывают рядом `days_in_7d`.
--
-- Поэтому строгость `warn`: тест называет пропущенные сутки в выводе прогона и
-- не красит прогон в красный. Красный прогон здесь означал бы «сервиса никто
-- не спрашивал», то есть останавливал бы сборку из-за отсутствия трафика.

{{ config(severity = 'warn') }}

with ordered as (
    select
        day,
        lagInFrame(day) over (
            order by day asc rows between unbounded preceding and current row
        ) as previous_day,
        row_number() over (order by day asc) as position
    from {{ ref('traffic_by_day') }}
)
select
    previous_day,
    day,
    dateDiff('day', toDate(previous_day), toDate(day)) - 1 as missing_days
from ordered
where position > 1
    and dateDiff('day', toDate(previous_day), toDate(day)) > 1
