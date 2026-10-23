-- Задержка по ступеням поиска в квантилях. Ступени приходят картой, поэтому
-- строка на ступень получается развертыванием карты, а не перечислением
-- колонок.
--
-- `quantileExact`, а не `quantile`: приближенный квантиль ClickHouse считает по
-- выборке из потока, и два режима хранения дали бы разные числа на одних
-- данных по причине, которая к хранению отношения не имеет.

select
    stage,
    count() as runs,
    round(quantileExact(0.5)(ms), 3) as p50_ms,
    round(quantileExact(0.95)(ms), 3) as p95_ms,
    round(max(ms), 3) as max_ms
from (
    select
        arrayJoin(timings_ms) as pair,
        pair.1 as stage,
        pair.2 as ms
    from {{ ref('stg_queries') }}
    where outcome = 'ok'
)
group by stage
order by stage
