# Выводы прогонов

Каждое число публичного текста лежит здесь, в файле, созданном кодом из этого репозитория. Исключения названы в строках: файл, который пишется руками, помечен «руками».

Колонка «Чем запускается» это либо цель Make, либо шаг ночного прогона (`.github/workflows/nightly.yml`), либо другой прогон, либо «вручную»: команда из колонки «Команда» запускается человеком, и ни один прогон ее не зовет.

| Файл | Команда | Чем запускается | Что в нем |
|---|---|---|---|
| `artifacts.md` | руками по замерам, команды в тексте | руками, не запускается | инвентаризация артефактов части А под сервис |
| `reproducibility.json` | `scripts/check_reproducibility.py` | вручную | совпадение выдачи сервиса с выдачей бенчмарка части А |
| `vectors.json` | `scripts/extend_vectors.py` | `make vectors` | догон плотных векторов до выросшего корпуса и сверка с матрицей части А по косинусу |
| `rerank-hybrid-rufin.json`, `rerank-bm25-rufin.json` | `scripts/rerank_run.py` | вручную на видеокарте (Kaggle или арендованная машина), в CI не идет | цена переранжирования на видеокарте: пары, время, скорость, пик видеопамяти |
| `rerank-quality.json`, `rerank-quality-bm25.json`, `rerank-first-stage.json` | `scripts/evaluate_run.py` | вручную после `rerank_run.py`, в CI не идет | качество связок с кросс-энкодером и прибавки с парными интервалами |
| `latency-local.json` | `scripts/measure_latency.py` | вручную | задержка связок поиска на маке по ступеням |
| `vector-store.json` | `scripts/compare_vector_stores.py` | вручную | Qdrant против FAISS против numpy: задержка, память, совпадение выдач |
| `quantization.json` | `scripts/measure_quantization.py` | вручную | квантизация векторов в int8: память и потеря качества с интервалом |
| `load.json` | `scripts/measure_load.py` | вручную | нагрузочный замер: p50, p95, p99, RPS, узкое место |
| `warehouse.json` | `scripts/compare_warehouses.py` | `make warehouse` | витрина над журналом и сверка двух режимов хранения: ClickHouse сервером против файлов через chdb |
| `logging.json`, `logging.md` | `scripts/measure_logging.py` создает `logging.json`; `logging.md` пишется руками по нему, команда прогона записана в самом тексте | вручную | чего стоит журнал: скорость записи, доля в задержке запроса, размер события, месяц по экстраполяции, поведение при отказе записи |
| `dashboard.json` | `scripts/measure_dashboard.py` | вручную | страница дашборда: чисел на ней, графиков, вес, сборка, поведение на пустом журнале, и список напечатанных чисел для пересборки |
| `regression.json`, `regression-baseline.json` | `scripts/run_regression.py` | ночной прогон (`nightly`, шаг «Regression on the frozen benchmark»); эталон только вручную с `--update-baseline` | качество на замороженном наборе и эталон, с которым его сравнивает каждый плановый прогон |
| `alert-threshold.json` | `scripts/measure_alert_threshold.py` | вручную | разброс метрики от прогона к прогону и от доливки корпуса, из него выведен порог алерта |
| `alert-drill.json` | `scripts/check_alert.py` | `make alert` | учебная деградация метрики: на какой потере алерт сработал и на какой промолчал |
| `dag-rerun.json`, `dag-rerun-ci.json` | `scripts/check_dag_rerun.py` | `make dag` (`scripts/run_dag.sh`); второй файл берется из артефакта прогона `dag.yml` | два прогона DAG подряд: время прохода, пик памяти, пропуск шагов на повторе. Второй файл это тот же замер в прогоне CI, взят из артефакта прогона |
| `dag-recovery.json` | `scripts/check_dag_recovery.py` | `make dag-drill` | учебное падение шага в середине цепочки: попытки, незапущенные задачи, целость прошлого прогона, восстановление |
| `dag.md` | `scripts/report_dag.py` | `make dag-report` и ночной прогон (шаг «Rebuild the regulation report») | три замера регламента одним отчетом: время прохода, акты за прогон, падение шага |
| `monitoring-thresholds.json` | `scripts/measure_monitoring_thresholds.py` | `make monitoring-thresholds` | пороги мониторинга трафика: разброс PSI по окнам из эталонного набора, порог дрейфа, признак опоры и порог доли запросов без хорошего ответа |
| `monitoring-drill.json` | `scripts/drill_monitoring.py` | `make monitoring-drill` | учебная лестница мониторинга: на каком сдвиге алерт сработал, на каком промолчал и как спит правило на коротком окне |
| `monitoring.json` | `scripts/check_monitoring.py` | `make monitoring` и ночной прогон (шаг «Monitor query drift») | состояние мониторинга в ночном прогоне: окно журнала, три сигнала и текст алерта для issue |
| `sample-size.json` | `scripts/plan_sample_size.py` | `make sample-size` | правило остановки сравнения двух конфигураций, записанное до данных, нужный размер выборки офлайн и онлайн, цена поправки на подглядывание и сколько живого трафика есть |
| `vector-store.md`, `load.md`, `free-tier.md` | руками | руками, не запускаются | тексты отчетов. `vector-store.md` и `load.md` ссылаются на `vector-store.json`, `quantization.json`, `load.json`, из которых берут числа; `free-tier.md` сверка бесплатных тарифов хостингов 21.10.2026 по их документации, числа не измерение |
| `vector-store-container.json` | `scripts/compare_vector_stores.py --qdrant-url ... --qdrant-container ... --out results/vector-store-container.json` | вручную, нужен `docker compose --profile vectors` | Qdrant контейнером по сети против встроенного режима: задержка, диск и память контейнера |
| `corpus.json` | `scripts/describe_corpus.py` | ночной прогон (шаг «Snapshot the corpus») | фрагменты, акты, версия и размер корпуса, прежний снимок в поле `replaced` |
| `corpus-published.json` | `scripts/corpus_numbers.py --write` (первый раз `--init`) | ночной прогон (шаг «Rewrite the corpus numbers») | корпусные числа, которые сейчас стоят в README: по ним скрипт находит и заменяет старые токены |
| `index.json` | `scripts/build_index.py` | ночной прогон (шаг «Rebuild index») | словарь, время сборки и размеры лексического индекса |
| `corpus-update.json` | `scripts/update_corpus.py` | ночной прогон (шаг «Fetch new acts») | что докачал последний прогон: новые акты, статус источника |
| `corpus-update-first.json`, `corpus-update-settled.json` | `scripts/freeze_settled_run.py` | вручную, разово | замороженные записи первой докачки и установившегося прогона, чтобы их числа не затирались ночью |
| `regulation.json` | `scripts/record_heartbeat.py` | ночной прогон (шаг «Record the heartbeat») | отметка живости: счетчик прогонов и подряд идущие отказы источника |
| `events.json` | `scripts/drain_events.py` | ночной прогон (шаг «Drain the query log into the repository») | состояние последней выгрузки журнала запросов |
| `events-rehearsal.json` | `scripts/rehearse_drain.py` | `make drain` | репетиция выгрузки журнала: сервис процессом, выгрузка по сети |
| `registry.json` | `scripts/register_versions.py` | `make registry` и ночной прогон (шаг «Rebuild the version registry») | реестр версий корпуса, индекса, весов и векторов |
| `tracking.json` | `scripts/track_runs.py` | `make track` | заведенные в MLflow прогоны из `results/*.json` |
| `acceptance.json` | `scripts/record_acceptance.py` | `acceptance.yml` (отдельный плановый прогон) | счет приемки автономности: подряд зеленые плановые прогоны |
| `answer.json` | `scripts/measure_answer.py` | `make answer-threshold` | порог отказа слоя ответа, поля цитаты, цена опоры |
| `answer-quality.json` | `scripts/measure_answer_quality.py` | `make answer-quality` | доля ответов со ссылкой на релевантный фрагмент и отказы |
| `router.json` | `scripts/measure_router.py` | `make router` | роутер против прямого поиска: качество и задержка |
| `graph.json` | `scripts/build_graph.py` | `make graph` | граф ссылок между актами: дуги, акты со ссылками внутрь |
| `graph-expansion.json`, `run-hybrid-graph.jsonl` | `scripts/measure_graph_expansion.py` | `make graph-expansion` | расширение выдачи по цитированию, отрицательный результат; второй файл это выдача с расширением |
| `compare-configs.json` | `scripts/compare_configs.py` | вручную | сравнение двух конфигураций по правилу остановки из `sample-size.json` |
| `dbt.json` | `scripts/build_mart.py` | `make mart` и работа витрины в `check.yml` | сборка витрины dbt в ClickHouse и сверка с бесплатным режимом |
| `memory.json` | `scripts/measure_memory.py` | вручную | память сервиса по режимам |
| `latency-hosted.json` | `scripts/measure_hosted.py --url https://ru-fin-rag.onrender.com` | вручную | задержка сервиса на хостинге |
| `render-rehearsal.json` | `scripts/rehearse_render.py` | вручную, нужен docker | репетиция лимита 512 МБ хостинга контейнером |
| `rerank-hybrid-rufin.jsonl`, `rerank-bm25-rufin.jsonl` | `scripts/rerank_run.py --out results/<имя>.jsonl` | вручную на видеокарте (там же, где одноименные `.json`) | переранжированные выдачи; их читает `plan_sample_size.py` |
| `inventory.json` | `scripts/inventory.py` | вручную | размеры открытых и закрытых каталогов части А в МиБ и КиБ, из них README берет вес выдач |
