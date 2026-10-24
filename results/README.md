# Выводы прогонов

Каждое число публичного текста лежит здесь, в файле, созданном кодом из этого репозитория.

| Файл | Чем создан | Что в нем |
|---|---|---|
| `artifacts.md` | руками по замерам, команды в тексте | инвентаризация артефактов части А под сервис |
| `reproducibility.json` | `scripts/check_reproducibility.py` | совпадение выдачи сервиса с выдачей бенчмарка части А |
| `vectors.json` | `scripts/extend_vectors.py` | догон плотных векторов до выросшего корпуса и сверка с матрицей части А по косинусу |
| `rerank-hybrid-rufin.json`, `rerank-bm25-rufin.json` | `scripts/rerank_run.py` | цена переранжирования на видеокарте: пары, время, скорость, пик видеопамяти |
| `rerank-quality.json`, `rerank-quality-bm25.json`, `rerank-first-stage.json` | `scripts/evaluate_run.py` | качество связок с кросс-энкодером и прибавки с парными интервалами |
| `latency-local.json` | `scripts/measure_latency.py` | задержка связок поиска на маке по ступеням |
| `vector-store.json` | `scripts/compare_vector_stores.py` | Qdrant против FAISS против numpy: задержка, память, совпадение выдач |
| `quantization.json` | `scripts/measure_quantization.py` | квантизация векторов в int8: память и потеря качества с интервалом |
| `load.json` | `scripts/measure_load.py` | нагрузочный замер: p50, p95, p99, RPS, узкое место |
| `warehouse.json` | `scripts/compare_warehouses.py` | витрина над журналом и сверка двух режимов хранения: ClickHouse сервером против файлов через chdb |
| `logging.json`, `logging.md` | `scripts/measure_logging.py` | чего стоит журнал: скорость записи, доля в задержке запроса, размер события, месяц по экстраполяции, поведение при отказе записи |
| `dashboard.json` | `scripts/measure_dashboard.py` | страница дашборда: чисел на ней, графиков, вес, сборка, поведение на пустом журнале, и список напечатанных чисел для пересборки |
| `regression.json`, `regression-baseline.json` | `scripts/run_regression.py` | качество на замороженном наборе и эталон, с которым его сравнивает каждый плановый прогон |
| `alert-threshold.json` | `scripts/measure_alert_threshold.py` | разброс метрики от прогона к прогону и от доливки корпуса, из него выведен порог алерта |
| `alert-drill.json` | `scripts/check_alert.py` | учебная деградация метрики: на какой потере алерт сработал и на какой промолчал |
| `dag-rerun.json`, `dag-rerun-ci.json` | `scripts/check_dag_rerun.py` | два прогона DAG подряд: время прохода, пик памяти, пропуск шагов на повторе. Второй файл это тот же замер в прогоне CI, взят из артефакта прогона |
| `dag-recovery.json` | `scripts/check_dag_recovery.py` | учебное падение шага в середине цепочки: попытки, незапущенные задачи, целость прошлого прогона, восстановление |
| `dag.md` | `scripts/report_dag.py` | три замера регламента одним отчетом: время прохода, акты за прогон, падение шага |
