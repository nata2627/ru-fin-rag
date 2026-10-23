# Выводы прогонов

Каждое число публичного текста лежит здесь, в файле, созданном кодом из этого репозитория.

| Файл | Чем создан | Что в нем |
|---|---|---|
| `artifacts.md` | руками по замерам, команды в тексте | инвентаризация артефактов части А под сервис |
| `reproducibility.json` | `scripts/check_reproducibility.py` | совпадение выдачи сервиса с выдачей бенчмарка части А |
| `latency-local.json` | `scripts/measure_latency.py` | задержка связок поиска на маке по ступеням |
| `vector-store.json` | `scripts/compare_vector_stores.py` | Qdrant против FAISS против numpy: задержка, память, совпадение выдач |
| `quantization.json` | `scripts/measure_quantization.py` | квантизация векторов в int8: память и потеря качества с интервалом |
| `load.json` | `scripts/measure_load.py` | нагрузочный замер: p50, p95, p99, RPS, узкое место |
| `warehouse.json` | `scripts/compare_warehouses.py` | витрина над журналом и сверка двух режимов хранения: ClickHouse сервером против файлов через chdb |
| `logging.json`, `logging.md` | `scripts/measure_logging.py` | чего стоит журнал: скорость записи, доля в задержке запроса, размер события, месяц по экстраполяции, поведение при отказе записи |
