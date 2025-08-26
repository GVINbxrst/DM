# Частное техническое задание: Конвейер обработки сигналов

Версия: 1.2 • Дата: 2025‑08‑25 • Автор: системный аналитик • Статус: рабочее

## Оглавление
- Цель и область работ
- Контракт конвейера (вход/выход/ошибки/SLA)
- Данные и схемы (минимум, типы, индексы)
- Механика CLAIM/конкурентность
- Алгоритм обработки (псевдокод)
- Валидация и извлечение признаков
- Аномалии и прогноз
- Конфигурация и параметры производительности
- Мониторинг, логи, метрики
- Тест‑план и приёмка
- Роли и ответственность
- Риски, допущения, план релиза

## Цель и область работ
Реализовать устойчивый, идемпотентный, масштабируемый конвейер «загрузка → валидация → признаки → аномалии → прогноз», поддерживающий параллельное исполнение без дублей и гонок, с чёткими SLA и наблюдаемостью.

## Контракт конвейера
- Вход: RawSignal.id (UUID) с бинарями фаз R/S/T (сжатые float32) или путём к CSV; sample_rate; метаданные (equipment, file_name, file_hash, meta.batch_index).
- Выход: набор Feature для окон (id, raw_id, window_start, rms_a/b/c, …), опционально AnomalyScore и обновлённые HourlyFeatureSummary; RawSignal.processing_status финализирован.
- Ошибки: ValidationFailed, InsufficientData, CSVLoaderError → RawSignal.status=FAILED, причина в meta.
- SLA: обработка файла до ~1.5 млн точек/фаза ≤ 10 мин (референс); 1000 сигналов подряд ≤ согласованного SLA.

## Данные и схемы
- Уникальность: `raw_signals(file_hash)` (NULLable для батчей, только первая партия имеет hash), индекс по `(processing_status, created_at DESC)`.
- Feature: уникальность по `(raw_id, window_start)` логическим контролем (SELECT‑проверка перед insert) или уникальным индексом, если добавим.
- HourlyFeatureSummary: числовые поля float; при обновлении избегать Decimal/float конфликтов.

## Механика CLAIM/конкурентность
Стратегия S1 (упрощённая и совместимая):
```
UPDATE raw_signals
SET processing_status='processing'
WHERE id=:id AND processing_status='pending'
RETURNING id;
```
Если строка не вернулась — сигнал уже взят другим воркером; пропустить.

Стратегия S2 (предпочтительно при высокой конкуренции):
```
BEGIN;
SELECT id FROM raw_signals
WHERE processing_status='pending'
ORDER BY created_at
FOR UPDATE SKIP LOCKED
LIMIT :n;
-- далее массовый UPDATE этих id → processing
COMMIT;
```

## Алгоритм обработки (псевдокод)
```
for raw_id in pick_batch(limit=N):
  if not claim(raw_id): continue
  try:
    raw = load(raw_id)
    if empty(phases) and file_name: backfill_from_csv()
    phases = decompress()
    validate(phases, sample_rate)
    feats = extract_features(phases, window=1000ms, overlap=0.5)
    upsert_hourly_summary(feats)
    maybe_embed_and_scores(feats)
    mark_completed(raw_id)
  except ValidationError as e:
    mark_failed(raw_id, reason=e)
  except Exception as e:
    mark_failed(raw_id, reason='internal_error')
```

## Валидация и извлечение признаков
- Валидация: длина, NaN/Inf, частота дискретизации, соответствие заголовка CSV; протоколировать summary в `meta.validation`.
- Признаки: окна 1000 мс, overlap 0.5; RMS, mean/min/max, базовые спектральные; результаты float.
- Идемпотентность: перед вставкой Feature проверять существование `(raw_id, window_start)`; при совпадении — пропуск.

## Аномалии и прогноз
- Если доступен манифест «latest» (models/anomaly_detection/latest/manifest.json):
  - stream: обновление стох. модели и сохранение AnomalyScore (score, threshold, is_anomaly), meta.online=true.
  - stats: расчёт при detect‑проходе (вне онлайн‑части).
- Прогноз трендов: опционально, при наличии модели/конфига; ошибки — best‑effort, не валят конвейер.

## Конфигурация и параметры производительности
- pydantic‑settings (.env): `DATABASE_URL`, `APP_ENVIRONMENT`, `CSV_BATCH_SIZE`, `FEATURE_EXTRACTION_WORKERS`, `FEATURE_EXTRACTION_PARALLEL`, `RETAIN_RAW_SIGNALS`, `ANOMALY_THRESHOLD`.
- Рекомендации: уменьшить подробность логов SQLAlchemy до WARNING; batches по 1000; несколько процессов при наличии ресурсов.

## Мониторинг, логи, метрики
- Метрики: количество обработанных сигналов, ошибки, длительности (гистограммы), очередь ожидания; экспонирование через Prometheus.
- Логи: структурированные (JSON/текст), ключевые этапы конвейера; без избыточного SQL‑чата.
- Health: проверка БД; статус пайплайна через вспомогательный скрипт.

## Тест‑план и приёмка
- Набор CSV (включая большие файлы): отсутствие дублей `file_hash`; статусы PENDING→COMPLETED/FAILED корректны.
- Обработка ≥1000 сигналов: рост `features_total`, отсутствие падений.
- Повторный запуск и параллельный запуск (2 процесса/хоста): отсутствие дублей Feature/AnomalyScore.
- Миграции Alembic на чистой БД успешны.

## Роли и ответственность
- Backend инженер: реализация конвейера, идемпотентности и CLAIM.
- DS инженер: конфигурация/валидация признаков и моделей аномалий/прогноза.
- DevOps: настройки БД/пулов, мониторинг, резервирование.

## Риски, допущения, план релиза
- Риски: медленные диски и большие CSV → тюнинг batch/параллельности; гонки при высокой конкуренции → переход на SKIP LOCKED.
- Допущения: общая БД Postgres для всех инстансов; доступ к данным CSV при backfill.
- План релиза: внедрение S1 CLAIM → нагрузочный тест → при необходимости миграция на S2 SKIP LOCKED.
