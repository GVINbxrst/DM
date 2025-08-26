#!/usr/bin/env python
"""Очистка БД проекта: TRUNCATE основных таблиц с CASCADE.
Требует валидного DATABASE_URL. Используйте осторожно.
"""
import asyncio
import os
from sqlalchemy import text
from src.database.connection import engine
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Список таблиц в порядке, безопасном для TRUNCATE CASCADE (на случай, если CASCADE недоступен, всё равно порядок помогает)
TABLES = [
    'anomaly_scores',
    'forecasts',
    'stream_stats',
    'hourly_feature_summary',
    'predictions',
    'features',
    'raw_signals',
    'cluster_labels',
    'defect_catalog',
    'maintenance_events',
    'system_logs',
    'user_sessions',
    'users',
    'equipment',
    'defect_types'
]

async def reset_db():
    async with engine.begin() as conn:
        # Проверим, что это не production
        env = os.getenv('APP_ENVIRONMENT', 'development').lower()
        if env == 'production':
            raise RuntimeError('Нельзя выполнять reset_db в production среде')
        # Включаем/отключаем проверки внешних ключей опционально
        try:
            await conn.execute(text('SET session_replication_role = replica'))
        except Exception:
            pass
        try:
            for t in TABLES:
                try:
                    await conn.execute(text(f'TRUNCATE TABLE {t} RESTART IDENTITY CASCADE'))
                except Exception:
                    # Если TRUNCATE недоступен (например, SQLite в тестах), используем DELETE
                    await conn.execute(text(f'DELETE FROM {t}'))
            await conn.commit()
            logger.info('База данных очищена')
        finally:
            try:
                await conn.execute(text('SET session_replication_role = DEFAULT'))
            except Exception:
                pass

if __name__ == '__main__':
    asyncio.run(reset_db())
