"""Настройка async SQLAlchemy engine и фабрики сессий.

Стандартная схема без скрытого commit/rollback внутри контекста.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os
import sys

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from src.config.settings import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Подмена URL для тестового окружения если не переопределён пользователем.
db_url = settings.DATABASE_URL
if ('PYTEST_CURRENT_TEST' in os.environ or 'pytest' in sys.modules) and settings.APP_ENVIRONMENT != 'production':
    # Авто-подмена дефолтного Postgres на file-based SQLite (устойчиво между коннектами)
    if db_url.startswith('postgresql') and 'diagmod_user:diagmod_password' in db_url:
        db_url = 'sqlite+aiosqlite:///./test_db.sqlite'
        logger.info('Авто-подмена DATABASE_URL на file SQLite test_db.sqlite для тестов (вместо внешнего Postgres)')
    # Если явно указали :memory:, заменим на файл чтобы сохранить схему между соединениями aiosqlite
    elif db_url.endswith(':///:memory:'):
        db_url = 'sqlite+aiosqlite:///./test_db.sqlite'
        logger.info('Заменён sqlite in-memory на файл test_db.sqlite для устойчивости между соединениями')

# Engine: включаем pre-ping; для sqlite используем NullPool (устойчиво в тестах)
engine_kwargs = dict(
    echo=getattr(settings, 'APP_DEBUG', False),
    pool_pre_ping=True,
    future=True,
)
if db_url.startswith('sqlite'):
    engine_kwargs["poolclass"] = NullPool

engine = create_async_engine(db_url, **engine_kwargs)

_SCHEMA_READY = False

async def _ensure_schema():  # pragma: no cover - инфраструктурный слой
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    # Авто-создание схемы только для sqlite тестовой БД
    if db_url.startswith('sqlite'):  # не трогаем Postgres
        try:
            from src.database.models import Base  # локальный импорт чтобы избежать циклов
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            _SCHEMA_READY = True
            logger.info('Инициализирована тестовая схема БД (SQLite)')
        except Exception:  # pragma: no cover
            logger.exception('Не удалось создать схему БД')


# Session maker согласно контракту.
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Асинхронный контекст для работы с БД без скрытого коммита.

    Коммит/роллбек выполняются вызывающим кодом.
    """
    await _ensure_schema()
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            try:
                await session.close()
            except Exception:
                logger.debug("Session close failed", exc_info=True)

async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: async generator, отдаёт сессию без автокоммита."""
    await _ensure_schema()
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            try:
                await session.close()
            except Exception:
                logger.debug("Session close failed", exc_info=True)


async def check_connection() -> bool:
    """Проверить доступность соединения (SELECT 1)."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("DB connection test failed")
        return False


__all__ = [
    'engine', 'async_session_maker', 'get_async_session', 'db_session', 'check_connection'
]
