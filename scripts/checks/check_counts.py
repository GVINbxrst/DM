import asyncio
import os

from sqlalchemy import select, func

from src.database.connection import get_async_session
from src.database.models import RawSignal, Feature, Prediction


def env_default(key: str, default: str) -> None:
    if not os.getenv(key):
        os.environ[key] = default


async def main():
    # Ensure env for local run
    env_default('APP_ENVIRONMENT', 'development')
    env_default('CELERY_TASK_ALWAYS_EAGER', '1')
    env_default('DATABASE_URL', 'postgresql+asyncpg://diagmod_user:diagmod@localhost:5432/diagmod')

    async with get_async_session() as session:
        raw_count = (await session.execute(select(func.count()).select_from(RawSignal))).scalar_one()
        feat_count = (await session.execute(select(func.count()).select_from(Feature))).scalar_one()
        anom_count = (await session.execute(select(func.count()).select_from(Prediction))).scalar_one()

    print({
            'raw_signals': int(raw_count),
            'features': int(feat_count),
            'predictions': int(anom_count),
        })


if __name__ == '__main__':
    asyncio.run(main())
