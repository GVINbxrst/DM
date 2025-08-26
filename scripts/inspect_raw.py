import asyncio
import os
from typing import Optional
from uuid import UUID

from sqlalchemy import select

from src.database.connection import get_async_session
from src.database.models import RawSignal, ProcessingStatus, Feature


def set_env():
    os.environ.setdefault('PYTHONPATH','.')
    os.environ.setdefault('APP_ENVIRONMENT','development')
    os.environ.setdefault('CELERY_TASK_ALWAYS_EAGER','1')
    os.environ.setdefault('DATABASE_URL','postgresql+asyncpg://diagmod_user:diagmod@localhost:5432/diagmod')


async def inspect(raw_id: Optional[str] = None):
    async with get_async_session() as session:
        if raw_id:
            rid = UUID(raw_id)
            raw = await session.get(RawSignal, rid)
        else:
            q = select(RawSignal).order_by(RawSignal.created_at.desc()).limit(1)
            raw = (await session.execute(q)).scalars().first()
        if not raw:
            print({'error': 'raw not found'})
            return
        # Count features for this raw
        cnt = (await session.execute(select(Feature).where(Feature.raw_id == raw.id))).scalars().all()
        print({
            'id': str(raw.id),
            'processed': bool(raw.processed),
            'processing_status': getattr(raw.processing_status, 'value', str(raw.processing_status)),
            'samples_count': int(raw.samples_count or 0),
            'features_for_raw': len(cnt),
            'meta_keys': list((raw.meta or {}).keys())
        })


if __name__ == '__main__':
    import sys
    set_env()
    rid = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(inspect(rid))
