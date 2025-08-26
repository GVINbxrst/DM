#!/usr/bin/env python
"""Пакетная детекция аномалий для Feature без Prediction.

Usage (PowerShell):
  python scripts/detect_anomalies_batch.py --limit 1000
"""
import asyncio
from typing import Optional, List

# Ensure project root is on sys.path when running from scripts/
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select, exists

from src.database.connection import get_async_session
from src.database.models import Feature, Prediction
from src.worker.tasks import _detect_anomalies_async
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def fetch_candidate_feature_ids(limit: Optional[int] = None) -> List[str]:
    async with get_async_session() as session:
        # Выбираем Feature, для которых ещё нет записей в Prediction
        subq = select(Prediction.feature_id).where(Prediction.feature_id == Feature.id)
        q = select(Feature.id).where(~exists(subq)).order_by(Feature.created_at.asc())
        if limit:
            q = q.limit(limit)
        rows = (await session.execute(q)).scalars().all()
        return [str(r) for r in rows]


async def main(limit: Optional[int] = None):
    ids = await fetch_candidate_feature_ids(limit)
    logger.info(f"Найдено {len(ids)} фичей для детекции аномалий")
    ok = 0
    for fid in ids:
        try:
            res = await _detect_anomalies_async(fid)
            if res.get('status') == 'success':
                ok += 1
        except Exception as e:
            logger.warning(f"Ошибка аномалий для {fid}: {e}")
    logger.info(f"Готово: успешно={ok} / всего={len(ids)}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    asyncio.run(main(args.limit))
