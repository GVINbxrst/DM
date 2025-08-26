import asyncio
from typing import Dict

# Ensure project root is on sys.path when running from scripts/
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select, func

from src.database.connection import get_async_session  # type: ignore
from src.database.models import RawSignal, Feature, Prediction  # type: ignore


async def main():
    async with get_async_session() as session:  # type: ignore
        # Всего raw_signals
        total = (await session.execute(select(func.count()).select_from(RawSignal))).scalar_one()

        # Группировка по статусам обработки
        rows = (await session.execute(
            select(RawSignal.processing_status, func.count()).group_by(RawSignal.processing_status)
        )).all()
        # Нормализуем имена статусов: оставляем хвост после точки и приводим к верхнему регистру
        def norm(s: object) -> str:
            name = str(s)
            if '.' in name:
                name = name.split('.')[-1]
            return name.upper()
        by_status: Dict[str, int] = {norm(status): int(cnt) for status, cnt in rows}

        # Сколько ещё не завершено (всё, кроме COMPLETED)
        unprocessed_left = sum(cnt for status, cnt in by_status.items() if status != 'COMPLETED')

        # Признаки и оценки аномалий
        features_total = (await session.execute(select(func.count()).select_from(Feature))).scalar_one()
        anomaly_total = (await session.execute(select(func.count()).select_from(Prediction))).scalar_one()

        out = {
            'raw_signals_total': int(total),
            'raw_signals_by_status': by_status,
            'unprocessed_left': int(unprocessed_left),
            'features_total': int(features_total),
            'predictions_total': int(anomaly_total),
        }
        print(out)


if __name__ == '__main__':
    asyncio.run(main())
