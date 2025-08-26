#!/usr/bin/env python3
"""Аудит инжеста: показывает возможные дубликаты и статистику по батчам.

Usage (PowerShell):
  python -X utf8 scripts\audit_ingest.py
"""
import asyncio
from typing import Any

# Ensure project root is on sys.path when running from scripts/
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select, func

from src.database.connection import get_async_session
from src.database.models import RawSignal


async def main() -> int:
    async with get_async_session() as session:
        print("\nTop-10 по именам файлов (кол-во RawSignal записей, это и есть кол-во пачек):")
        res1 = await session.execute(
            select(RawSignal.file_name, func.count())
            .group_by(RawSignal.file_name)
            .order_by(func.count().desc())
            .limit(10)
        )
        for fname, cnt in res1.fetchall():
            print(f"  {fname or '-'}: {cnt}")

        print("\nДубликаты по file_hash (ожидается 1 на файл; >1 указывает на проблему):")
        res2 = await session.execute(
            select(RawSignal.file_hash, func.count())
            .where(RawSignal.file_hash.is_not(None))
            .group_by(RawSignal.file_hash)
            .having(func.count() > 1)
            .order_by(func.count().desc())
        )
        rows2 = res2.fetchall()
        if not rows2:
            print("  Не обнаружено")
        else:
            for h, cnt in rows2:
                print(f"  {h[:12]}…: {cnt}")

        print("\nСуммарное число RawSignal (все пачки):")
        res3 = await session.execute(select(func.count()).select_from(RawSignal))
        print(f"  {res3.scalar_one()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
