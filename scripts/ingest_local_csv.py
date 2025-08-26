#!/usr/bin/env python3
"""Загрузка локальных CSV файлов (одноколоночный или 3-колоночный ток) в RawSignal.

Использование:
  python scripts/ingest_local_csv.py --path C:/data/signals --pattern "*.csv" --limit 20
  python scripts/ingest_local_csv.py file1.csv file2.csv

Шаги:
  1. Поиск файлов по списку путей или каталогу с шаблоном.
  2. Асинхронная загрузка через CSVLoader (батчами) в БД.
  3. Вывод сводной статистики и списка RawSignal id.

По умолчанию пытается подобрать Equipment автоматически (см. CSVLoader.find_equipment_by_filename).
"""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict

# Ensure project root is on sys.path when running from scripts/
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing.csv_loader import CSVLoader
from src.database.connection import get_async_session
from src.database.models import Equipment, EquipmentType, EquipmentStatus
from sqlalchemy import select, func, cast, String
import hashlib
from src.utils.logger import get_logger

logger = get_logger(__name__)


def collect_files(inputs: List[str], pattern: str | None, recursive: bool) -> List[Path]:
    files: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            glob_pat = pattern or '*.csv'
            if recursive:
                files.extend(p.rglob(glob_pat))
            else:
                files.extend(p.glob(glob_pat))
        else:
            logger.warning(f"Путь не найден: {inp}")
    # Удаляем дубликаты, сортируем по имени
    uniq = sorted({f.resolve() for f in files})
    return list(uniq)


async def _ensure_equipment_for_folder(folder: Path) -> Equipment:
    """Найти или создать Equipment для папки.

    Критерий поиска: specifications.source_folder == <abs-folder>.
    Если нет – создаём: name = "Двигатель N" (N = count+1), equipment_id = ENG-<sha1>.
    """
    folder_key = str(folder.resolve())
    async with get_async_session() as session:
        try:
            res = await session.execute(
                select(Equipment).where(Equipment.specifications.op('->>')('source_folder') == folder_key)
            )
        except Exception:
            res = await session.execute(
                select(Equipment).where(cast(Equipment.specifications, String).ilike(f'%"source_folder": "{folder_key}"%'))
            )
        eq = res.scalar_one_or_none()
        if eq:
            return eq

        sha = hashlib.sha1(folder_key.encode('utf-8')).hexdigest()[:10]
        total = (await session.execute(select(func.count()).select_from(Equipment))).scalar_one()
        index = int(total) + 1
        eq = Equipment(
            equipment_id=f"ENG-{sha}",
            name=f"Двигатель {index}",
            type=EquipmentType.INDUCTION_MOTOR,
            status=EquipmentStatus.ACTIVE,
            model='auto',
            location=folder_key,
            specifications={'source_folder': folder_key, 'engine_index': index}
        )
        session.add(eq)
        await session.commit()
        await session.refresh(eq)
        logger.info(f"Создано оборудование: {eq.name} для {folder_key}")
        return eq


async def ingest(files: List[Path], limit: int | None):
    """Ингест CSV, группируя файлы по папкам -> отдельные двигатели."""
    loader = CSVLoader()
    processed_total = 0
    raw_ids: List[str] = []

    # Группировка по родительским папкам
    groups: Dict[Path, List[Path]] = {}
    for f in files:
        groups.setdefault(f.parent.resolve(), []).append(f)

    folder_list = sorted(groups.keys())
    files_done = 0
    for folder in folder_list:
        eq = await _ensure_equipment_for_folder(folder)
        file_list = sorted(groups[folder])
        logger.info(f"Папка {folder} -> {eq.name} ({eq.equipment_id}); файлов: {len(file_list)}")
        for f in file_list:
            if limit and processed_total >= limit:
                break
            files_done += 1
            logger.info(f"[{files_done}] Загрузка {f}")
            stats = await loader.load_csv_file(f, equipment_id=eq.id)
            stats.finish()
            processed_total += stats.processed_rows
            raw_ids.extend([str(r) for r in stats.raw_signal_ids])
            logger.info(f"Файл {f.name}: строк={stats.processed_rows} raw_signals={len(stats.raw_signal_ids)}")
        if limit and processed_total >= limit:
            break

    print("\n=== ИНЖЕСТ ЗАВЕРШЁН ===")
    print(f"Папок: {len(folder_list)}; Файлов обработано: {files_done}")
    print(f"Всего строк: {processed_total}")
    print(f"RawSignal IDs: {', '.join(raw_ids) if raw_ids else '-'}")


def parse_args():
    ap = argparse.ArgumentParser(description="Загрузка локальных CSV файлов в систему")
    ap.add_argument('inputs', nargs='*', help='Файлы или директории')
    ap.add_argument('--path', help='Каталог с CSV')
    ap.add_argument('--pattern', default='*.csv', help='Шаблон поиска (по умолчанию *.csv)')
    ap.add_argument('--recursive', action='store_true', help='Рекурсивный поиск файлов')
    ap.add_argument('--limit', type=int, help='Ограничение количества строк (общий лимит)')
    return ap.parse_args()


def main():
    args = parse_args()
    inputs: List[str] = []
    if args.path:
        inputs.append(args.path)
    inputs.extend(args.inputs)
    if not inputs:
        print('Укажите хотя бы файл или директорию')
        return 1
    files = collect_files(inputs, args.pattern, args.recursive)
    if not files:
        print('Не найдено CSV файлов')
        return 2
    print(f'Найдено файлов: {len(files)}')
    try:
        asyncio.run(ingest(files, args.limit))
    except KeyboardInterrupt:
        print('\nПрервано пользователем')
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
